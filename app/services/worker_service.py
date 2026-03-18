"""
WorkerService — manages worker cameras, background pollers, and shift tracking.

Shift management is built directly into this service to avoid cross-service
dependencies when reading per-person stats from the tracker.

Shift lifecycle:
  start_shift(camera_id, worker_name)
    → snapshots baseline tracker stats
    → stores ShiftRecord in memory

  (background poller) → process_frame() → new_events → stored on active shift

  end_shift(camera_id)
    → snapshots final tracker stats
    → calculates productivity (delta = final - baseline)
    → marks shift as ended, returns ProductivityReport
"""

import cv2
import uuid
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

from vision.camera_manager import CameraManager
from vision.worker_tracker import WorkerTracker, WorkerStatusResult, EventRecord
from app.models.worker_model import WorkerCameraConfig, ZoneConfig

logger = logging.getLogger(__name__)

POLL_INTERVAL = 0.5  # seconds (~2 fps)


# ─── Shift data (in-memory) ───────────────────────────────────────────────────

@dataclass
class ShiftRecord:
    shift_id:       str
    worker_name:    str
    camera_id:      str
    started_at:     datetime
    notes:          str
    baseline_stats: List[dict]          # tracker stats snapshot at shift start
    events:         List[dict] = field(default_factory=list)
    ended_at:       Optional[datetime] = None
    final_stats:    Optional[List[dict]] = None
    is_active:      bool = True


# ─── WorkerService ────────────────────────────────────────────────────────────

class WorkerService:

    def __init__(self):
        logger.info("Starting WorkerService...")
        self._cameras:  Dict[str, CameraManager]   = {}
        self._trackers: Dict[str, WorkerTracker]   = {}
        self._latest:   Dict[str, dict]             = {}    # {result, polled_at}
        self._pollers:  Dict[str, threading.Thread] = {}
        self._running:  Dict[str, bool]             = {}

        # Shift management
        self._shifts:        Dict[str, ShiftRecord] = {}  # shift_id → record
        self._active_shifts: Dict[str, str]         = {}  # camera_id → shift_id

        # Zone config (camera_id → ZoneConfig)
        self._zones: Dict[str, ZoneConfig] = {}

        logger.info("WorkerService ready.")

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def register_camera(self, config: WorkerCameraConfig) -> None:
        camera_id = config.camera_id

        if camera_id in self._cameras:
            self._running[camera_id] = False
            if camera_id in self._pollers:
                self._pollers[camera_id].join(timeout=3)
            self._cameras[camera_id].stop()

        parsed_source = int(config.source) if config.source.isdigit() else config.source
        cam = CameraManager(source=parsed_source, camera_id=camera_id, loop=config.loop)
        cam.start()
        self._cameras[camera_id] = cam

        tracker = WorkerTracker(
            min_confidence      = config.min_confidence,
            active_threshold    = config.active_threshold,
            absent_timeout      = config.absent_timeout,
            idle_timeout        = config.idle_timeout,
            maintenance_alert_s = config.maintenance_alert_s,
        )
        self._trackers[camera_id] = tracker
        self._latest.pop(camera_id, None)

        # Restore saved rack zone if it exists
        saved_zones = self._zones.get(camera_id)
        if saved_zones and saved_zones.rack_zone:
            tracker.set_rack_zone(saved_zones.rack_zone.model_dump())

        self._running[camera_id] = True
        t = threading.Thread(
            target=self._poll_loop,
            args=(camera_id,),
            daemon=True,
            name=f"worker-poller-{camera_id}",
        )
        t.start()
        self._pollers[camera_id] = t
        logger.info(f"Worker camera '{camera_id}' registered.")

    def unregister_camera(self, camera_id: str) -> bool:
        if camera_id not in self._cameras:
            return False
        # End any active shift first
        if camera_id in self._active_shifts:
            self.end_shift(camera_id)

        self._running[camera_id] = False
        if camera_id in self._pollers:
            self._pollers[camera_id].join(timeout=3)
        self._cameras[camera_id].stop()
        del self._cameras[camera_id]
        del self._trackers[camera_id]
        self._latest.pop(camera_id, None)
        self._pollers.pop(camera_id, None)
        self._running.pop(camera_id, None)
        logger.info(f"Worker camera '{camera_id}' removed.")
        return True

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_latest_status(self, camera_id: str) -> Optional[dict]:
        return self._latest.get(camera_id)

    def get_event_log(self, camera_id: str) -> Optional[List[EventRecord]]:
        tracker = self._trackers.get(camera_id)
        return tracker.get_event_log() if tracker else None

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        cam = self._cameras.get(camera_id)
        if not cam:
            return None
        latest = self._latest.get(camera_id)
        result = latest["result"] if latest else None
        return {
            "camera_id":    camera_id,
            "source":       str(cam.source),
            "connected":    cam.is_connected(),
            "has_frame":    cam.get_frame() is not None,
            "current_state": result.persons[0].state if result and result.persons else None,
            "person_count":  result.total_persons if result else 0,
        }

    def list_cameras(self) -> list:
        return [self.get_camera_status(cid) for cid in self._cameras]

    def get_snapshot_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Raw frame JPEG with no annotations — used by zone-setup UI."""
        cam = self._cameras.get(camera_id)
        if not cam:
            return None
        frame = cam.get_frame()
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()

    def set_zones(self, camera_id: str, zones: ZoneConfig) -> None:
        self._zones[camera_id] = zones
        # Push rack zone into live tracker immediately
        tracker = self._trackers.get(camera_id)
        if tracker:
            rack = zones.rack_zone
            tracker.set_rack_zone(rack.model_dump() if rack else None)
        logger.info(
            f"Zones saved for '{camera_id}': "
            f"rack={'set' if zones.rack_zone else 'none'}"
        )

    def get_zones(self, camera_id: str) -> ZoneConfig:
        return self._zones.get(camera_id, ZoneConfig())

    def get_debug_image(self, camera_id: str) -> Optional[bytes]:
        cam     = self._cameras.get(camera_id)
        tracker = self._trackers.get(camera_id)
        if not cam or not tracker:
            return None
        frame = cam.get_frame()
        if frame is None:
            return None
        latest = self._latest.get(camera_id)
        result = latest["result"] if latest else None
        debug_frame = tracker.get_debug_frame(frame, result=result)
        _, buffer = cv2.imencode(".jpg", debug_frame)
        return buffer.tobytes()

    # ------------------------------------------------------------------
    # Shift management
    # ------------------------------------------------------------------

    def start_shift(self, camera_id: str, worker_name: str, notes: str = "") -> ShiftRecord:
        """Start a new shift. Snapshots current tracker stats as baseline."""
        if camera_id not in self._cameras:
            raise ValueError(f"Camera '{camera_id}' not registered.")

        # End previous active shift on this camera if any
        if camera_id in self._active_shifts:
            self.end_shift(camera_id)

        tracker  = self._trackers[camera_id]
        baseline = tracker.get_person_stats()

        shift = ShiftRecord(
            shift_id       = str(uuid.uuid4())[:8],
            worker_name    = worker_name,
            camera_id      = camera_id,
            started_at     = datetime.now(),
            notes          = notes,
            baseline_stats = baseline,
        )
        self._shifts[shift.shift_id]       = shift
        self._active_shifts[camera_id]     = shift.shift_id
        logger.info(f"Shift {shift.shift_id} started — worker: {worker_name}")
        return shift

    def end_shift(self, camera_id: str) -> Optional[dict]:
        """End active shift, compute productivity report. Returns report dict or None."""
        shift_id = self._active_shifts.pop(camera_id, None)
        if not shift_id:
            return None

        shift = self._shifts.get(shift_id)
        if not shift:
            return None

        tracker       = self._trackers.get(camera_id)
        final_stats   = tracker.get_person_stats() if tracker else []
        shift.ended_at    = datetime.now()
        shift.final_stats = final_stats
        shift.is_active   = False

        report = self._calculate_productivity(shift)
        logger.info(
            f"Shift {shift_id} ended — cycles: {report['total_cycles']}, "
            f"sheets/hr: {report['sheets_per_hour']}, eff: {report['efficiency_pct']}%"
        )
        return report

    def get_active_shift(self, camera_id: str) -> Optional[ShiftRecord]:
        shift_id = self._active_shifts.get(camera_id)
        return self._shifts.get(shift_id) if shift_id else None

    def get_shift(self, shift_id: str) -> Optional[ShiftRecord]:
        return self._shifts.get(shift_id)

    def list_shifts(self, camera_id: Optional[str] = None) -> List[ShiftRecord]:
        shifts = list(self._shifts.values())
        if camera_id:
            shifts = [s for s in shifts if s.camera_id == camera_id]
        return sorted(shifts, key=lambda s: s.started_at, reverse=True)

    def get_shift_events(self, shift_id: str) -> Optional[List[dict]]:
        shift = self._shifts.get(shift_id)
        return shift.events if shift else None

    def get_productivity_report(self, shift_id: str) -> Optional[dict]:
        shift = self._shifts.get(shift_id)
        if not shift or shift.is_active:
            return None
        return self._calculate_productivity(shift)

    def get_live_productivity(self, camera_id: str) -> Optional[dict]:
        """Real-time productivity for the active shift (uses current tracker stats)."""
        shift = self.get_active_shift(camera_id)
        if not shift:
            return None
        tracker = self._trackers.get(camera_id)
        if not tracker:
            return None
        current_stats = tracker.get_person_stats()
        return self._calculate_live(shift, current_stats)

    # ------------------------------------------------------------------
    # Background poller
    # ------------------------------------------------------------------

    def _poll_loop(self, camera_id: str) -> None:
        logger.info(f"[{camera_id}] Worker poller started.")
        while self._running.get(camera_id, False):
            cam     = self._cameras.get(camera_id)
            tracker = self._trackers.get(camera_id)
            if not cam or not tracker:
                break
            frame = cam.get_frame()
            if frame is not None:
                try:
                    result = tracker.process_frame(frame)
                    now    = datetime.now()
                    self._latest[camera_id] = {"result": result, "polled_at": now}

                    # Store events in active shift
                    if result.new_events:
                        shift_id = self._active_shifts.get(camera_id)
                        if shift_id and shift_id in self._shifts:
                            shift = self._shifts[shift_id]
                            for ev in result.new_events:
                                shift.events.append(ev.to_dict())
                            logger.debug(
                                f"[{camera_id}] {len(result.new_events)} events → shift {shift_id}"
                            )
                except Exception as e:
                    logger.error(f"[{camera_id}] Worker poll error: {e}")
            time.sleep(POLL_INTERVAL)
        logger.info(f"[{camera_id}] Worker poller stopped.")

    # ------------------------------------------------------------------
    # Productivity calculation
    # ------------------------------------------------------------------

    def _delta_stats(self, baseline: List[dict], current: List[dict]) -> List[dict]:
        """
        Compute per-person stats relative to baseline (shift start).
        Persons who appeared after shift start have baseline = 0.
        """
        base_map = {s["track_id"]: s for s in baseline}
        result   = []
        for stat in current:
            base = base_map.get(stat["track_id"], {})
            result.append({
                "label":          stat["label"],
                "active_s":       max(0.0, stat["total_active_s"]  - base.get("total_active_s",  0.0)),
                "idle_s":         max(0.0, stat["total_idle_s"]    - base.get("total_idle_s",    0.0)),
                "bending_s":      max(0.0, stat["total_bending_s"] - base.get("total_bending_s", 0.0)),
                "bending_events": max(0,   stat["bending_events"]  - base.get("bending_events",  0)),
                "sheet_picks":    max(0,   stat["sheet_picks"]     - base.get("sheet_picks",     0)),
            })
        return result

    def _calculate_live(self, shift: ShiftRecord, current_stats: List[dict]) -> dict:
        duration_s = (datetime.now() - shift.started_at).total_seconds()
        deltas     = self._delta_stats(shift.baseline_stats, current_stats)
        return self._compute_metrics(deltas, duration_s)

    def _calculate_productivity(self, shift: ShiftRecord) -> dict:
        duration_s = (
            (shift.ended_at - shift.started_at).total_seconds()
            if shift.ended_at else
            (datetime.now() - shift.started_at).total_seconds()
        )
        final   = shift.final_stats or []
        deltas  = self._delta_stats(shift.baseline_stats, final)
        metrics = self._compute_metrics(deltas, duration_s)

        hours, rem = divmod(int(duration_s), 3600)
        mins       = rem // 60
        dur_fmt    = f"{hours}h {mins}m" if hours else f"{mins}m"

        persons = [
            {
                "person_label":     d["label"],
                "active_s":         round(d["active_s"], 1),
                "idle_s":           round(d["idle_s"], 1),
                "bending_s":        round(d["bending_s"], 1),
                "bending_events":   d["bending_events"],
                "sheet_picks":      d["sheet_picks"],
                "estimated_cycles": d["sheet_picks"],   # 1 sheet pick = 1 cycle
            }
            for d in deltas
        ]

        return {
            "shift_id":        shift.shift_id,
            "worker_name":     shift.worker_name,
            "camera_id":       shift.camera_id,
            "started_at":      shift.started_at.isoformat(),
            "ended_at":        shift.ended_at.isoformat() if shift.ended_at else None,
            "duration_s":      round(duration_s, 1),
            "duration_fmt":    dur_fmt,
            "total_cycles":    metrics["total_cycles"],
            "sheets_per_hour": metrics["sheets_per_hour"],
            "efficiency_pct":  metrics["efficiency_pct"],
            "persons":         persons,
        }

    @staticmethod
    def _compute_metrics(deltas: List[dict], duration_s: float) -> dict:
        total_cycles  = sum(d.get("sheet_picks", 0) for d in deltas)  # 1 pick = 1 cycle
        total_active  = sum(d["active_s"]           for d in deltas)
        total_idle    = sum(d["idle_s"]              for d in deltas)

        hours           = max(duration_s / 3600, 0.001)
        sheets_per_hour = round(total_cycles / hours, 1)

        tracked_s   = total_active + total_idle
        efficiency  = round((total_active / max(tracked_s, 1)) * 100, 1) if tracked_s > 0 else 0.0

        return {
            "total_cycles":    total_cycles,
            "sheets_per_hour": sheets_per_hour,
            "efficiency_pct":  efficiency,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down WorkerService...")
        for camera_id in list(self._cameras.keys()):
            self.unregister_camera(camera_id)
        logger.info("WorkerService shut down.")
