"""
Downtime Detection Service — Phase 3 Level 2

Monitors the OCR counter for a camera. If the counter value stops
changing for STOPPED_THRESHOLD_S seconds → machine is stopped → downtime event starts.
When the counter starts moving again → downtime event ends and is logged.

Level 2 adds stop classification by cross-referencing the worker tracker:
  - unattended    : no confirmed worker in frame
  - plate_change  : worker present + active, stop < 10 min
  - maintenance   : worker present + active, stop >= 10 min
  - breakdown     : worker present + idle, stop >= 10 min
  - investigating : worker present + idle, stop < 10 min
  - planned_stop  : no active shift at time of stop
  - unknown       : worker camera not available

Usage:
    service = DowntimeService(ocr_service, worker_service)
    service.watch("machine_a_display", worker_camera_id="machine_a_worker")
    service.unwatch("machine_a_display")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from app.models.downtime_model import DowntimeRecord, DowntimeStatus, DowntimeSummary

logger = logging.getLogger(__name__)

# Seconds with no counter change before we declare machine stopped
# Set low for testing — change to 60 for real factory use
STOPPED_THRESHOLD_S = 15

# How often the downtime monitor checks the OCR cache (seconds)
POLL_INTERVAL_S = 5


BREAKDOWN_THRESHOLD_S = 600   # 10 minutes — long stop with idle worker = breakdown


@dataclass
class _CameraWatch:
    camera_id:         str
    worker_camera_id:  Optional[str]      = None   # Camera 2 ID for cross-reference
    last_value:        Optional[int]      = None   # last counter value seen
    last_changed_at:   Optional[datetime] = None   # when counter last changed
    active_event:      Optional[DowntimeRecord] = None


class DowntimeService:

    def __init__(self, ocr_service, worker_service=None):
        self._ocr     = ocr_service
        self._worker  = worker_service   # optional — for Level 2 classification
        self._watches: Dict[str, _CameraWatch] = {}
        self._events: List[DowntimeRecord]     = []
        self._counter = 0
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("DowntimeService ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def watch(self, camera_id: str, worker_camera_id: Optional[str] = None) -> None:
        """Start monitoring an OCR camera for downtime."""
        with self._lock:
            if camera_id not in self._watches:
                self._watches[camera_id] = _CameraWatch(
                    camera_id=camera_id,
                    worker_camera_id=worker_camera_id,
                )
                logger.info(f"DowntimeService: watching '{camera_id}' (worker cam: {worker_camera_id})")

    def unwatch(self, camera_id: str) -> None:
        """Stop monitoring a camera."""
        with self._lock:
            self._watches.pop(camera_id, None)
            logger.info(f"DowntimeService: stopped watching '{camera_id}'")

    def get_status(self, camera_id: str) -> Optional[DowntimeStatus]:
        """Current machine running/stopped status for a camera."""
        with self._lock:
            watch = self._watches.get(camera_id)
            if watch is None:
                return None
            stopped_secs = 0.0
            if watch.active_event:
                stopped_secs = (datetime.now() - watch.active_event.started_at).total_seconds()
            return DowntimeStatus(
                camera_id       = camera_id,
                machine_running = watch.active_event is None,
                stopped_since   = watch.active_event.started_at if watch.active_event else None,
                stopped_secs    = round(stopped_secs, 1),
                last_counter    = watch.last_value,
                last_seen_at    = watch.last_changed_at,
            )

    def get_events(self, camera_id: str) -> List[DowntimeRecord]:
        """All downtime events (resolved + active) for a camera."""
        with self._lock:
            return [e for e in self._events if e.camera_id == camera_id]

    def get_summary(self, camera_id: str) -> DowntimeSummary:
        """Downtime totals for a camera."""
        with self._lock:
            events   = [e for e in self._events if e.camera_id == camera_id]
            watch    = self._watches.get(camera_id)
            resolved = [e for e in events if e.status == "resolved" and e.duration_s is not None]
            total_s  = sum(e.duration_s for e in resolved)
            longest  = max((e.duration_s for e in resolved), default=0.0)
            active   = watch.active_event if watch else None
            return DowntimeSummary(
                camera_id        = camera_id,
                total_events     = len(resolved),
                total_downtime_s = round(total_s, 1),
                longest_event_s  = round(longest, 1),
                active_event     = active,
            )

    def shutdown(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    camera_ids = list(self._watches.keys())
                for cid in camera_ids:
                    self._check_camera(cid)
            except Exception:
                logger.exception("DowntimeService poll error")
            time.sleep(POLL_INTERVAL_S)

    def _check_camera(self, camera_id: str) -> None:
        """Check one camera's counter and update downtime state."""
        # Get latest OCR reading from cache (no new OCR run)
        latest = self._ocr._latest_valid.get(camera_id)
        if latest is None:
            return   # camera not yet producing readings

        result    = latest.get("result")
        polled_at = latest.get("polled_at")
        if result is None or result.value is None:
            return

        current_value = result.value
        now = datetime.now()

        with self._lock:
            watch = self._watches.get(camera_id)
            if watch is None:
                return

            # First reading — just record it
            if watch.last_value is None:
                watch.last_value      = current_value
                watch.last_changed_at = now
                return

            # Counter changed → machine is running
            if current_value != watch.last_value:
                watch.last_value      = current_value
                watch.last_changed_at = now
                # If there was an active downtime → end it
                if watch.active_event:
                    self._end_downtime(watch, now)
                return

            # Counter unchanged — check how long
            if watch.last_changed_at is None:
                watch.last_changed_at = now
                return

            stale_s = (now - watch.last_changed_at).total_seconds()

            if stale_s >= STOPPED_THRESHOLD_S and watch.active_event is None:
                # Machine just stopped — start a downtime event
                stopped_since = watch.last_changed_at
                self._start_downtime(watch, stopped_since)
            elif watch.active_event is not None:
                # Already stopped — re-classify reason as more data comes in
                self._update_classification(watch, now)

    def _get_worker_presence(self, worker_camera_id: Optional[str]):
        """Check Camera 2 for worker presence and state. Returns (present, state)."""
        if self._worker is None or worker_camera_id is None:
            return None, None
        try:
            latest = self._worker.get_latest_status(worker_camera_id)
            if latest is None:
                return None, None
            result = latest.get("result")
            if result is None:
                return False, None
            # Any confirmed worker in frame?
            persons = [p for p in result.persons if p.confirmed_worker]
            if not persons:
                return False, None
            # Dominant state among workers
            states = [p.state for p in persons]
            state = "active" if "active" in states else "idle"
            return True, state
        except Exception:
            return None, None

    def _classify_reason(self, duration_s: float, worker_present, worker_state,
                         shift_active: bool) -> str:
        """Classify stop reason from available signals."""
        if not shift_active:
            return "planned_stop"
        if worker_present is None:
            return "unknown"        # no worker camera data
        if not worker_present:
            return "unattended"
        # Worker is present
        if worker_state == "active":
            return "plate_change" if duration_s < BREAKDOWN_THRESHOLD_S else "maintenance"
        # Worker idle
        return "investigating" if duration_s < BREAKDOWN_THRESHOLD_S else "breakdown"

    def _shift_active(self, worker_camera_id: Optional[str]) -> bool:
        """Check if a shift is currently active on the worker camera."""
        if self._worker is None or worker_camera_id is None:
            return True   # assume shift active if no data
        try:
            return getattr(self._worker, "_active_shift", {}).get(worker_camera_id) is not None
        except Exception:
            return True

    def _start_downtime(self, watch: _CameraWatch, stopped_since: datetime) -> None:
        self._counter += 1
        worker_present, worker_state = self._get_worker_presence(watch.worker_camera_id)
        shift_active = self._shift_active(watch.worker_camera_id)
        reason = self._classify_reason(0, worker_present, worker_state, shift_active)
        event = DowntimeRecord(
            event_id       = self._counter,
            camera_id      = watch.camera_id,
            started_at     = stopped_since,
            status         = "active",
            reason         = reason,
            worker_present = worker_present,
            worker_state   = worker_state,
        )
        watch.active_event = event
        self._events.append(event)
        logger.warning(
            f"[{watch.camera_id}] MACHINE STOPPED — event #{event.event_id} "
            f"reason={reason} worker_present={worker_present} worker_state={worker_state}"
        )

    def _update_classification(self, watch: _CameraWatch, now: datetime) -> None:
        """Re-classify the active event as more time passes."""
        event = watch.active_event
        if event is None:
            return
        duration_s = (now - event.started_at).total_seconds()
        worker_present, worker_state = self._get_worker_presence(watch.worker_camera_id)
        shift_active = self._shift_active(watch.worker_camera_id)
        new_reason = self._classify_reason(duration_s, worker_present, worker_state, shift_active)
        if new_reason != event.reason:
            logger.info(f"[{watch.camera_id}] Stop reason updated: {event.reason} → {new_reason}")
        event.reason         = new_reason
        event.worker_present = worker_present
        event.worker_state   = worker_state

    def _end_downtime(self, watch: _CameraWatch, now: datetime) -> None:
        event = watch.active_event
        if event is None:
            return
        event.ended_at   = now
        event.duration_s = round((now - event.started_at).total_seconds(), 1)
        event.status     = "resolved"
        # Final classification with full duration
        worker_present, worker_state = self._get_worker_presence(watch.worker_camera_id)
        shift_active = self._shift_active(watch.worker_camera_id)
        event.reason = self._classify_reason(event.duration_s, worker_present, worker_state, shift_active)
        watch.active_event = None
        logger.info(
            f"[{watch.camera_id}] Machine resumed — downtime #{event.event_id} "
            f"duration={event.duration_s}s reason={event.reason}"
        )
