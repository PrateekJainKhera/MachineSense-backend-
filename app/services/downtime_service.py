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


def _fmt_s(s: float) -> str:
    """Format seconds as human-readable string."""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:   return f"{h}h {m}m"
    if m > 0:   return f"{m}m {sec}s"
    return f"{sec}s"


BREAKDOWN_THRESHOLD_S  = 600   # 10 minutes — long stop with idle worker = breakdown
UNATTENDED_THRESHOLD_S = 30    # machine running but no worker for this long = alert


@dataclass
class _CameraWatch:
    camera_id:            str
    worker_camera_id:     Optional[str]      = None
    last_value:           Optional[int]      = None
    last_changed_at:      Optional[datetime] = None
    active_event:         Optional[DowntimeRecord] = None
    unattended_since:     Optional[datetime] = None   # when worker disappeared while running


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
            now = datetime.now()
            stopped_secs = 0.0
            if watch.active_event:
                stopped_secs = (now - watch.active_event.started_at).total_seconds()
            unattended_secs = 0.0
            if watch.unattended_since:
                unattended_secs = (now - watch.unattended_since).total_seconds()
            return DowntimeStatus(
                camera_id         = camera_id,
                machine_running   = watch.active_event is None,
                stopped_since     = watch.active_event.started_at if watch.active_event else None,
                stopped_secs      = round(stopped_secs, 1),
                last_counter      = watch.last_value,
                last_seen_at      = watch.last_changed_at,
                unattended_alert  = watch.unattended_since is not None and unattended_secs >= UNATTENDED_THRESHOLD_S,
                unattended_secs   = round(unattended_secs, 1),
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

    def get_oee(self, camera_id: str, worker_camera_id: Optional[str] = None):
        """Calculate OEE Availability for the current active shift window."""
        from app.models.downtime_model import OEEResult
        now = datetime.now()

        # Get shift start time from worker service
        shift_start = None
        if self._worker and worker_camera_id:
            try:
                shift = self._worker.get_active_shift(worker_camera_id)
                if shift:
                    shift_start = shift.started_at
            except Exception:
                pass

        if shift_start is None:
            # No shift — use all recorded events
            shift_start = min((e.started_at for e in self._events if e.camera_id == camera_id), default=now)

        shift_duration_s = max((now - shift_start).total_seconds(), 1)

        # Sum downtime events that overlap with shift window
        with self._lock:
            events = [e for e in self._events if e.camera_id == camera_id]

        downtime_s = 0.0
        for e in events:
            start = max(e.started_at, shift_start)
            end   = e.ended_at if e.ended_at else now
            overlap = (end - start).total_seconds()
            if overlap > 0:
                downtime_s += overlap

        uptime_s         = max(shift_duration_s - downtime_s, 0)
        availability_pct = round((uptime_s / shift_duration_s) * 100, 1)

        return OEEResult(
            camera_id        = camera_id,
            shift_duration_s = round(shift_duration_s, 1),
            downtime_s       = round(downtime_s, 1),
            uptime_s         = round(uptime_s, 1),
            availability_pct = availability_pct,
        )

    def get_history(self, camera_id: str):
        """Level 3 — historical statistics across all resolved events."""
        from app.models.downtime_model import HistoryStats
        with self._lock:
            resolved = [e for e in self._events
                        if e.camera_id == camera_id and e.status == "resolved" and e.duration_s is not None]

        if not resolved:
            return HistoryStats(
                camera_id=camera_id, total_stops=0, total_downtime_s=0,
                avg_duration_s=0, longest_s=0, most_common_reason="none",
                reasons_breakdown={}, avg_interval_s=0, last_stop_ago_s=None,
            )

        total_s      = sum(e.duration_s for e in resolved)
        avg_s        = total_s / len(resolved)
        longest      = max(e.duration_s for e in resolved)

        reasons: dict = {}
        for e in resolved:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        most_common = max(reasons, key=reasons.get)

        # Avg interval between stop starts
        sorted_events = sorted(resolved, key=lambda e: e.started_at)
        intervals = []
        for i in range(1, len(sorted_events)):
            gap = (sorted_events[i].started_at - sorted_events[i-1].started_at).total_seconds()
            intervals.append(gap)
        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        last_stop_ago = None
        if sorted_events:
            last_end = sorted_events[-1].ended_at
            if last_end:
                last_stop_ago = round((datetime.now() - last_end).total_seconds(), 1)

        return HistoryStats(
            camera_id          = camera_id,
            total_stops        = len(resolved),
            total_downtime_s   = round(total_s, 1),
            avg_duration_s     = round(avg_s, 1),
            longest_s          = round(longest, 1),
            most_common_reason = most_common,
            reasons_breakdown  = reasons,
            avg_interval_s     = round(avg_interval, 1),
            last_stop_ago_s    = last_stop_ago,
        )

    def get_pattern(self, camera_id: str):
        """Level 3 — detect abnormal stop patterns."""
        from app.models.downtime_model import PatternInsight
        stats = self.get_history(camera_id)

        if stats.total_stops < 2:
            return PatternInsight(
                camera_id=camera_id, avg_interval_s=0,
                last_stop_ago_s=stats.last_stop_ago_s,
                overdue=False, high_frequency=False,
                message="Not enough data yet — need at least 2 stops for pattern analysis.",
            )

        avg      = stats.avg_interval_s
        last_ago = stats.last_stop_ago_s or 0
        overdue  = last_ago > avg * 1.5 if avg > 0 else False

        # Check last 3 intervals vs average
        with self._lock:
            resolved = sorted(
                [e for e in self._events if e.camera_id == camera_id and e.status == "resolved"],
                key=lambda e: e.started_at
            )
        recent_intervals = []
        for i in range(max(0, len(resolved) - 3), len(resolved)):
            if i == 0:
                continue
            gap = (resolved[i].started_at - resolved[i-1].started_at).total_seconds()
            recent_intervals.append(gap)
        recent_avg     = sum(recent_intervals) / len(recent_intervals) if recent_intervals else avg
        high_frequency = recent_avg < avg * 0.6 if avg > 0 else False

        if high_frequency:
            msg = f"⚠ Stops more frequent than usual — avg interval {_fmt_s(avg)}, recently {_fmt_s(recent_avg)}"
        elif overdue:
            msg = f"Machine overdue for a stop — avg every {_fmt_s(avg)}, last was {_fmt_s(last_ago)} ago"
        else:
            msg = f"Normal pattern — machine stops every ~{_fmt_s(avg)} on average"

        return PatternInsight(
            camera_id       = camera_id,
            avg_interval_s  = round(avg, 1),
            last_stop_ago_s = stats.last_stop_ago_s,
            overdue         = overdue,
            high_frequency  = high_frequency,
            message         = msg,
        )

    def get_csv(self, camera_id: str) -> str:
        """Level 3 — export all downtime events as CSV string."""
        with self._lock:
            events = [e for e in self._events if e.camera_id == camera_id]
        lines = ["#,Started,Ended,Duration(s),Status,Reason,Worker Present,Worker State"]
        for e in sorted(events, key=lambda x: x.started_at):
            lines.append(",".join([
                str(e.event_id),
                e.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                e.ended_at.strftime("%Y-%m-%d %H:%M:%S") if e.ended_at else "",
                str(e.duration_s or ""),
                e.status,
                e.reason,
                str(e.worker_present) if e.worker_present is not None else "",
                e.worker_state or "",
            ]))
        return "\n".join(lines)

    def override_reason(self, event_id: int, reason: str) -> bool:
        """Manually override the classified reason for a downtime event."""
        with self._lock:
            for event in self._events:
                if event.event_id == event_id:
                    event.reason = reason
                    logger.info(f"Downtime #{event_id} reason manually set to '{reason}'")
                    return True
        return False

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
                # Machine is running — check worker presence for unattended alert
                self._check_unattended(watch, now)
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

    def _check_unattended(self, watch: _CameraWatch, now: datetime) -> None:
        """Check if machine is running but no worker present — set/clear unattended alert."""
        if watch.worker_camera_id is None or self._worker is None:
            return
        worker_present, _ = self._get_worker_presence(watch.worker_camera_id)
        if worker_present is None:
            return   # no data from worker camera
        if not worker_present:
            if watch.unattended_since is None:
                watch.unattended_since = now
                logger.warning(f"[{watch.camera_id}] Machine running but no worker detected — unattended timer started")
        else:
            if watch.unattended_since is not None:
                watch.unattended_since = None
                logger.info(f"[{watch.camera_id}] Worker returned — unattended alert cleared")

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
            return worker_camera_id in self._worker._active_shifts
        except Exception:
            return True

    def _start_downtime(self, watch: _CameraWatch, stopped_since: datetime) -> None:
        watch.unattended_since = None   # machine stopped — clear unattended alert
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
