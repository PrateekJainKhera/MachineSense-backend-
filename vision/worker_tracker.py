"""
Worker Tracker — Multi-Person Phase 2
======================================
Uses yolov8n-pose.pt + ByteTrack to track multiple workers simultaneously.

Each detected person gets:
  - Persistent track ID from ByteTrack (stable across frames)
  - Independent state machine: ABSENT / IDLE / ACTIVE / BENDING
  - Pose classification from keypoints: standing / bending
  - Cumulative timing stats for productivity calculation

Events (emitted per person on state transitions):
  arrived, left, idle_start, active_start, bending_start,
  standing_resumed, maintenance_alert
"""

import cv2
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, List, Dict

from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PERSON_CLASS_ID = 0
KP_NOSE                      = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW,    KP_R_ELBOW    = 7, 8
KP_L_WRIST,    KP_R_WRIST    = 9, 10
KP_L_HIP,      KP_R_HIP      = 11, 12
KP_L_ANKLE,    KP_R_ANKLE    = 15, 16
BEND_RATIO  = 0.35   # torso/leg height ratio below which pose = bending
KP_CONF_MIN = 0.20   # minimum keypoint confidence to use
MOTION_WINDOW = 10   # frames for rolling motion average


# ─── Enums ────────────────────────────────────────────────────────────────────

class WorkerState(str, Enum):
    ABSENT  = "absent"
    IDLE    = "idle"      # present, standing, not moving
    ACTIVE  = "active"    # present, standing, moving
    BENDING = "bending"   # bending pose — likely picking / loading


class WorkerEvent(str, Enum):
    ARRIVED           = "arrived"
    LEFT              = "left"
    IDLE_START        = "idle_start"
    ACTIVE_START      = "active_start"
    BENDING_START     = "bending_start"
    STANDING_RESUMED  = "standing_resumed"
    MAINTENANCE_ALERT = "maintenance_alert"
    SHEET_PICKED      = "sheet_picked"      # bending in rack zone → stood back up
    ENTERED_ZONE      = "entered_zone"      # person entered rack zone
    EXITED_ZONE       = "exited_zone"       # person left rack zone


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    event:        WorkerEvent
    state:        WorkerState
    timestamp:    datetime
    person_track: int  = 0
    person_label: str  = ""
    detail:       str  = ""

    def to_dict(self) -> dict:
        return {
            "event":        self.event.value,
            "state":        self.state.value,
            "timestamp":    self.timestamp.isoformat(),
            "person_track": self.person_track,
            "person_label": self.person_label,
            "detail":       self.detail,
        }


@dataclass
class PersonStatus:
    """Status of one tracked person in the current frame."""
    track_id:     int
    label:        str          # "Person 1", "Person 2", ...
    state:        WorkerState
    pose_state:   str          # "standing" | "bending" | "unknown"
    confidence:   float
    motion_score: float
    state_since:  datetime
    # Cumulative stats since tracker started (or last reset)
    total_active_s:  float = 0.0
    total_idle_s:    float = 0.0
    total_bending_s: float = 0.0
    bending_events:  int   = 0
    sheet_picks:     int   = 0
    in_rack_zone:    bool  = False


@dataclass
class WorkerStatusResult:
    persons:       List[PersonStatus]
    total_persons: int
    new_events:    List[EventRecord]
    timestamp:     datetime = field(default_factory=datetime.now)


# ─── Internal per-person tracking state ──────────────────────────────────────

@dataclass
class _PersonTrack:
    track_id:    int
    label:       str
    state:       WorkerState      = WorkerState.ABSENT
    state_since: datetime         = field(default_factory=datetime.now)

    prev_centroid:     Optional[Tuple[int, int]] = None
    motion_buffer:     deque = field(default_factory=lambda: deque(maxlen=MOTION_WINDOW))
    last_seen:         Optional[datetime] = None
    last_motion_time:  Optional[datetime] = None

    was_present:          bool  = False
    maintenance_alerted:  bool  = False

    # Zone tracking
    in_rack_zone:        bool            = False   # currently inside rack zone
    zone_enter_time:     Optional[datetime] = None # when they entered zone
    bent_in_zone:        bool            = False   # started bending while in zone
    zone_dwell_s:        float           = 0.0     # total seconds spent in zone
    confirmed_worker:    bool            = False   # True once they've dwelled in rack zone
    zone_entry_emitted:  bool            = False   # prevents duplicate ENTERED_ZONE events

    # Bending reference — head Y when standing upright (updated while standing)
    standing_head_y:     Optional[float] = None

    # Last known bounding box (x1, y1, x2, y2) for debug drawing
    last_bbox:           Optional[Tuple[int, int, int, int]] = None
    last_kps:            Optional[object] = None  # np.ndarray stored as object

    # Cumulative timing (added on every state transition)
    total_active_s:  float = 0.0
    total_idle_s:    float = 0.0
    total_bending_s: float = 0.0
    bending_events:  int   = 0
    sheet_picks:     int   = 0   # confirmed picks (bending in zone → stood up)

    def accumulate(self, now: datetime) -> None:
        """Add elapsed time in current state to running totals."""
        dur = (now - self.state_since).total_seconds()
        if self.state == WorkerState.ACTIVE:
            self.total_active_s += dur
        elif self.state == WorkerState.IDLE:
            self.total_idle_s += dur
        elif self.state == WorkerState.BENDING:
            self.total_bending_s += dur
            self.bending_events  += 1


# ─── WorkerTracker ────────────────────────────────────────────────────────────

class WorkerTracker:
    """
    Multi-person worker tracker using YOLOv8 Pose + ByteTrack.

    Call process_frame() from a background thread at ~2 fps.
    Read get_event_log() and get_person_stats() for shift integration.
    """

    MERGE_DIST_PX  = 300   # max pixels between last-known and new centroid to merge tracks
    MERGE_WINDOW_S = 20.0  # seconds a track stays eligible for merge after disappearing
    STALE_PURGE_S  = 60.0  # seconds before a non-present track is deleted entirely

    def __init__(
        self,
        model_path:          str   = "yolov8n-pose.pt",
        gpu:                 bool  = False,
        min_confidence:      float = 0.35,
        active_threshold:    float = 15.0,
        absent_timeout:      float = 5.0,   # increased: gives ByteTrack more time to recover track
        idle_timeout:        float = 5.0,
        maintenance_alert_s: float = 300.0,
        event_log_size:      int   = 200,
    ):
        logger.info(f"Loading YOLOv8 pose model: {model_path}")
        self.model = YOLO(model_path)
        self.device = "cuda" if gpu else "cpu"

        self.min_confidence      = min_confidence
        self.active_threshold    = active_threshold
        self.absent_timeout      = absent_timeout
        self.idle_timeout        = idle_timeout
        self.maintenance_alert_s = maintenance_alert_s

        self._persons:         Dict[int, _PersonTrack] = {}
        self._label_counter:   int                     = 0
        self._event_log:       deque                   = deque(maxlen=event_log_size)
        self._completed_stats: List[dict]              = []   # stats saved before track purge

        # Rack zone — normalized rect (set via set_rack_zone)
        # stored as pixel fractions: (x, y, w, h) all 0-1
        self._rack_zone: Optional[Tuple[float, float, float, float]] = None

        # Zone dwell filter: person must be in zone this long before events fire
        self.zone_dwell_threshold_s: float = 2.0

        logger.info("WorkerTracker ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> WorkerStatusResult:
        """Run one inference step. Returns status of all tracked persons."""
        now        = datetime.now()
        new_events: List[EventRecord] = []

        # ── Inference with tracking ────────────────────────────────────
        results = self.model.track(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.min_confidence,
            persist=True,
            device=self.device,
            verbose=False,
        )

        # Parse detections → {track_id: {conf, centroid, kps, bbox}}
        detected: Dict[int, dict] = {}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                confs     = boxes.conf.cpu().tolist()
                xyxy      = boxes.xyxy.cpu().tolist()
                kps_all   = (
                    results[0].keypoints.data.cpu().numpy()
                    if results[0].keypoints is not None else None
                )
                for i, (tid, conf, box) in enumerate(zip(track_ids, confs, xyxy)):
                    x1, y1, x2, y2 = box
                    detected[tid] = {
                        "conf":     conf,
                        "centroid": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "kps":      kps_all[i] if (kps_all is not None and i < len(kps_all)) else None,
                        "bbox":     (int(x1), int(y1), int(x2), int(y2)),
                    }

        # ── Stale track cleanup ────────────────────────────────────────
        self._cleanup_stale_tracks(now)

        # ── Update detected persons ────────────────────────────────────
        active_tids = set(detected.keys())
        for tid, det in detected.items():
            if tid not in self._persons:
                self._get_or_create_track(tid, det["centroid"], now, active_tids)

            track = self._persons[tid]

            # Motion scoring
            motion = 0.0
            if track.prev_centroid is not None:
                dx = det["centroid"][0] - track.prev_centroid[0]
                dy = det["centroid"][1] - track.prev_centroid[1]
                motion = (dx ** 2 + dy ** 2) ** 0.5
            track.motion_buffer.append(motion)
            track.prev_centroid   = det["centroid"]
            track.last_seen       = now
            if motion >= self.active_threshold:
                track.last_motion_time = now

            kps          = det["kps"]
            pose         = self._classify_pose(kps, track) if kps is not None else "unknown"
            motion_score = float(np.mean(track.motion_buffer)) if track.motion_buffer else 0.0
            track.last_bbox = det["bbox"]
            track.last_kps  = kps

            # Update standing head reference while person is upright
            if pose == "standing" and kps is not None and kps[KP_NOSE][2] > KP_CONF_MIN:
                # Smooth update — slowly track their normal head height
                if track.standing_head_y is None:
                    track.standing_head_y = float(kps[KP_NOSE][1])
                else:
                    track.standing_head_y = 0.9 * track.standing_head_y + 0.1 * float(kps[KP_NOSE][1])

            # Zone check
            frame_h, frame_w = frame.shape[:2]
            in_zone = self._in_zone(det["centroid"], frame_w, frame_h)

            events = self._update_state(track, now, motion_score, pose, in_zone)
            new_events.extend(events)

        # ── Handle persons not seen this frame ─────────────────────────
        for tid, track in list(self._persons.items()):
            if tid not in detected:
                track.motion_buffer.append(0.0)
                if track.last_seen and track.was_present:
                    if (now - track.last_seen).total_seconds() > self.absent_timeout:
                        events = self._mark_absent(track, now)
                        new_events.extend(events)

        # ── Build result ───────────────────────────────────────────────
        persons_status: List[PersonStatus] = []
        for tid, det in detected.items():
            track = self._persons[tid]
            # Skip unconfirmed persons when rack zone is active
            if self._rack_zone is not None and not track.confirmed_worker:
                continue
            motion_score = float(np.mean(track.motion_buffer)) if track.motion_buffer else 0.0
            pose         = self._classify_pose(det["kps"], track) if det["kps"] is not None else "unknown"
            persons_status.append(PersonStatus(
                track_id        = tid,
                label           = track.label,
                state           = track.state,
                pose_state      = pose,
                confidence      = round(det["conf"], 3),
                motion_score    = round(motion_score, 2),
                state_since     = track.state_since,
                total_active_s  = round(track.total_active_s, 1),
                total_idle_s    = round(track.total_idle_s, 1),
                total_bending_s = round(track.total_bending_s, 1),
                bending_events  = track.bending_events,
                sheet_picks     = track.sheet_picks,
                in_rack_zone    = track.in_rack_zone,
            ))

        persons_status.sort(key=lambda p: p.label)

        return WorkerStatusResult(
            persons       = persons_status,
            total_persons = len(persons_status),
            new_events    = new_events,
            timestamp     = now,
        )

    def get_event_log(self) -> List[EventRecord]:
        """All stored events, newest first."""
        return list(reversed(self._event_log))

    def get_person_stats(self) -> List[dict]:
        """
        Cumulative stats per person — used for shift productivity calculation.
        Returns stats for all persons who have ever been seen (including purged tracks).
        """
        stats = list(self._completed_stats)   # include purged-but-counted tracks
        active_ids = {s["track_id"] for s in stats}
        for t in self._persons.values():
            if t.track_id in active_ids:
                continue   # already saved (shouldn't happen, but guard)
            if not t.was_present and t.total_active_s == 0 and t.total_idle_s == 0:
                continue
            stats.append({
                "track_id":        t.track_id,
                "label":           t.label,
                "total_active_s":  round(t.total_active_s, 1),
                "total_idle_s":    round(t.total_idle_s, 1),
                "total_bending_s": round(t.total_bending_s, 1),
                "bending_events":  t.bending_events,
                "sheet_picks":     t.sheet_picks,
            })
        return stats

    def reset(self) -> None:
        self._persons.clear()
        self._completed_stats.clear()
        self._label_counter = 0
        self._event_log.clear()
        logger.info("WorkerTracker reset.")

    def set_rack_zone(self, zone: Optional[dict]) -> None:
        """
        Set the rack zone from a ZoneConfig-style dict
        {x, y, width, height} all normalized 0-1.
        Pass None to clear.
        """
        if zone is None:
            self._rack_zone = None
            logger.info("Rack zone cleared.")
        else:
            self._rack_zone = (zone["x"], zone["y"], zone["width"], zone["height"])
            logger.info(f"Rack zone set: {self._rack_zone}")

    def _in_zone(self, centroid: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """Return True if centroid is inside the rack zone."""
        if self._rack_zone is None:
            return False
        zx, zy, zw, zh = self._rack_zone
        cx, cy = centroid[0] / frame_w, centroid[1] / frame_h
        return zx <= cx <= zx + zw and zy <= cy <= zy + zh

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def get_debug_frame(
        self,
        frame:  np.ndarray,
        result: Optional[WorkerStatusResult] = None,
    ) -> np.ndarray:
        """Draw debug overlay using stored track state — no extra inference."""
        debug = frame.copy()

        STATE_COLORS = {
            WorkerState.ACTIVE:  (0, 200, 0),
            WorkerState.BENDING: (0, 140, 255),
            WorkerState.IDLE:    (0, 165, 255),
            WorkerState.ABSENT:  (0, 0, 200),
        }

        for track in self._persons.values():
            if track.last_bbox is None:
                continue
            if track.state == WorkerState.ABSENT:
                continue
            x1, y1, x2, y2 = track.last_bbox
            color = STATE_COLORS.get(track.state, (128, 128, 128))
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                debug,
                f"{track.label}  {track.state.value}",
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
            )
            if track.last_kps is not None:
                self._draw_skeleton(debug, track.last_kps, color)

        # Draw rack zone overlay
        if self._rack_zone is not None:
            h, w = debug.shape[:2]
            zx, zy, zw, zh = self._rack_zone
            rx1, ry1 = int(zx * w), int(zy * h)
            rx2, ry2 = int((zx + zw) * w), int((zy + zh) * h)
            overlay = debug.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 165, 0), -1)
            cv2.addWeighted(overlay, 0.15, debug, 0.85, 0, debug)
            cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 165, 0), 2)
            cv2.putText(debug, "RACK ZONE", (rx1 + 4, ry1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 1)

        # Top banner
        if result:
            parts   = [f"{p.label}: {p.state.value}" for p in result.persons]
            summary = f"Workers: {result.total_persons}  |  " + "   ".join(parts) if parts else f"Workers: 0"
            cv2.rectangle(debug, (0, 0), (debug.shape[1], 40), (20, 20, 20), -1)
            cv2.putText(debug, summary, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)

        return debug

    # ------------------------------------------------------------------
    # Internal: track ID merge + stale cleanup
    # ------------------------------------------------------------------

    def _get_or_create_track(
        self, tid: int, centroid: Tuple[int, int], now: datetime, active_tids: set
    ) -> _PersonTrack:
        """
        Return existing track for `tid`, or create a new one.

        MERGE LOGIC: ByteTrack sometimes drops a track for 1-2 frames and
        assigns a new ID to the same physical person. To prevent "Person 1"
        becoming "Person 2, 3, 4...", before creating a new label we search
        for any existing track that:
          1. Is NOT currently being tracked this frame (not in active_tids)
          2. Was last seen within MERGE_WINDOW_S seconds
          3. Its last-known centroid is within MERGE_DIST_PX of the new detection

        If found, the new tid takes over that track's label + accumulated stats.
        """
        if tid in self._persons:
            return self._persons[tid]

        best_track: Optional[_PersonTrack] = None
        best_dist = float("inf")

        for existing in list(self._persons.values()):
            # Skip tracks that ARE currently detected (they can't be the same person)
            if existing.track_id in active_tids:
                continue
            if existing.prev_centroid is None or existing.last_seen is None:
                continue
            # Must be within the merge time window
            since_seen = (now - existing.last_seen).total_seconds()
            if since_seen > self.MERGE_WINDOW_S:
                continue
            # Must be within the merge distance
            dx = centroid[0] - existing.prev_centroid[0]
            dy = centroid[1] - existing.prev_centroid[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < self.MERGE_DIST_PX and dist < best_dist:
                best_dist  = dist
                best_track = existing

        if best_track is not None:
            old_tid = best_track.track_id
            gap_s = (now - best_track.last_seen).total_seconds() if best_track.last_seen else 0
            logger.debug(
                f"Track merge: new tid={tid} ← old tid={old_tid} "
                f"({best_track.label}, dist={best_dist:.0f}px, gap={gap_s:.1f}s)"
            )
            best_track.track_id = tid
            del self._persons[old_tid]
            self._persons[tid] = best_track
            return best_track

        # No merge candidate — genuinely new person
        self._label_counter += 1
        track = _PersonTrack(track_id=tid, label=f"Person {self._label_counter}")
        self._persons[tid] = track
        return track

    def _cleanup_stale_tracks(self, now: datetime) -> None:
        """
        Delete _PersonTrack entries for persons who left more than
        STALE_PURGE_S seconds ago. Also resets the label counter when
        no persons remain, so labelling restarts at Person 1.
        """
        to_delete = []
        for tid, track in self._persons.items():
            if track.state == WorkerState.ABSENT and track.last_seen is not None:
                if (now - track.last_seen).total_seconds() > self.STALE_PURGE_S:
                    to_delete.append(tid)
        for tid in to_delete:
            t = self._persons[tid]
            logger.debug(f"Purging stale track tid={tid} ({t.label})")
            # Save stats before deleting so shift report doesn't lose picks
            self._completed_stats.append({
                "track_id":        t.track_id,
                "label":           t.label,
                "total_active_s":  round(t.total_active_s, 1),
                "total_idle_s":    round(t.total_idle_s, 1),
                "total_bending_s": round(t.total_bending_s, 1),
                "bending_events":  t.bending_events,
                "sheet_picks":     t.sheet_picks,
            })
            del self._persons[tid]
        if not self._persons:
            self._label_counter = 0

    # ------------------------------------------------------------------
    # Internal: state machine
    # ------------------------------------------------------------------

    def _update_state(
        self,
        track:        _PersonTrack,
        now:          datetime,
        motion_score: float,
        pose_state:   str,
        in_zone:      bool = False,
    ) -> List[EventRecord]:
        events: List[EventRecord] = []

        def emit(ev: WorkerEvent, detail: str = "") -> None:
            rec = EventRecord(
                event=ev, state=track.state, timestamp=now,
                person_track=track.track_id, person_label=track.label, detail=detail,
            )
            # Only log if no rack zone set (show all) OR person is confirmed worker
            if self._rack_zone is None or track.confirmed_worker:
                self._event_log.append(rec)
            events.append(rec)

        def transition(new_state: WorkerState) -> None:
            track.accumulate(now)
            track.state       = new_state
            track.state_since = now

        # ── Arrival ───────────────────────────────────────────────────
        if not track.was_present:
            emit(WorkerEvent.ARRIVED, f"{track.label} entered frame")
            track.was_present         = True
            track.maintenance_alerted = False
            track.last_motion_time    = now

        # ── Zone entry / exit (rack zone) ─────────────────────────────
        if self._rack_zone is not None:
            if in_zone and not track.in_rack_zone:
                # Just entered zone — start dwell timer
                track.in_rack_zone    = True
                track.zone_enter_time = now
                track.bent_in_zone    = False
            elif not in_zone and track.in_rack_zone:
                # Just exited zone
                track.in_rack_zone = False
                if track.zone_enter_time:
                    dwell = (now - track.zone_enter_time).total_seconds()
                    track.zone_dwell_s += dwell
                    if dwell >= self.zone_dwell_threshold_s:
                        emit(WorkerEvent.EXITED_ZONE,
                             f"{track.label} left rack zone after {dwell:.0f}s")
                track.zone_enter_time = None
                track.bent_in_zone    = False

            # Fire ENTERED_ZONE only after dwell threshold met
            if in_zone and track.zone_enter_time and not track.zone_entry_emitted:
                dwell_so_far = (now - track.zone_enter_time).total_seconds()
                if dwell_so_far >= self.zone_dwell_threshold_s:
                    track.confirmed_worker   = True  # confirm before emit so log captures it
                    track.zone_entry_emitted = True
                    emit(WorkerEvent.ENTERED_ZONE,
                         f"{track.label} confirmed as worker — in rack zone")
        else:
            track.in_rack_zone = False

        # ── Target state ──────────────────────────────────────────────
        if pose_state == "bending":
            target = WorkerState.BENDING
        elif motion_score >= self.active_threshold:
            target = WorkerState.ACTIVE
        else:
            since_motion = (
                (now - track.last_motion_time).total_seconds()
                if track.last_motion_time else float("inf")
            )
            target = WorkerState.IDLE if since_motion >= self.idle_timeout else WorkerState.ACTIVE

        # ── Transition events ─────────────────────────────────────────
        if target != track.state:
            prev = track.state

            if target == WorkerState.ACTIVE:
                emit(WorkerEvent.ACTIVE_START, f"motion={motion_score:.1f}px")
            elif target == WorkerState.IDLE:
                emit(WorkerEvent.IDLE_START)
            elif target == WorkerState.BENDING:
                emit(WorkerEvent.BENDING_START,
                     "bending in rack zone — picking sheet" if track.in_rack_zone
                     else "bending detected")
                if track.in_rack_zone:
                    track.bent_in_zone = True

            # Standing back up after bending
            if prev == WorkerState.BENDING and target in (WorkerState.IDLE, WorkerState.ACTIVE):
                emit(WorkerEvent.STANDING_RESUMED)
                # SHEET_PICKED: was bending inside rack zone, now standing up
                if track.bent_in_zone and track.in_rack_zone:
                    track.sheet_picks += 1
                    emit(WorkerEvent.SHEET_PICKED,
                         f"{track.label} picked sheet from rack (pick #{track.sheet_picks})")
                    track.bent_in_zone = False

            transition(target)

        # Reset zone entry flag when person leaves zone (so re-entry fires again)
        if not in_zone:
            track.zone_entry_emitted = False

        # ── Maintenance alert ─────────────────────────────────────────
        if track.state == WorkerState.IDLE and not track.maintenance_alerted:
            idle_s = (now - track.state_since).total_seconds()
            if idle_s >= self.maintenance_alert_s:
                emit(WorkerEvent.MAINTENANCE_ALERT,
                     f"{track.label} idle for {idle_s:.0f}s")
                track.maintenance_alerted = True

        return events

    def _mark_absent(self, track: _PersonTrack, now: datetime) -> List[EventRecord]:
        rec = EventRecord(
            event=WorkerEvent.LEFT, state=track.state, timestamp=now,
            person_track=track.track_id, person_label=track.label,
            detail=f"{track.label} left frame",
        )
        self._event_log.append(rec)
        track.accumulate(now)
        track.state               = WorkerState.ABSENT
        track.state_since         = now
        track.was_present         = False
        track.maintenance_alerted = False
        return [rec]

    # ------------------------------------------------------------------
    # Internal: pose classification
    # ------------------------------------------------------------------

    def _classify_pose(self, kps: np.ndarray, track: "_PersonTrack" = None) -> str:
        """
        Classify pose using a voting system across 4 detection methods.
        Returns 'bending' if 2+ methods agree, else 'standing' or 'unknown'.

        Methods (most → least reliable for factory sheet picking):
          1. Wrist below hip  — hand reaching down to pick
          2. Head drop        — head Y drops vs standing reference
          3. Torso collapse   — torso shrinks (upper body only)
          4. Full body ratio  — classic torso/leg ratio (needs full body)
        """
        def v(i): return kps[i][2] > KP_CONF_MIN
        def y(i): return float(kps[i][1])
        def x(i): return float(kps[i][0])

        bending_votes = 0
        standing_votes = 0

        # ── Method 1: wrist below hip (most reliable for picking) ─────
        has_wrists = v(KP_L_WRIST) or v(KP_R_WRIST)
        has_hips   = v(KP_L_HIP)   or v(KP_R_HIP)
        if has_wrists and has_hips:
            wrist_ys = [y(i) for i in (KP_L_WRIST, KP_R_WRIST) if v(i)]
            hip_ys   = [y(i) for i in (KP_L_HIP,   KP_R_HIP)   if v(i)]
            min_wrist = min(wrist_ys)   # highest wrist (smallest y = highest in frame)
            avg_hip   = sum(hip_ys) / len(hip_ys)
            if min_wrist > avg_hip + 20:   # at least one wrist clearly below hips
                bending_votes += 2         # strong signal — double vote
            else:
                standing_votes += 1

        # ── Method 2: head / nose drop vs standing reference ──────────
        if (track is not None and track.standing_head_y is not None
                and v(KP_NOSE)):
            head_y = y(KP_NOSE)
            drop   = head_y - track.standing_head_y   # positive = dropped below normal
            if drop > 40:    # head dropped 40+ px below normal standing height
                bending_votes += 2
            elif drop > 20:
                bending_votes += 1
            else:
                standing_votes += 1

        # ── Method 3: torso collapse (shoulders + hips visible) ───────
        has_sh = v(KP_L_SHOULDER) or v(KP_R_SHOULDER)
        if has_sh and has_hips:
            sh_ys  = [y(i) for i in (KP_L_SHOULDER, KP_R_SHOULDER) if v(i)]
            hip_ys = [y(i) for i in (KP_L_HIP,      KP_R_HIP)      if v(i)]
            torso_h  = (sum(hip_ys) / len(hip_ys)) - (sum(sh_ys) / len(sh_ys))
            sh_xs    = [x(i) for i in (KP_L_SHOULDER, KP_R_SHOULDER) if v(i)]
            sh_width = (max(sh_xs) - min(sh_xs)) if len(sh_xs) >= 2 else 0
            if torso_h < 15:                                # torso nearly gone
                bending_votes += 2
            elif sh_width > 0 and (torso_h / max(sh_width, 1)) < 0.35:
                bending_votes += 1                          # torso much shorter than width
            else:
                standing_votes += 1

        # ── Method 4: full body torso/leg ratio ───────────────────────
        if (v(KP_L_SHOULDER) and v(KP_R_SHOULDER) and
                v(KP_L_HIP) and v(KP_R_HIP) and
                (v(KP_L_ANKLE) or v(KP_R_ANKLE))):
            shoulder_y = (y(KP_L_SHOULDER) + y(KP_R_SHOULDER)) / 2
            hip_y      = (y(KP_L_HIP)      + y(KP_R_HIP))      / 2
            ankle_ys   = [y(i) for i in (KP_L_ANKLE, KP_R_ANKLE) if v(i)]
            ankle_y    = sum(ankle_ys) / len(ankle_ys)
            torso_h    = hip_y - shoulder_y
            leg_h      = max(ankle_y - hip_y, 1.0)
            if (torso_h / leg_h) < BEND_RATIO:
                bending_votes += 1
            else:
                standing_votes += 1

        # ── Verdict ───────────────────────────────────────────────────
        total = bending_votes + standing_votes
        if total == 0:
            return "unknown"
        if bending_votes >= 2:
            return "bending"
        if standing_votes >= 2:
            return "standing"
        return "unknown"

    # ------------------------------------------------------------------
    # Internal: skeleton drawing
    # ------------------------------------------------------------------

    _SKELETON_PAIRS = [
        (5, 6),
        (5, 7),  (7, 9),
        (6, 8),  (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, color=(0, 255, 180)) -> None:
        for a, b in self._SKELETON_PAIRS:
            if kps[a][2] > KP_CONF_MIN and kps[b][2] > KP_CONF_MIN:
                cv2.line(
                    frame,
                    (int(kps[a][0]), int(kps[a][1])),
                    (int(kps[b][0]), int(kps[b][1])),
                    color, 2,
                )
        for i in range(17):
            if kps[i][2] > KP_CONF_MIN:
                cv2.circle(frame, (int(kps[i][0]), int(kps[i][1])), 3, (255, 255, 0), -1)
