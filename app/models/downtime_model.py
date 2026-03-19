from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DowntimeRecord(BaseModel):
    event_id:       int
    camera_id:      str
    started_at:     datetime
    ended_at:       Optional[datetime] = None
    duration_s:     Optional[float]    = None   # filled when resolved
    status:         str                = "active"   # "active" | "resolved"
    reason:         str                = "unknown"
    # unknown | unattended | plate_change | maintenance | breakdown | planned_stop
    worker_present: Optional[bool]     = None
    worker_state:   Optional[str]      = None   # active | idle | None


class DowntimeStatus(BaseModel):
    camera_id:       str
    machine_running: bool
    stopped_since:   Optional[datetime] = None   # when it stopped
    stopped_secs:    float              = 0.0    # how long stopped
    last_counter:    Optional[int]      = None
    last_seen_at:    Optional[datetime] = None


class DowntimeSummary(BaseModel):
    camera_id:        str
    total_events:     int
    total_downtime_s: float
    longest_event_s:  float
    active_event:     Optional[DowntimeRecord] = None
