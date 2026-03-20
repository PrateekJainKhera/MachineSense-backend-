from pydantic import BaseModel
from typing import Optional, List
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
    camera_id:          str
    machine_running:    bool
    stopped_since:      Optional[datetime] = None
    stopped_secs:       float              = 0.0
    last_counter:       Optional[int]      = None
    last_seen_at:       Optional[datetime] = None
    unattended_alert:   bool               = False
    unattended_secs:    float              = 0.0  # how long unattended while running


class DowntimeSummary(BaseModel):
    camera_id:        str
    total_events:     int
    total_downtime_s: float
    longest_event_s:  float
    active_event:     Optional[DowntimeRecord] = None


class OEEResult(BaseModel):
    camera_id:        str
    shift_duration_s: float
    downtime_s:       float
    uptime_s:         float
    availability_pct: float   # (shift - downtime) / shift × 100


class HistoryStats(BaseModel):
    camera_id:          str
    total_stops:        int
    total_downtime_s:   float
    avg_duration_s:     float
    longest_s:          float
    most_common_reason: str
    reasons_breakdown:  dict        # reason → count
    avg_interval_s:     float       # avg time between stops
    last_stop_ago_s:    Optional[float]  # seconds since last stop ended


class PatternInsight(BaseModel):
    camera_id:          str
    avg_interval_s:     float       # expected time between stops
    last_stop_ago_s:    Optional[float]
    overdue:            bool        # last_stop_ago > avg_interval * 1.5
    high_frequency:     bool        # stops coming faster than avg
    message:            str         # human-readable insight
