from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from vision.worker_tracker import WorkerState, WorkerEvent


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class WorkerCameraConfig(BaseModel):
    """Configuration for registering a worker-tracking camera."""
    camera_id:           str   = Field(..., description="Unique camera identifier")
    source:              str   = Field(..., description="RTSP URL, video file, or camera index")
    min_confidence:      float = Field(0.35,  ge=0.0, le=1.0)
    active_threshold:    float = Field(15.0,  gt=0,   description="Motion px/frame → ACTIVE")
    absent_timeout:      float = Field(3.0,   gt=0,   description="Seconds without detection → ABSENT")
    idle_timeout:        float = Field(5.0,   gt=0,   description="Seconds without motion → IDLE")
    maintenance_alert_s: float = Field(300.0, gt=0,   description="Seconds idle → MAINTENANCE_ALERT")
    loop:                bool  = Field(True)


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class EventRecordResponse(BaseModel):
    event:        WorkerEvent
    state:        WorkerState
    timestamp:    datetime
    person_track: int = 0
    person_label: str = ""
    detail:       str = ""


class PersonStatusResponse(BaseModel):
    """Live status of one tracked person."""
    track_id:        int
    label:           str          # "Person 1", "Person 2"
    state:           WorkerState
    pose_state:      str          # "standing" | "bending" | "unknown"
    confidence:      float
    motion_score:    float
    state_since:     datetime
    total_active_s:  float = 0.0
    total_idle_s:    float = 0.0
    total_bending_s: float = 0.0
    bending_events:  int   = 0
    sheet_picks:     int   = 0
    in_rack_zone:    bool  = False


class ActiveShiftSummary(BaseModel):
    """Compact shift info embedded in worker status — for live display."""
    shift_id:          str
    worker_name:       str
    started_at:        datetime
    duration_s:        float
    live_cycles:       int
    live_sheets_per_hr: float
    live_efficiency:   float


class WorkerStatusResponse(BaseModel):
    """
    Full worker status from the background poller.
    Includes all tracked persons + recent events + active shift summary.
    """
    camera_id:     str
    total_persons: int                      = 0
    persons:       List[PersonStatusResponse] = Field(default_factory=list)
    recent_events: List[EventRecordResponse]  = Field(default_factory=list)
    active_shift:  Optional[ActiveShiftSummary] = None
    polled_at:     Optional[datetime]        = None
    stale_seconds: Optional[float]           = None


class WorkerEventLogResponse(BaseModel):
    camera_id:    str
    total_events: int
    events:       List[EventRecordResponse]


class WorkerCameraStatusResponse(BaseModel):
    camera_id:     str
    source:        str
    connected:     bool
    has_frame:     bool
    current_state: Optional[WorkerState] = None
    person_count:  int = 0


class RegisterWorkerCameraResponse(BaseModel):
    message:   str
    camera_id: str


# ------------------------------------------------------------------
# Zone config
# ------------------------------------------------------------------

class ZoneRect(BaseModel):
    """Normalized rectangle (all values 0.0–1.0 relative to frame dimensions)."""
    x:      float = Field(..., ge=0.0, le=1.0, description="Left edge")
    y:      float = Field(..., ge=0.0, le=1.0, description="Top edge")
    width:  float = Field(..., gt=0.0, le=1.0, description="Width")
    height: float = Field(..., gt=0.0, le=1.0, description="Height")


class ZoneConfig(BaseModel):
    rack_zone:    Optional[ZoneRect] = None   # where sheets are picked / returned
    inspect_zone: Optional[ZoneRect] = None   # where worker inspects sheets
