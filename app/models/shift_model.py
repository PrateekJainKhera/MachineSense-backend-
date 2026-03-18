from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

from vision.worker_tracker import WorkerEvent, WorkerState


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class StartShiftRequest(BaseModel):
    """Start a new worker shift on a camera."""
    camera_id:   str = Field(..., description="Camera ID to attach this shift to")
    worker_name: str = Field(..., description="Name of the worker starting the shift")
    notes:       str = Field("", description="Optional notes (e.g. job/order number)")


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class ShiftEventResponse(BaseModel):
    """One event that occurred during a shift."""
    event:        WorkerEvent
    state:        WorkerState
    person_track: int
    person_label: str
    timestamp:    datetime
    detail:       str = ""


class PersonProductivity(BaseModel):
    """Productivity stats for one tracked person within a shift."""
    person_label:      str
    active_s:          float
    idle_s:            float
    bending_s:         float
    bending_events:    int
    estimated_cycles:  int    = Field(..., description="sheet_picks — 1 confirmed pick = 1 cycle")


class ProductivityReport(BaseModel):
    """
    Full productivity report generated when a shift ends.

    total_cycles:    confirmed sheet picks across all workers (1 pick = 1 cycle)
    sheets_per_hour: total_cycles / shift_hours
    efficiency_pct:  active_time / (active_time + idle_time) × 100
    """
    shift_id:         str
    worker_name:      str
    camera_id:        str
    started_at:       datetime
    ended_at:         datetime
    duration_s:       float
    duration_fmt:     str               # e.g. "2h 34m"
    total_cycles:     int
    sheets_per_hour:  float
    efficiency_pct:   float
    persons:          List[PersonProductivity]


class ShiftResponse(BaseModel):
    """Summary of a shift (active or ended)."""
    shift_id:    str
    worker_name: str
    camera_id:   str
    started_at:  datetime
    ended_at:    Optional[datetime]
    notes:       str
    is_active:   bool
    duration_s:  float                  # elapsed so far (or total if ended)
    event_count: int
    # Live productivity (only meaningful while active)
    live_cycles:       int   = 0
    live_sheets_per_hr: float = 0.0
    live_efficiency:   float = 0.0
