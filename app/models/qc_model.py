from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class QCEntryRequest(BaseModel):
    """Worker-submitted QC form data."""
    camera_id:    str
    shift_id:     Optional[str] = None
    worker_name:  str
    worker_label: str
    pick_time:    datetime       # from SHEET_PICKED event timestamp
    qty_picked:   int = Field(..., gt=0,  description="Total sheets picked from rack")
    qty_pass:     int = Field(..., ge=0,  description="Sheets that passed QC")
    qty_reject:   int = Field(..., ge=0,  description="Sheets rejected")


class QCEntryResponse(BaseModel):
    entry_id:     int
    camera_id:    str
    shift_id:     Optional[str]
    worker_name:  str
    worker_label: str
    pick_time:    datetime
    submitted_at: datetime
    qty_picked:   int
    qty_pass:     int
    qty_reject:   int
    pass_rate:    float          # qty_pass / qty_picked × 100


class QCReport(BaseModel):
    """Aggregate QC stats — per shift or per camera."""
    shift_id:          Optional[str]
    camera_id:         Optional[str]
    total_entries:     int
    total_picked:      int
    total_pass:        int
    total_reject:      int
    overall_pass_rate: float
    entries:           List[QCEntryResponse]
