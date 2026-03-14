from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class CountingLineConfig(BaseModel):
    """Position of the counting line in the camera frame."""
    position: int = Field(..., gt=0, description="Pixel coordinate of the line (y for horizontal, x for vertical)")
    orientation: str = Field("horizontal", description="'horizontal' or 'vertical'")


class SheetCameraConfig(BaseModel):
    """Configuration for registering a sheet-counting camera."""
    camera_id: str = Field(..., description="Unique camera identifier e.g. 'machine_a_sheets'")
    source: str = Field(..., description="RTSP URL, video file path, or camera index as string")
    counting_line: Optional[CountingLineConfig] = Field(None, description="Counting line position")
    confidence: float = Field(0.3, ge=0.1, le=1.0, description="YOLOv8 detection confidence threshold")


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class SheetCountResponse(BaseModel):
    """Current sheet count for a camera."""
    camera_id: str
    total_count: int = Field(..., description="Total sheets counted since last reset")
    newly_counted: int = Field(..., description="Sheets counted in the last processed frame")
    active_tracks: int = Field(..., description="Number of objects tracked in the last frame")
    timestamp: datetime


class SheetCameraStatusResponse(BaseModel):
    """Status of a registered sheet-counting camera."""
    camera_id: str
    source: str
    connected: bool
    has_frame: bool
    total_count: int
    counting_line: Optional[CountingLineConfig] = None


class RegisterSheetCameraResponse(BaseModel):
    """Confirmation when a sheet camera is registered."""
    message: str
    camera_id: str


class ResetCountResponse(BaseModel):
    """Confirmation when count is reset."""
    message: str
    camera_id: str
    previous_count: int
