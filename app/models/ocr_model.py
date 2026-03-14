from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ------------------------------------------------------------------
# Request schemas
# ------------------------------------------------------------------

class ROIConfig(BaseModel):
    """Region of Interest coordinates for a camera."""
    x: int = Field(..., ge=0, description="Left edge (pixels)")
    y: int = Field(..., ge=0, description="Top edge (pixels)")
    width: int = Field(..., gt=0, description="Width (pixels)")
    height: int = Field(..., gt=0, description="Height (pixels)")


class CameraConfig(BaseModel):
    """Configuration for registering a camera source."""
    camera_id: str = Field(..., description="Unique camera identifier e.g. 'machine_a_display'")
    source: str = Field(..., description="RTSP URL, video file path, or camera index as string")
    roi: Optional[ROIConfig] = Field(None, description="Region of Interest for OCR cropping")
    loop: bool = Field(True, description="Loop video files. Set False to stop OCR when video ends.")


# ------------------------------------------------------------------
# Response schemas
# ------------------------------------------------------------------

class ValidationInfo(BaseModel):
    """Result of confidence + rate validation on an OCR reading."""
    is_valid: bool
    confidence_ok: bool
    rate_ok: bool
    direction_ok: bool
    reason: str


class OCRResponse(BaseModel):
    """Response from a single-value OCR read operation."""
    success: bool
    camera_id: str
    value: Optional[int] = Field(None, description="Extracted machine counter value")
    raw_text: str = Field("", description="Raw OCR text before parsing")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime
    error: Optional[str] = None
    validation: Optional[ValidationInfo] = Field(
        None,
        description="Validation result — present when validate=true was requested"
    )


class BBoxModel(BaseModel):
    """Bounding box of a detected number in the image."""
    x: int
    y: int
    width: int
    height: int


class DetectedNumberResponse(BaseModel):
    """One number detected by read_all_counters()."""
    value: int
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBoxModel


class MultiOCRResponse(BaseModel):
    """Response from a read-all operation — returns every number found in the image."""
    camera_id: str
    count: int = Field(..., description="Total numbers detected")
    numbers: List[DetectedNumberResponse]
    timestamp: datetime


class CameraStatusResponse(BaseModel):
    """Live status of a registered camera."""
    camera_id: str
    source: str
    connected: bool
    has_frame: bool
    roi: Optional[ROIConfig] = None


class RegisterCameraResponse(BaseModel):
    """Confirmation response when a camera is registered."""
    message: str
    camera_id: str


class LatestReadingResponse(BaseModel):
    """
    Response from GET /ocr/latest/{camera_id}.

    Returns the background poller's cached result — instant, no OCR wait.
    Use this for live dashboards polling every few seconds.

    rate_per_second: estimated production speed derived from the last
                     10 valid readings (None until enough history exists).
    polled_at:       when the background poller last captured this value.
    stale_seconds:   how many seconds ago the value was last updated.
    """
    camera_id: str
    value: Optional[int] = None
    confidence: float = 0.0
    raw_text: str = ""
    rate_per_second: Optional[float] = Field(
        None, description="Estimated production rate — counts per second"
    )
    polled_at: Optional[datetime] = None
    stale_seconds: Optional[float] = Field(
        None, description="Seconds since last valid reading was captured"
    )
    validation: Optional[ValidationInfo] = None
