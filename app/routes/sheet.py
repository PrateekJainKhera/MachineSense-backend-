from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from datetime import datetime

from app.models.sheet_model import (
    SheetCameraConfig,
    CountingLineConfig,
    SheetCountResponse,
    SheetCameraStatusResponse,
    RegisterSheetCameraResponse,
    ResetCountResponse,
)

router = APIRouter(prefix="/sheets", tags=["Sheet Counting"])


def _get_service(request: Request):
    return request.app.state.sheet_service


# ------------------------------------------------------------------
# Camera management
# ------------------------------------------------------------------

@router.post("/cameras/register", response_model=RegisterSheetCameraResponse)
def register_camera(config: SheetCameraConfig, request: Request):
    """
    Register a sheet-counting camera (Camera 2).
    Starts capturing frames and sets up the YOLOv8 tracker immediately.

    Example body:
        {
            "camera_id": "machine_a_sheets",
            "source": "rtsp://192.168.1.101:554/stream",
            "counting_line": { "position": 300, "orientation": "horizontal" },
            "confidence": 0.3
        }
    """
    service = _get_service(request)
    service.register_camera(config)
    return RegisterSheetCameraResponse(
        message=f"Sheet camera '{config.camera_id}' registered successfully.",
        camera_id=config.camera_id,
    )


@router.delete("/cameras/{camera_id}")
def unregister_camera(camera_id: str, request: Request):
    """Stop and remove a sheet-counting camera."""
    service = _get_service(request)
    removed = service.unregister_camera(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"Camera '{camera_id}' removed."}


@router.put("/cameras/{camera_id}/counting-line", response_model=RegisterSheetCameraResponse)
def update_counting_line(camera_id: str, line: CountingLineConfig, request: Request):
    """
    Update the counting line for a registered camera.
    The counting line is where sheets are counted as they cross it.

    Example body:
        { "position": 350, "orientation": "horizontal" }
    """
    service = _get_service(request)
    updated = service.set_counting_line(camera_id, line)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return RegisterSheetCameraResponse(
        message=f"Counting line updated for '{camera_id}' at position {line.position}px.",
        camera_id=camera_id,
    )


@router.get("/cameras", response_model=list[SheetCameraStatusResponse])
def list_cameras(request: Request):
    """List all registered sheet-counting cameras and their current counts."""
    service = _get_service(request)
    cameras = service.list_cameras()
    result = []
    for c in cameras:
        line = None
        if c.get("counting_line") and c["counting_line"].get("position") is not None:
            line = CountingLineConfig(
                position=c["counting_line"]["position"],
                orientation=c["counting_line"]["orientation"],
            )
        result.append(SheetCameraStatusResponse(
            camera_id=c["camera_id"],
            source=c["source"],
            connected=c["connected"],
            has_frame=c["has_frame"],
            total_count=c["total_count"],
            counting_line=line,
        ))
    return result


@router.get("/cameras/{camera_id}/status", response_model=SheetCameraStatusResponse)
def camera_status(camera_id: str, request: Request):
    """Get status and current count for a specific sheet camera."""
    service = _get_service(request)
    status = service.get_status(camera_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    line = None
    if status.get("counting_line") and status["counting_line"].get("position") is not None:
        line = CountingLineConfig(
            position=status["counting_line"]["position"],
            orientation=status["counting_line"]["orientation"],
        )
    return SheetCameraStatusResponse(
        camera_id=status["camera_id"],
        source=status["source"],
        connected=status["connected"],
        has_frame=status["has_frame"],
        total_count=status["total_count"],
        counting_line=line,
    )


# ------------------------------------------------------------------
# Sheet counting endpoints
# ------------------------------------------------------------------

@router.get("/count/{camera_id}", response_model=SheetCountResponse)
def get_count(camera_id: str, request: Request):
    """
    Process the latest frame from a camera and return the current sheet count.
    Call this repeatedly to get live sheet counts (e.g. every 100ms from frontend).
    """
    service = _get_service(request)
    result = service.process_latest_frame(camera_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found or no frame available yet.",
        )
    return SheetCountResponse(
        camera_id=camera_id,
        total_count=result.total_count,
        newly_counted=result.newly_counted,
        active_tracks=result.active_tracks,
        timestamp=result.timestamp,
    )


@router.post("/reset/{camera_id}", response_model=ResetCountResponse)
def reset_count(camera_id: str, request: Request):
    """
    Reset the sheet count for a camera to zero.
    Call this at the start of each new job.
    """
    service = _get_service(request)
    previous = service.reset_count(camera_id)
    if previous is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return ResetCountResponse(
        message=f"Sheet count reset for '{camera_id}'.",
        camera_id=camera_id,
        previous_count=previous,
    )


@router.get("/debug/{camera_id}")
def debug_frame(camera_id: str, request: Request):
    """
    Returns a JPEG image with the counting line, bounding boxes,
    track IDs, and current sheet count drawn on the live frame.
    Open in browser to visually verify counting line placement.
    """
    service = _get_service(request)
    jpeg_bytes = service.get_debug_image(camera_id)
    if jpeg_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{camera_id}' not found or no frame available yet.",
        )
    return Response(content=jpeg_bytes, media_type="image/jpeg")
