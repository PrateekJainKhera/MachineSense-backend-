import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from datetime import datetime

from app.models.worker_model import (
    WorkerCameraConfig,
    WorkerStatusResponse,
    WorkerEventLogResponse,
    EventRecordResponse,
    PersonStatusResponse,
    ActiveShiftSummary,
    WorkerCameraStatusResponse,
    RegisterWorkerCameraResponse,
    ZoneConfig,
)

router = APIRouter(prefix="/workers", tags=["Worker Tracking"])


def _svc(request: Request):
    return request.app.state.worker_service


# ------------------------------------------------------------------
# Camera management
# ------------------------------------------------------------------

@router.post("/cameras/register", response_model=RegisterWorkerCameraResponse)
def register_camera(config: WorkerCameraConfig, request: Request):
    """
    Register a camera for multi-person worker tracking.

    Uses yolov8n-pose.pt + ByteTrack — detects all persons in frame,
    tracks each with a persistent ID, and classifies pose (standing/bending).
    No ROI needed — full frame is analysed.
    """
    _svc(request).register_camera(config)
    return RegisterWorkerCameraResponse(
        message=f"Worker camera '{config.camera_id}' registered.",
        camera_id=config.camera_id,
    )


@router.delete("/cameras/{camera_id}")
def unregister_camera(camera_id: str, request: Request):
    removed = _svc(request).unregister_camera(camera_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {"message": f"Camera '{camera_id}' removed."}


@router.get("/cameras", response_model=list[WorkerCameraStatusResponse])
def list_cameras(request: Request):
    return [
        WorkerCameraStatusResponse(**c)
        for c in _svc(request).list_cameras()
    ]


@router.get("/cameras/{camera_id}/status", response_model=WorkerCameraStatusResponse)
def camera_status(camera_id: str, request: Request):
    status = _svc(request).get_camera_status(camera_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return WorkerCameraStatusResponse(**status)


# ------------------------------------------------------------------
# Zone config
# ------------------------------------------------------------------

@router.post("/cameras/{camera_id}/zones")
def set_zones(camera_id: str, zones: ZoneConfig, request: Request):
    """Save Rack Zone and Inspect Zone for a camera (normalized 0-1 coords)."""
    svc = _svc(request)
    if camera_id not in [c["camera_id"] for c in svc.list_cameras()]:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    svc.set_zones(camera_id, zones)
    return {"message": "Zones saved.", "camera_id": camera_id}


@router.get("/cameras/{camera_id}/zones", response_model=ZoneConfig)
def get_zones(camera_id: str, request: Request):
    """Get saved zone config for a camera. Returns empty ZoneConfig if not set."""
    return _svc(request).get_zones(camera_id)


@router.get("/snapshot/{camera_id}")
def snapshot(camera_id: str, request: Request):
    """Raw JPEG snapshot with no annotations — used by zone-setup canvas."""
    jpeg = _svc(request).get_snapshot_jpeg(camera_id)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Camera not found or no frame yet.")
    return Response(content=jpeg, media_type="image/jpeg")


# ------------------------------------------------------------------
# Worker status (main polling endpoint)
# ------------------------------------------------------------------

@router.get("/status/{camera_id}", response_model=WorkerStatusResponse)
def get_worker_status(camera_id: str, request: Request):
    """
    **Main endpoint for the live worker dashboard.**

    Returns cached result from background poller — instant, no inference wait.

    Includes:
      - All currently tracked persons (state, pose, motion, cumulative stats)
      - Last 10 events (newest first)
      - Active shift summary with live productivity
      - stale_seconds — how long since last poll
    """
    svc = _svc(request)

    if camera_id not in [c["camera_id"] for c in svc.list_cameras()]:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    latest = svc.get_latest_status(camera_id)
    if latest is None:
        return WorkerStatusResponse(camera_id=camera_id)

    result    = latest["result"]
    polled_at = latest["polled_at"]
    stale     = (datetime.now() - polled_at).total_seconds()

    # Persons
    persons = [
        PersonStatusResponse(
            track_id        = p.track_id,
            label           = p.label,
            state           = p.state,
            pose_state      = p.pose_state,
            confidence      = p.confidence,
            motion_score    = p.motion_score,
            state_since     = p.state_since,
            total_active_s  = p.total_active_s,
            total_idle_s    = p.total_idle_s,
            total_bending_s = p.total_bending_s,
            bending_events  = p.bending_events,
            sheet_picks     = p.sheet_picks,
            in_rack_zone    = p.in_rack_zone,
        )
        for p in result.persons
    ]

    # Recent events (last 10 from log)
    log = svc.get_event_log(camera_id) or []
    recent = [
        EventRecordResponse(
            event        = e.event,
            state        = e.state,
            timestamp    = e.timestamp,
            person_track = e.person_track,
            person_label = e.person_label,
            detail       = e.detail,
        )
        for e in log[:10]
    ]

    # Active shift summary
    active_shift_summary = None
    shift = svc.get_active_shift(camera_id)
    if shift:
        live = svc.get_live_productivity(camera_id) or {}
        duration_s = (datetime.now() - shift.started_at).total_seconds()
        active_shift_summary = ActiveShiftSummary(
            shift_id           = shift.shift_id,
            worker_name        = shift.worker_name,
            started_at         = shift.started_at,
            duration_s         = round(duration_s, 1),
            live_cycles        = live.get("total_cycles", 0),
            live_sheets_per_hr = live.get("sheets_per_hour", 0.0),
            live_efficiency    = live.get("efficiency_pct", 0.0),
        )

    return WorkerStatusResponse(
        camera_id     = camera_id,
        total_persons = result.total_persons,
        persons       = persons,
        recent_events = recent,
        active_shift  = active_shift_summary,
        polled_at     = polled_at,
        stale_seconds = round(stale, 1),
    )


# ------------------------------------------------------------------
# Event log
# ------------------------------------------------------------------

@router.get("/events/{camera_id}", response_model=WorkerEventLogResponse)
def get_event_log(camera_id: str, request: Request, limit: int = 100):
    """Full event log for a camera (newest first, max 200 stored)."""
    svc = _svc(request)
    if camera_id not in [c["camera_id"] for c in svc.list_cameras()]:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    log = (svc.get_event_log(camera_id) or [])[:limit]
    return WorkerEventLogResponse(
        camera_id    = camera_id,
        total_events = len(log),
        events=[
            EventRecordResponse(
                event=e.event, state=e.state, timestamp=e.timestamp,
                person_track=e.person_track, person_label=e.person_label, detail=e.detail,
            )
            for e in log
        ],
    )


# ------------------------------------------------------------------
# Debug / stream
# ------------------------------------------------------------------

@router.get("/debug/{camera_id}")
def debug_frame(camera_id: str, request: Request):
    """JPEG with skeleton overlay + state banners for all tracked persons."""
    jpeg = _svc(request).get_debug_image(camera_id)
    if jpeg is None:
        raise HTTPException(status_code=404,
                            detail=f"Camera '{camera_id}' not found or no frame yet.")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: str, request: Request):
    """MJPEG live stream with skeleton + multi-person state banners."""
    svc = _svc(request)
    if camera_id not in [c["camera_id"] for c in svc.list_cameras()]:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")

    async def generate():
        while True:
            if await request.is_disconnected():
                break
            jpeg = svc.get_debug_image(camera_id)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
