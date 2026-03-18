from fastapi import APIRouter, HTTPException, Request
from datetime import datetime

from app.models.shift_model import (
    StartShiftRequest,
    ShiftResponse,
    ShiftEventResponse,
    ProductivityReport,
    PersonProductivity,
)

router = APIRouter(prefix="/shifts", tags=["Shift Management"])


def _svc(request: Request):
    return request.app.state.worker_service


def _to_shift_response(shift) -> ShiftResponse:
    duration_s = (
        (shift.ended_at - shift.started_at).total_seconds()
        if shift.ended_at else
        (datetime.now() - shift.started_at).total_seconds()
    )
    return ShiftResponse(
        shift_id    = shift.shift_id,
        worker_name = shift.worker_name,
        camera_id   = shift.camera_id,
        started_at  = shift.started_at,
        ended_at    = shift.ended_at,
        notes       = shift.notes,
        is_active   = shift.is_active,
        duration_s  = round(duration_s, 1),
        event_count = len(shift.events),
    )


# ------------------------------------------------------------------
# Shift lifecycle
# ------------------------------------------------------------------

@router.post("/start", response_model=ShiftResponse)
def start_shift(body: StartShiftRequest, request: Request):
    """
    Start a new worker shift on a camera.

    Call this when a worker starts their shift.
    The system immediately begins recording all worker events
    and accumulating productivity stats for this shift.

    Example:
        { "camera_id": "machine_a_worker", "worker_name": "Ahmed", "notes": "Job #1042" }
    """
    svc = _svc(request)
    try:
        shift = svc.start_shift(
            camera_id   = body.camera_id,
            worker_name = body.worker_name,
            notes       = body.notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _to_shift_response(shift)


@router.post("/{shift_id}/end", response_model=ProductivityReport)
def end_shift(shift_id: str, request: Request):
    """
    End a shift and get the full productivity report.

    Returns:
      - total_cycles: estimated sheet pick+return cycles
      - sheets_per_hour: cycles per hour of shift
      - efficiency_pct: active time / (active + idle) × 100
      - per-person breakdown
    """
    svc   = _svc(request)
    shift = svc.get_shift(shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift '{shift_id}' not found.")
    if not shift.is_active:
        raise HTTPException(status_code=400, detail=f"Shift '{shift_id}' already ended.")

    report = svc.end_shift(shift.camera_id)
    if not report:
        raise HTTPException(status_code=500, detail="Failed to end shift.")

    return _build_productivity_response(report)


@router.post("/end/{camera_id}", response_model=ProductivityReport)
def end_shift_by_camera(camera_id: str, request: Request):
    """End the active shift on a camera (alternative — use camera_id instead of shift_id)."""
    svc   = _svc(request)
    shift = svc.get_active_shift(camera_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"No active shift on camera '{camera_id}'.")

    report = svc.end_shift(camera_id)
    if not report:
        raise HTTPException(status_code=500, detail="Failed to end shift.")

    return _build_productivity_response(report)


# ------------------------------------------------------------------
# Queries
# ------------------------------------------------------------------

@router.get("/active/{camera_id}", response_model=ShiftResponse)
def get_active_shift(camera_id: str, request: Request):
    """Get the currently active shift on a camera, including live productivity."""
    svc   = _svc(request)
    shift = svc.get_active_shift(camera_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"No active shift on camera '{camera_id}'.")

    resp = _to_shift_response(shift)

    # Add live productivity
    live = svc.get_live_productivity(camera_id)
    if live:
        resp.live_cycles        = live["total_cycles"]
        resp.live_sheets_per_hr = live["sheets_per_hour"]
        resp.live_efficiency    = live["efficiency_pct"]

    return resp


@router.get("", response_model=list[ShiftResponse])
def list_shifts(request: Request, camera_id: str = None):
    """List all shifts. Optionally filter by camera_id."""
    svc    = _svc(request)
    shifts = svc.list_shifts(camera_id=camera_id)
    return [_to_shift_response(s) for s in shifts]


@router.get("/{shift_id}", response_model=ShiftResponse)
def get_shift(shift_id: str, request: Request):
    """Get a specific shift by ID."""
    svc   = _svc(request)
    shift = svc.get_shift(shift_id)
    if not shift:
        raise HTTPException(status_code=404, detail=f"Shift '{shift_id}' not found.")
    return _to_shift_response(shift)


@router.get("/{shift_id}/events", response_model=list[ShiftEventResponse])
def get_shift_events(shift_id: str, request: Request, limit: int = 100):
    """Get all events recorded during a shift (newest first)."""
    svc    = _svc(request)
    events = svc.get_shift_events(shift_id)
    if events is None:
        raise HTTPException(status_code=404, detail=f"Shift '{shift_id}' not found.")

    events = list(reversed(events))[:limit]
    return [
        ShiftEventResponse(
            event        = e["event"],
            state        = e["state"],
            person_track = e.get("person_track", 0),
            person_label = e.get("person_label", ""),
            timestamp    = e["timestamp"],
            detail       = e.get("detail", ""),
        )
        for e in events
    ]


@router.get("/{shift_id}/productivity", response_model=ProductivityReport)
def get_productivity(shift_id: str, request: Request):
    """Get the final productivity report for a completed shift."""
    svc    = _svc(request)
    report = svc.get_productivity_report(shift_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Shift '{shift_id}' not found or still active (use /shifts/active/{{camera_id}} for live stats).",
        )
    return _build_productivity_response(report)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _build_productivity_response(report: dict) -> ProductivityReport:
    return ProductivityReport(
        shift_id        = report["shift_id"],
        worker_name     = report["worker_name"],
        camera_id       = report["camera_id"],
        started_at      = report["started_at"],
        ended_at        = report["ended_at"],
        duration_s      = report["duration_s"],
        duration_fmt    = report["duration_fmt"],
        total_cycles    = report["total_cycles"],
        sheets_per_hour = report["sheets_per_hour"],
        efficiency_pct  = report["efficiency_pct"],
        persons=[
            PersonProductivity(
                person_label     = p["person_label"],
                active_s         = p["active_s"],
                idle_s           = p["idle_s"],
                bending_s        = p["bending_s"],
                bending_events   = p["bending_events"],
                estimated_cycles = p["estimated_cycles"],
            )
            for p in report["persons"]
        ],
    )
