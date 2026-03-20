from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse
from typing import List, Optional
from pydantic import BaseModel

class ReasonOverride(BaseModel):
    reason: str

from app.models.downtime_model import DowntimeRecord, DowntimeStatus, DowntimeSummary, OEEResult, HistoryStats, PatternInsight

router = APIRouter(prefix="/downtime", tags=["Downtime Detection"])


def _svc(req: Request):
    return req.app.state.downtime_service


@router.post("/watch/{camera_id}", summary="Start monitoring an OCR camera for downtime")
def watch_camera(camera_id: str, req: Request, worker_camera_id: Optional[str] = Query(None)):
    _svc(req).watch(camera_id, worker_camera_id=worker_camera_id)
    return {"message": f"Watching '{camera_id}' for downtime (worker cam: {worker_camera_id})"}


@router.delete("/watch/{camera_id}", summary="Stop monitoring a camera")
def unwatch_camera(camera_id: str, req: Request):
    _svc(req).unwatch(camera_id)
    return {"message": f"Stopped watching '{camera_id}'"}


@router.get("/status/{camera_id}", response_model=DowntimeStatus, summary="Current machine running/stopped status")
def get_status(camera_id: str, req: Request):
    status = _svc(req).get_status(camera_id)
    if status is None:
        raise HTTPException(404, f"Camera '{camera_id}' not being monitored. Call POST /downtime/watch/{camera_id} first.")
    return status


@router.get("/events/{camera_id}", response_model=List[DowntimeRecord], summary="All downtime events for a camera")
def get_events(camera_id: str, req: Request):
    return _svc(req).get_events(camera_id)


@router.get("/summary/{camera_id}", response_model=DowntimeSummary, summary="Downtime totals for a camera")
def get_summary(camera_id: str, req: Request):
    return _svc(req).get_summary(camera_id)


@router.get("/history/{camera_id}", response_model=HistoryStats, summary="Level 3 — historical stats across all stops")
def get_history(camera_id: str, req: Request):
    return _svc(req).get_history(camera_id)


@router.get("/pattern/{camera_id}", response_model=PatternInsight, summary="Level 3 — stop pattern analysis")
def get_pattern(camera_id: str, req: Request):
    return _svc(req).get_pattern(camera_id)


@router.get("/export/{camera_id}", summary="Level 3 — export downtime log as CSV")
def export_csv(camera_id: str, req: Request):
    csv_data = _svc(req).get_csv(camera_id)
    return PlainTextResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="downtime_{camera_id}.csv"'},
    )


@router.put("/events/{event_id}/reason", summary="Manually override the stop reason for a downtime event")
def override_reason(event_id: int, body: ReasonOverride, req: Request):
    ok = _svc(req).override_reason(event_id, body.reason)
    if not ok:
        raise HTTPException(404, f"Event #{event_id} not found")
    return {"message": f"Event #{event_id} reason updated to '{body.reason}'"}


@router.get("/oee/{camera_id}", response_model=OEEResult, summary="OEE Availability for current shift")
def get_oee(camera_id: str, req: Request, worker_camera_id: Optional[str] = Query(None)):
    return _svc(req).get_oee(camera_id, worker_camera_id)
