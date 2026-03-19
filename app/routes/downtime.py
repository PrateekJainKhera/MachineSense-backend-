from fastapi import APIRouter, HTTPException, Request, Query
from typing import List, Optional

from app.models.downtime_model import DowntimeRecord, DowntimeStatus, DowntimeSummary

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
