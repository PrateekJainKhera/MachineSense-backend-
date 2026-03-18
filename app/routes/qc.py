from fastapi import APIRouter, Request
from typing import Optional

from app.models.qc_model import QCEntryRequest, QCEntryResponse, QCReport

router = APIRouter(prefix="/qc", tags=["QC Management"])


def _svc(request: Request):
    return request.app.state.qc_service


def _to_response(e) -> QCEntryResponse:
    return QCEntryResponse(
        entry_id     = e.entry_id,
        camera_id    = e.camera_id,
        shift_id     = e.shift_id,
        worker_name  = e.worker_name,
        worker_label = e.worker_label,
        pick_time    = e.pick_time,
        submitted_at = e.submitted_at,
        qty_picked   = e.qty_picked,
        qty_pass     = e.qty_pass,
        qty_reject   = e.qty_reject,
        pass_rate    = e.pass_rate,
    )


# ------------------------------------------------------------------
# Submit QC form
# ------------------------------------------------------------------

@router.post("/entries", response_model=QCEntryResponse)
def submit_entry(body: QCEntryRequest, request: Request):
    """
    Submit a QC form entry after a SHEET_PICKED event.

    Triggered automatically by the frontend when the worker fills the form.
    pick_time should come from the SHEET_PICKED event timestamp.
    """
    entry = _svc(request).submit(
        camera_id    = body.camera_id,
        shift_id     = body.shift_id,
        worker_name  = body.worker_name,
        worker_label = body.worker_label,
        pick_time    = body.pick_time,
        qty_picked   = body.qty_picked,
        qty_pass     = body.qty_pass,
        qty_reject   = body.qty_reject,
    )
    return _to_response(entry)


# ------------------------------------------------------------------
# Query entries
# ------------------------------------------------------------------

@router.get("/entries", response_model=list[QCEntryResponse])
def list_entries(
    request:   Request,
    camera_id: Optional[str] = None,
    shift_id:  Optional[str] = None,
):
    """List QC entries. Filter by camera_id and/or shift_id."""
    entries = _svc(request).get_entries(camera_id=camera_id, shift_id=shift_id)
    return [_to_response(e) for e in entries]


@router.get("/report", response_model=QCReport)
def get_report(
    request:   Request,
    camera_id: Optional[str] = None,
    shift_id:  Optional[str] = None,
):
    """
    Aggregate QC report for a shift or camera.

    Returns: total entries, total picked/pass/reject, overall pass rate, entry list.
    """
    report = _svc(request).get_report(camera_id=camera_id, shift_id=shift_id)
    return QCReport(
        shift_id          = report["shift_id"],
        camera_id         = report["camera_id"],
        total_entries     = report["total_entries"],
        total_picked      = report["total_picked"],
        total_pass        = report["total_pass"],
        total_reject      = report["total_reject"],
        overall_pass_rate = report["overall_pass_rate"],
        entries           = [_to_response(e) for e in report["entries"]],
    )
