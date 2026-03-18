"""
QCService — stores and aggregates worker QC form submissions.

Each submission is created when a worker fills the QC form after a SHEET_PICKED event.
Entries are stored in memory (list), optionally linked to a shift_id.
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _QCEntry:
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

    @property
    def pass_rate(self) -> float:
        return round((self.qty_pass / max(self.qty_picked, 1)) * 100, 1)


class QCService:

    def __init__(self):
        self._entries: List[_QCEntry] = []
        self._counter: int = 0
        logger.info("QCService ready.")

    def submit(
        self,
        camera_id:    str,
        worker_name:  str,
        worker_label: str,
        pick_time:    datetime,
        qty_picked:   int,
        qty_pass:     int,
        qty_reject:   int,
        shift_id:     Optional[str] = None,
    ) -> _QCEntry:
        self._counter += 1
        entry = _QCEntry(
            entry_id     = self._counter,
            camera_id    = camera_id,
            shift_id     = shift_id,
            worker_name  = worker_name,
            worker_label = worker_label,
            pick_time    = pick_time,
            submitted_at = datetime.now(),
            qty_picked   = qty_picked,
            qty_pass     = qty_pass,
            qty_reject   = qty_reject,
        )
        self._entries.append(entry)
        logger.info(
            f"QC entry {entry.entry_id}: {worker_name} picked={qty_picked} "
            f"pass={qty_pass} reject={qty_reject} rate={entry.pass_rate}%"
        )
        return entry

    def get_entries(
        self,
        camera_id: Optional[str] = None,
        shift_id:  Optional[str] = None,
    ) -> List[_QCEntry]:
        entries = self._entries
        if camera_id:
            entries = [e for e in entries if e.camera_id == camera_id]
        if shift_id:
            entries = [e for e in entries if e.shift_id == shift_id]
        return sorted(entries, key=lambda e: e.submitted_at, reverse=True)

    def get_report(
        self,
        camera_id: Optional[str] = None,
        shift_id:  Optional[str] = None,
    ) -> dict:
        entries      = self.get_entries(camera_id=camera_id, shift_id=shift_id)
        total_picked = sum(e.qty_picked for e in entries)
        total_pass   = sum(e.qty_pass   for e in entries)
        total_reject = sum(e.qty_reject for e in entries)
        return {
            "shift_id":          shift_id,
            "camera_id":         camera_id,
            "total_entries":     len(entries),
            "total_picked":      total_picked,
            "total_pass":        total_pass,
            "total_reject":      total_reject,
            "overall_pass_rate": round((total_pass / max(total_picked, 1)) * 100, 1),
            "entries":           entries,
        }

    def shutdown(self) -> None:
        logger.info("QCService shut down.")
