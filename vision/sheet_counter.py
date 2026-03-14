import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict
from ultralytics import YOLO

logger = logging.getLogger(__name__)

ROI = Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class SheetCountResult:
    """Result returned by SheetCounter.process_frame()"""
    total_count: int           # Total sheets counted since last reset
    newly_counted: int         # Sheets counted in THIS frame (0 or more)
    active_tracks: int         # Number of objects being tracked in this frame
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "newly_counted": self.newly_counted,
            "active_tracks": self.active_tracks,
            "timestamp": self.timestamp.isoformat(),
        }


class SheetCounter:
    """
    Counts sheets coming out of a printing machine using YOLOv8 + ByteTrack.

    How it works:
        1. Camera 2 watches the sheet output area of the machine.
        2. YOLOv8 detects sheet objects in each frame.
        3. ByteTrack assigns each detected object a unique persistent ID.
        4. When a tracked object's centroid crosses the counting line,
           the count increments by 1.
        5. Each unique track ID is only counted once (no double counting).

    Counting line:
        A horizontal (or vertical) line drawn across the camera frame.
        When a sheet's center crosses this line, it is counted.
        Position is set as a pixel coordinate along the frame axis.

    Usage:
        counter = SheetCounter()
        counter.set_counting_line(position=300, orientation='horizontal')
        result = counter.process_frame(frame, roi=(x, y, w, h))
        print(result.total_count)
    """

    def __init__(self, model_path: str = "yolov8n.pt", gpu: bool = False):
        """
        Args:
            model_path: Path to YOLOv8 weights. 'yolov8n.pt' auto-downloads
                        the pretrained nano model (~6MB) on first run.
            gpu:        Use CUDA if True.
        """
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        self.device = "cuda" if gpu else "cpu"
        logger.info("YOLOv8 ready.")

        # Counting line config
        self._line_position: Optional[int] = None     # pixel coordinate
        self._line_orientation: str = "horizontal"    # 'horizontal' or 'vertical'

        # Tracking state
        self._count: int = 0
        self._counted_ids: set = set()
        self._prev_centroids: Dict[int, Tuple[int, int]] = {}  # track_id -> (cx, cy)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_counting_line(self, position: int, orientation: str = "horizontal") -> None:
        """
        Set the counting line position.

        Args:
            position:    Pixel coordinate of the line.
                         For horizontal line: y-coordinate.
                         For vertical line:   x-coordinate.
            orientation: 'horizontal' (default) or 'vertical'.
        """
        self._line_position = position
        self._line_orientation = orientation
        logger.info(f"Counting line set: {orientation} at position {position}px")

    def reset(self) -> None:
        """Reset sheet count and tracking state (e.g. start of new job)."""
        self._count = 0
        self._counted_ids.clear()
        self._prev_centroids.clear()
        logger.info("Sheet counter reset.")

    def get_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        confidence: float = 0.3,
    ) -> SheetCountResult:
        """
        Process one camera frame: detect sheets, track them, count line crossings.

        Args:
            frame:      BGR frame from camera.
            roi:        Optional (x, y, w, h) crop region.
                        Use to focus detection on the sheet exit area only.
            confidence: YOLOv8 detection confidence threshold (default 0.3).

        Returns:
            SheetCountResult with updated total count.
        """
        if self._line_position is None:
            logger.warning("Counting line not set. Call set_counting_line() first.")
            return SheetCountResult(
                total_count=self._count,
                newly_counted=0,
                active_tracks=0,
            )

        # Crop to ROI if provided
        if roi:
            x, y, w, h = roi
            working_frame = frame[y: y + h, x: x + w]
        else:
            working_frame = frame

        # Run YOLOv8 with ByteTrack
        results = self.model.track(
            working_frame,
            persist=True,          # maintain track IDs across frames
            conf=confidence,
            device=self.device,
            verbose=False,
        )

        newly_counted = 0
        active_tracks = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            # Only process detections that have track IDs assigned
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()
                active_tracks = len(track_ids)

                for track_id, box in zip(track_ids, xyxy):
                    x1, y1, x2, y2 = box

                    # Centroid of the bounding box (in ROI-local coords)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    prev = self._prev_centroids.get(track_id)

                    if prev is not None and track_id not in self._counted_ids:
                        crossed = self._check_line_crossing(prev, (cx, cy))
                        if crossed:
                            self._count += 1
                            self._counted_ids.add(track_id)
                            newly_counted += 1
                            logger.info(
                                f"Sheet counted! ID={track_id} | Total={self._count}"
                            )

                    self._prev_centroids[track_id] = (cx, cy)

        return SheetCountResult(
            total_count=self._count,
            newly_counted=newly_counted,
            active_tracks=active_tracks,
        )

    def get_debug_frame(
        self,
        frame: np.ndarray,
        roi: Optional[ROI] = None,
        result: Optional[SheetCountResult] = None,
    ) -> np.ndarray:
        """
        Annotate frame with counting line, bounding boxes, track IDs, and count.
        Returns annotated BGR frame.
        """
        debug = frame.copy()
        h, w = debug.shape[:2]

        # Draw ROI
        if roi:
            rx, ry, rw, rh = roi
            cv2.rectangle(debug, (rx, ry), (rx + rw, ry + rh), (255, 165, 0), 2)

        # Draw counting line
        if self._line_position is not None:
            if self._line_orientation == "horizontal":
                line_y = self._line_position
                cv2.line(debug, (0, line_y), (w, line_y), (0, 0, 255), 2)
                cv2.putText(debug, "COUNT LINE", (10, line_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                line_x = self._line_position
                cv2.line(debug, (line_x, 0), (line_x, h), (0, 0, 255), 2)
                cv2.putText(debug, "COUNT LINE", (line_x + 5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw sheet count overlay
        if result:
            label = f"SHEETS: {result.total_count}"
            cv2.rectangle(debug, (0, 0), (220, 45), (0, 0, 0), -1)
            cv2.putText(debug, label, (8, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        return debug

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_line_crossing(
        self,
        prev: Tuple[int, int],
        curr: Tuple[int, int],
    ) -> bool:
        """
        Returns True if centroid crossed the counting line between frames.
        Counts both directions (sheet going forward or backward — handles
        edge cases during machine startup/stop).
        """
        if self._line_orientation == "horizontal":
            line = self._line_position
            # Crossed if prev and curr are on opposite sides of the line
            return (prev[1] < line <= curr[1]) or (prev[1] >= line > curr[1])
        else:
            line = self._line_position
            return (prev[0] < line <= curr[0]) or (prev[0] >= line > curr[0])
