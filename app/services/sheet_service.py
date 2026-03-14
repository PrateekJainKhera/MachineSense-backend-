import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from vision.camera_manager import CameraManager
from vision.sheet_counter import SheetCounter, SheetCountResult
from app.models.sheet_model import CountingLineConfig, SheetCameraConfig

logger = logging.getLogger(__name__)


class SheetService:
    """
    Manages sheet counting cameras and SheetCounter instances.

    One SheetCounter per camera — each camera tracks its own count,
    counting line, and tracked object IDs independently.

    Created once at app startup and shared across all requests.
    """

    def __init__(self):
        logger.info("Starting SheetService...")
        self._cameras: Dict[str, CameraManager] = {}
        self._counters: Dict[str, SheetCounter] = {}
        self._confidences: Dict[str, float] = {}
        logger.info("SheetService ready.")

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def register_camera(self, config: SheetCameraConfig) -> None:
        """
        Register a camera and create its SheetCounter instance.

        Args:
            config: Camera ID, source, optional counting line, confidence.
        """
        camera_id = config.camera_id

        # Stop existing camera if re-registering
        if camera_id in self._cameras:
            self._cameras[camera_id].stop()

        # Parse source
        parsed_source = int(config.source) if config.source.isdigit() else config.source

        cam = CameraManager(source=parsed_source, camera_id=camera_id)
        cam.start()
        self._cameras[camera_id] = cam

        counter = SheetCounter(gpu=False)
        if config.counting_line:
            counter.set_counting_line(
                position=config.counting_line.position,
                orientation=config.counting_line.orientation,
            )
        self._counters[camera_id] = counter
        self._confidences[camera_id] = config.confidence

        logger.info(f"Sheet camera '{camera_id}' registered.")

    def unregister_camera(self, camera_id: str) -> bool:
        """Stop and remove a sheet camera. Returns False if not found."""
        if camera_id not in self._cameras:
            return False
        self._cameras[camera_id].stop()
        del self._cameras[camera_id]
        del self._counters[camera_id]
        self._confidences.pop(camera_id, None)
        logger.info(f"Sheet camera '{camera_id}' removed.")
        return True

    def set_counting_line(self, camera_id: str, line: CountingLineConfig) -> bool:
        """Update the counting line for a registered camera."""
        counter = self._counters.get(camera_id)
        if not counter:
            return False
        counter.set_counting_line(position=line.position, orientation=line.orientation)
        return True

    def reset_count(self, camera_id: str) -> Optional[int]:
        """
        Reset sheet count for a camera.
        Returns the count before reset, or None if camera not found.
        """
        counter = self._counters.get(camera_id)
        if not counter:
            return None
        previous = counter.get_count()
        counter.reset()
        logger.info(f"Sheet count reset for '{camera_id}'. Previous count: {previous}")
        return previous

    def get_status(self, camera_id: str) -> Optional[dict]:
        """Return status dict for a camera."""
        cam = self._cameras.get(camera_id)
        counter = self._counters.get(camera_id)
        if not cam or not counter:
            return None
        return {
            "camera_id": camera_id,
            "source": str(cam.source),
            "connected": cam.is_connected(),
            "has_frame": cam.get_frame() is not None,
            "total_count": counter.get_count(),
            "counting_line": {
                "position": counter._line_position,
                "orientation": counter._line_orientation,
            } if counter._line_position is not None else None,
        }

    def list_cameras(self) -> list:
        return [self.get_status(cid) for cid in self._cameras]

    # ------------------------------------------------------------------
    # Sheet counting operations
    # ------------------------------------------------------------------

    def process_latest_frame(self, camera_id: str) -> Optional[SheetCountResult]:
        """
        Grab latest frame from camera and run sheet detection + counting.
        Returns None if camera not found or no frame available.
        """
        cam = self._cameras.get(camera_id)
        counter = self._counters.get(camera_id)
        if not cam or not counter:
            return None

        frame = cam.get_frame()
        if frame is None:
            return None

        confidence = self._confidences.get(camera_id, 0.3)
        return counter.process_frame(frame, confidence=confidence)

    def get_count(self, camera_id: str) -> Optional[int]:
        """Return current total sheet count for a camera."""
        counter = self._counters.get(camera_id)
        return counter.get_count() if counter else None

    def get_debug_image(self, camera_id: str) -> Optional[bytes]:
        """
        Return a JPEG-encoded debug image with counting line,
        bounding boxes, and sheet count drawn on the latest frame.
        """
        cam = self._cameras.get(camera_id)
        counter = self._counters.get(camera_id)
        if not cam or not counter:
            return None

        frame = cam.get_frame()
        if frame is None:
            return None

        confidence = self._confidences.get(camera_id, 0.3)
        result = counter.process_frame(frame, confidence=confidence)
        debug_frame = counter.get_debug_frame(frame, result=result)

        _, buffer = cv2.imencode(".jpg", debug_frame)
        return buffer.tobytes()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self):
        logger.info("Shutting down SheetService...")
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()
        self._counters.clear()
        logger.info("SheetService shut down.")
