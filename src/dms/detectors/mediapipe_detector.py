"""MediaPipe face detection implementation supporting legacy solutions and modern Tasks API."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from dms.domain.detection import BoundingBox, FaceDetectionResult
from dms.detectors.base import FaceDetector
from dms.utils.config import DetectorConfig
from dms.utils.models import ensure_model_asset


class MediaPipeFaceDetector(FaceDetector):
    """MediaPipe face detector supporting legacy solutions and modern Tasks API."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._is_tasks_api = False
        self._detector = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            self._mp_face_detection = mp.solutions.face_detection
            self._detector = self._mp_face_detection.FaceDetection(
                model_selection=config.model_selection,
                min_detection_confidence=config.min_detection_confidence,
            )
        else:
            self._is_tasks_api = True
            model_path = ensure_model_asset("blaze_face_short_range.tflite")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=config.min_detection_confidence,
            )
            self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image_bgr: NDArray[np.uint8]) -> FaceDetectionResult:
        height, width = image_bgr.shape[:2]

        if not self._is_tasks_api:
            image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
            results = self._detector.process(image_rgb)
            if not results.detections:
                return FaceDetectionResult(detected=False, bbox=None, confidence=0.0)

            detection = max(results.detections, key=lambda d: d.score[0] if d.score else 0.0)
            score = float(detection.score[0]) if detection.score else 0.0
            bbox_rel = detection.location_data.relative_bounding_box

            x_min = max(0.0, bbox_rel.xmin * width)
            y_min = max(0.0, bbox_rel.ymin * height)
            box_w = bbox_rel.width * width
            box_h = bbox_rel.height * height
            x_max = min(float(width), x_min + box_w)
            y_max = min(float(height), y_min + box_h)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self._detector.detect(mp_image)

            if not results.detections:
                return FaceDetectionResult(detected=False, bbox=None, confidence=0.0)

            detection = max(
                results.detections,
                key=lambda d: d.categories[0].score if d.categories else 0.0,
            )
            score = float(detection.categories[0].score) if detection.categories else 0.0
            box = detection.bounding_box

            x_min = max(0.0, float(box.origin_x))
            y_min = max(0.0, float(box.origin_y))
            x_max = min(float(width), x_min + float(box.width))
            y_max = min(float(height), y_min + float(box.height))

        bbox = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            confidence=score,
        )
        return FaceDetectionResult(detected=True, bbox=bbox, confidence=score)

    def close(self) -> None:
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()

