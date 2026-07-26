"""Image quality and detection confidence features."""

from __future__ import annotations

import cv2
import numpy as np

from dms.features.base import FeatureCalculator, FeatureContext


class QualityFeatureCalculator(FeatureCalculator):
    group = "quality"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        gray = cv2.cvtColor(context.image_bgr, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray) / 255.0)

        detection_confidence = context.detection.confidence if context.detection.detected else 0.0
        bbox_area_ratio = 0.0
        if context.detection.bbox is not None:
            bbox = context.detection.bbox
            image_area = context.landmarks.image_width * context.landmarks.image_height
            bbox_area_ratio = bbox.area / max(image_area, 1)

        return {
            "blur": blur,
            "brightness": brightness,
            "detection_confidence": detection_confidence,
            "face_bbox_area_ratio": bbox_area_ratio,
        }
