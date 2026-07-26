"""Eye-related features: EAR, opening, blink proxy."""

from __future__ import annotations

import numpy as np

from dms.features.base import FeatureCalculator, FeatureContext
from dms.landmarks.geometry import euclidean_distance


def _eye_aspect_ratio(points: np.ndarray, indices: tuple[int, ...]) -> float:
    """Standard 6-point Eye Aspect Ratio (Soukupová & Čech)."""
    p1, p2, p3, p4, p5, p6 = (points[i] for i in indices)
    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal < 1e-8:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EyeFeatureCalculator(FeatureCalculator):
    group = "eye"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        points = context.points
        schema = context.schema
        blink_threshold = context.config_values.get("ear_blink_threshold", 0.21)

        ear_left = _eye_aspect_ratio(points, schema.left_eye)
        ear_right = _eye_aspect_ratio(points, schema.right_eye)
        ear_avg = (ear_left + ear_right) / 2.0

        left_height = euclidean_distance(points[schema.left_eye[1]], points[schema.left_eye[5]])
        right_height = euclidean_distance(points[schema.right_eye[1]], points[schema.right_eye[5]])
        eye_opening = (left_height + right_height) / (2.0 * max(context.normalized.inter_eye_distance, 1e-6))

        eyes_closed = float(ear_avg < blink_threshold)
        blink_proxy = eyes_closed  # single-frame proxy; temporal blink in Phase 2

        return {
            "ear_left": ear_left,
            "ear_right": ear_right,
            "ear_avg": ear_avg,
            "eye_opening": eye_opening,
            "eyes_closed": eyes_closed,
            "blink_proxy": blink_proxy,
        }
