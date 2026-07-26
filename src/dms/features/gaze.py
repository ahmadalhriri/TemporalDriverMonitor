"""Gaze direction proxy features."""

from __future__ import annotations

import numpy as np

from dms.features.base import FeatureCalculator, FeatureContext
from dms.landmarks.geometry import euclidean_distance


class GazeFeatureCalculator(FeatureCalculator):
    group = "gaze"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        points = context.points
        schema = context.schema
        num_points = points.shape[0]

        left_eye_center = points[schema.left_eye[0]]
        right_eye_center = points[schema.right_eye[0]]
        eye_mid = (left_eye_center + right_eye_center) / 2.0
        nose = points[schema.head_pose_indices[0]]

        # Iris landmarks available when refine_landmarks=True (indices 468+)
        gaze_horizontal = 0.0
        gaze_vertical = 0.0
        gaze_valid = 0.0

        if (
            schema.left_iris is not None
            and schema.right_iris is not None
            and schema.left_iris < num_points
            and schema.right_iris < num_points
        ):
            left_iris = points[schema.left_iris]
            right_iris = points[schema.right_iris]

            left_offset = (left_iris - left_eye_center) / max(
                context.normalized.inter_eye_distance, 1e-6
            )
            right_offset = (right_iris - right_eye_center) / max(
                context.normalized.inter_eye_distance, 1e-6
            )
            avg_offset = (left_offset + right_offset) / 2.0
            gaze_horizontal = float(avg_offset[0])
            gaze_vertical = float(avg_offset[1])
            gaze_valid = 1.0
        else:
            # Fallback: nose-to-eye-mid vector as coarse gaze proxy
            vec = nose - eye_mid
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
                gaze_horizontal = float(vec[0])
                gaze_vertical = float(vec[1])

        gaze_magnitude = float(np.sqrt(gaze_horizontal**2 + gaze_vertical**2))

        return {
            "gaze_horizontal": gaze_horizontal,
            "gaze_vertical": gaze_vertical,
            "gaze_magnitude": gaze_magnitude,
            "gaze_valid": gaze_valid,
        }
