"""Mouth-related features: MAR, opening, yawning indicator."""

from __future__ import annotations

from dms.features.base import FeatureCalculator, FeatureContext
from dms.landmarks.geometry import euclidean_distance, midpoint


class MouthFeatureCalculator(FeatureCalculator):
    group = "mouth"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        points = context.points
        schema = context.schema
        mar_threshold = context.config_values.get("mar_yawn_threshold", 0.6)

        upper = points[schema.mouth_upper_lip]
        lower = points[schema.mouth_lower_lip]
        left = points[schema.mouth_left_corner]
        right = points[schema.mouth_right_corner]

        vertical = euclidean_distance(upper, lower)
        horizontal = euclidean_distance(left, right)
        mar = vertical / max(horizontal, 1e-8)

        mouth_opening = vertical / max(context.normalized.inter_eye_distance, 1e-6)
        mouth_width_norm = horizontal / max(context.normalized.inter_eye_distance, 1e-6)
        mouth_center = midpoint(left, right)
        nose = points[schema.head_pose_indices[0]]
        nose_to_mouth = euclidean_distance(nose, mouth_center) / max(
            context.normalized.inter_eye_distance, 1e-6
        )

        yawning_indicator = float(mar > mar_threshold)

        return {
            "mar": mar,
            "mouth_opening": mouth_opening,
            "mouth_width_norm": mouth_width_norm,
            "nose_to_mouth_ratio": nose_to_mouth,
            "yawning_indicator": yawning_indicator,
        }
