"""Additional geometric and symmetry features."""

from __future__ import annotations

import numpy as np

from dms.features.base import FeatureCalculator, FeatureContext
from dms.landmarks.geometry import angle_degrees, euclidean_distance, midpoint


class GeometricFeatureCalculator(FeatureCalculator):
    group = "geometric"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        points = context.points_scaled
        schema = context.schema
        raw = context.points

        left_eye = raw[schema.left_eye[0]]
        right_eye = raw[schema.right_eye[0]]
        nose = raw[schema.head_pose_indices[0]]
        chin = raw[schema.head_pose_indices[1]]
        left_mouth = raw[schema.mouth_left_corner]
        right_mouth = raw[schema.mouth_right_corner]

        eye_distance = context.normalized.inter_eye_distance
        eye_distance_norm = eye_distance / max(context.landmarks.image_width, 1)

        face_height = euclidean_distance(
            (left_eye + right_eye) / 2.0,
            (left_mouth + right_mouth) / 2.0,
        )
        face_height_ratio = face_height / max(eye_distance, 1e-6)

        mouth_width = euclidean_distance(left_mouth, right_mouth)
        face_width_ratio = mouth_width / max(face_height, 1e-6)

        symmetry_error = 0.0
        valid_pairs = 0
        for left_idx, right_idx in schema.symmetry_pairs:
            if left_idx == right_idx or left_idx >= len(points) or right_idx >= len(points):
                continue
            left_point = points[left_idx]
            right_point = points[right_idx]
            mirrored = np.array([-right_point[0], right_point[1]])
            symmetry_error += euclidean_distance(left_point, mirrored)
            valid_pairs += 1
        symmetry_score = symmetry_error / max(valid_pairs, 1)

        nose_angle = angle_degrees(left_eye, nose, right_eye)

        eye_mid = midpoint(left_eye, right_eye)
        mouth_mid = midpoint(left_mouth, right_mouth)
        vertical_alignment = abs(nose[0] - eye_mid[0]) / max(eye_distance, 1e-6)
        mouth_nose_offset = euclidean_distance(nose, mouth_mid) / max(eye_distance, 1e-6)

        return {
            "eye_distance": eye_distance,
            "eye_distance_norm": eye_distance_norm,
            "face_height_ratio": face_height_ratio,
            "face_width_ratio": face_width_ratio,
            "symmetry_score": symmetry_score,
            "nose_angle": nose_angle,
            "vertical_alignment": vertical_alignment,
            "mouth_nose_offset": mouth_nose_offset,
        }
