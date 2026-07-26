"""Landmark normalization and scale invariance."""

from __future__ import annotations

import numpy as np

from dms.domain.landmarks import LandmarkSet, NormalizedLandmarks
from dms.landmarks.geometry import euclidean_distance
from dms.landmarks.schema import LandmarkSchema


class LandmarkProcessor:
    """Convert raw landmarks into normalized, scale-invariant form."""

    def __init__(self, schema: LandmarkSchema) -> None:
        self._schema = schema

    def process(self, landmarks: LandmarkSet) -> NormalizedLandmarks:
        width = landmarks.image_width
        height = landmarks.image_height
        points = landmarks.points

        points_norm = points.copy()
        points_norm[:, 0] /= max(width, 1)
        points_norm[:, 1] /= max(height, 1)

        left_eye_idx = self._schema.left_eye[0]
        right_eye_idx = self._schema.right_eye[0]
        inter_eye = euclidean_distance(points[left_eye_idx], points[right_eye_idx])
        inter_eye = max(inter_eye, 1e-6)

        points_scaled = points.copy()
        points_scaled[:, 0] = (points[:, 0] - points[left_eye_idx, 0]) / inter_eye
        points_scaled[:, 1] = (points[:, 1] - points[left_eye_idx, 1]) / inter_eye

        return NormalizedLandmarks(
            points_norm=points_norm,
            points_scaled=points_scaled,
            image_width=width,
            image_height=height,
            inter_eye_distance=inter_eye,
            schema_id=landmarks.schema_id,
        )
