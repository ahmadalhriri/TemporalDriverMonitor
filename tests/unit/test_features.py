"""Tests for geometric utilities and feature calculators."""

import numpy as np

from dms.domain.detection import FaceDetectionResult
from dms.domain.landmarks import LandmarkSet, NormalizedLandmarks
from dms.features.base import FeatureContext
from dms.features.eye import _eye_aspect_ratio
from dms.features.mouth import MouthFeatureCalculator
from dms.landmarks.geometry import euclidean_distance
from dms.landmarks.schema import LandmarkSchema
from dms.utils.config import load_landmark_schema


def _make_schema() -> LandmarkSchema:
    return LandmarkSchema.from_dict(load_landmark_schema())


def test_euclidean_distance():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    assert euclidean_distance(p1, p2) == 5.0


def test_eye_aspect_ratio_open_eye():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, -2.0],
            [2.0, -2.0],
            [6.0, 0.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]
    )
    ear = _eye_aspect_ratio(points, (0, 1, 2, 3, 4, 5))
    assert ear > 0.5


def test_mouth_feature_calculator():
    schema = _make_schema()
    num_points = 478
    points = np.random.default_rng(42).random((num_points, 2)) * 100

    landmarks = LandmarkSet(
        points=points,
        image_width=320,
        image_height=240,
        schema_id=schema.schema_id,
    )
    normalized = NormalizedLandmarks(
        points_norm=points / [320, 240],
        points_scaled=points / 50.0,
        image_width=320,
        image_height=240,
        inter_eye_distance=50.0,
        schema_id=schema.schema_id,
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    context = FeatureContext(
        landmarks=landmarks,
        normalized=normalized,
        schema=schema,
        image_bgr=image,
        detection=FaceDetectionResult(detected=True, bbox=None, confidence=0.9),
    )

    features = MouthFeatureCalculator().compute(context)
    assert "mar" in features
    assert features["mar"] >= 0.0
