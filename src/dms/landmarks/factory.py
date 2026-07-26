"""Landmark extractor factory."""

from __future__ import annotations

from dms.landmarks.base import LandmarkExtractor
from dms.landmarks.mediapipe_mesh import MediaPipeFaceMeshExtractor
from dms.landmarks.schema import LandmarkSchema
from dms.utils.config import LandmarksConfig, load_landmark_schema


def create_landmark_extractor(config: LandmarksConfig) -> LandmarkExtractor:
    schema = LandmarkSchema.from_dict(load_landmark_schema(config.schema_path))
    provider = config.provider.lower()

    if provider == "mediapipe_face_mesh":
        return MediaPipeFaceMeshExtractor(config, schema)

    raise ValueError(f"Unknown landmark provider: {config.provider}")
