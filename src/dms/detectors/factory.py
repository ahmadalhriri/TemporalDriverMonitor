"""Face detector factory."""

from __future__ import annotations

from dms.detectors.base import FaceDetector
from dms.detectors.mediapipe_detector import MediaPipeFaceDetector
from dms.detectors.retinaface_detector import RetinaFaceDetector
from dms.utils.config import DetectorConfig


def create_face_detector(config: DetectorConfig) -> FaceDetector:
    provider = config.provider.lower()
    if provider == "mediapipe":
        return MediaPipeFaceDetector(config)
    if provider == "retinaface":
        return RetinaFaceDetector(config)
    raise ValueError(f"Unknown face detector provider: {config.provider}")
