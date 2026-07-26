"""Modular face detection."""

from dms.detectors.base import FaceDetector
from dms.detectors.factory import create_face_detector

__all__ = ["FaceDetector", "create_face_detector"]
