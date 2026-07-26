"""Abstract face detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from dms.domain.detection import FaceDetectionResult


class FaceDetector(ABC):
    """Detect faces in a BGR image frame."""

    @abstractmethod
    def detect(self, image_bgr: NDArray[np.uint8]) -> FaceDetectionResult:
        """Return the primary face detection for the frame."""

    @abstractmethod
    def close(self) -> None:
        """Release detector resources."""

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
