"""RetinaFace detector stub — optional future implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dms.domain.detection import FaceDetectionResult
from dms.detectors.base import FaceDetector
from dms.utils.config import DetectorConfig


class RetinaFaceDetector(FaceDetector):
    """Placeholder RetinaFace adapter.

    Install retina-face separately and implement when needed.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        raise NotImplementedError(
            "RetinaFace detector is not bundled. Install 'retina-face' and implement "
            "RetinaFaceDetector, or use detector.provider=mediapipe."
        )

    def detect(self, image_bgr: NDArray[np.uint8]) -> FaceDetectionResult:
        raise NotImplementedError

    def close(self) -> None:
        return None
