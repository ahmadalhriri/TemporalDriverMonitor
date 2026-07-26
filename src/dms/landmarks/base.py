"""Abstract landmark extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from dms.domain.landmarks import LandmarkSet


class LandmarkExtractor(ABC):
    """Extract dense face landmarks from a BGR image."""

    @abstractmethod
    def extract(self, image_bgr: NDArray[np.uint8]) -> LandmarkSet | None:
        """Return landmarks for the primary face, or None if extraction fails."""

    @abstractmethod
    def close(self) -> None:
        """Release extractor resources."""

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
