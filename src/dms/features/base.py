"""Feature calculator base class and shared context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dms.domain.detection import FaceDetectionResult
from dms.domain.landmarks import LandmarkSet, NormalizedLandmarks
from dms.landmarks.schema import LandmarkSchema


@dataclass
class FeatureContext:
    """Shared inputs available to all feature calculators."""

    landmarks: LandmarkSet
    normalized: NormalizedLandmarks
    schema: LandmarkSchema
    image_bgr: NDArray[np.uint8]
    detection: FaceDetectionResult
    config_values: dict[str, float] = field(default_factory=dict)

    @property
    def points(self) -> NDArray[np.float64]:
        return self.landmarks.points

    @property
    def points_scaled(self) -> NDArray[np.float64]:
        return self.normalized.points_scaled


class FeatureCalculator(ABC):
    """Compute one or more explainable features."""

    group: str

    @abstractmethod
    def compute(self, context: FeatureContext) -> dict[str, float]:
        """Return feature name -> value mapping."""
