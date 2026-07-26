"""Landmark domain models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LandmarkSet:
    """Raw pixel landmarks extracted by MediaPipe Face Mesh."""

    points: NDArray[np.float64]  # shape (N, 2) — x, y in pixels
    image_width: int
    image_height: int
    schema_id: str
    confidence: float = 1.0

    @property
    def num_landmarks(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class NormalizedLandmarks:
    """Resolution- and scale-invariant landmark representation."""

    points_norm: NDArray[np.float64]  # shape (N, 2) — x/w, y/h
    points_scaled: NDArray[np.float64]  # inter-eye distance normalized to 1.0
    image_width: int
    image_height: int
    inter_eye_distance: float
    schema_id: str
