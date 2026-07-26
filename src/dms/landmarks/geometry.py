"""Pure geometric utilities for landmarks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def euclidean_distance(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(p1 - p2))


def midpoint(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> NDArray[np.float64]:
    return (p1 + p2) / 2.0


def angle_degrees(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
) -> float:
    """Return angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
