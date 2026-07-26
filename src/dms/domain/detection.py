"""Face detection result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class FaceDetectionResult:
    """Output of a face detector for a single frame."""

    detected: bool
    bbox: BoundingBox | None
    confidence: float
