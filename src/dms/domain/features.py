"""Feature vector and processed sample models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dms.domain.enums import DriverState
from dms.domain.sample import DriverSample


@dataclass
class FeatureVector:
    """Explainable geometric/physiological features for one frame."""

    sample_id: str
    subject_id: str
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten features for DataFrame export."""
        row = {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            **self.features,
        }
        row.update({f"meta_{k}": v for k, v in self.metadata.items()})
        return row


@dataclass
class ProcessedSample:
    """Complete pipeline output for one image."""

    sample: DriverSample
    features: FeatureVector
    predicted_label: DriverState
    detection_confidence: float
    landmark_confidence: float
    processing_success: bool
    error_message: str | None = None

    def to_dataset_row(self) -> dict[str, Any]:
        """Export row for feature dataset (CSV/Parquet)."""
        row = self.features.to_flat_dict()
        row.update(
            {
                "image_id": self.sample.image_name,
                "image_path": str(self.sample.image_path),
                "split": self.sample.split.value,
                "ground_truth_label": self.sample.label.value,
                "predicted_label": self.predicted_label.value,
                "detection_confidence": self.detection_confidence,
                "landmark_confidence": self.landmark_confidence,
                "processing_success": self.processing_success,
                "error_message": self.error_message or "",
            }
        )
        return row
