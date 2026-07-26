"""Dataset integrity validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dms.domain.enums import Split
from dms.domain.sample import DriverSample
from dms.utils.io import save_json


@dataclass
class ValidationReport:
    """Summary of dataset validation results."""

    total_samples: int = 0
    valid_samples: int = 0
    missing_images: int = 0
    invalid_labels: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    subject_distribution: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "missing_images": self.missing_images,
            "invalid_labels": self.invalid_labels,
            "label_distribution": self.label_distribution,
            "subject_distribution": self.subject_distribution,
            "splits": self.splits,
            "issues": self.issues,
        }


class DatasetValidator:
    """Validate loaded FL3D samples."""

    def validate(self, samples: list[DriverSample]) -> ValidationReport:
        report = ValidationReport(total_samples=len(samples))

        for sample in samples:
            if not sample.image_path.exists():
                report.missing_images += 1
                report.issues.append(f"Missing image: {sample.image_path}")
                continue

            report.valid_samples += 1

        labels = Counter(s.label.value for s in samples)
        subjects = Counter(s.subject_id for s in samples)
        splits = Counter(s.split.value for s in samples)

        report.label_distribution = dict(sorted(labels.items()))
        report.subject_distribution = dict(sorted(subjects.items()))
        report.splits = dict(sorted(splits.items()))

        return report

    def validate_and_save(
        self,
        samples_by_split: dict[Split, list[DriverSample]],
        output_path: Path,
    ) -> ValidationReport:
        all_samples = [s for group in samples_by_split.values() for s in group]
        report = self.validate(all_samples)
        save_json(report.to_dict(), output_path)
        return report
