"""Misclassification and failure analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dms.utils.io import ensure_dir, save_json


class FailureAnalyzer:
    """Analyze misclassified samples and export failure reports."""

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        valid = df[df["processing_success"] == True].copy()  # noqa: E712
        misclassified = valid[valid["ground_truth_label"] != valid["predicted_label"]].copy()

        failure_by_pair: dict[str, int] = {}
        for _, row in misclassified.iterrows():
            key = f"{row['ground_truth_label']}->{row['predicted_label']}"
            failure_by_pair[key] = failure_by_pair.get(key, 0) + 1

        processing_failures = df[df["processing_success"] != True]  # noqa: E712

        per_class_errors: dict[str, dict[str, Any]] = {}
        for label in valid["ground_truth_label"].unique():
            subset = misclassified[misclassified["ground_truth_label"] == label]
            per_class_errors[label] = {
                "count": int(len(subset)),
                "top_predicted": subset["predicted_label"].value_counts().head(3).to_dict(),
            }

        return {
            "total_samples": int(len(df)),
            "valid_samples": int(len(valid)),
            "misclassified_count": int(len(misclassified)),
            "misclassification_rate": float(len(misclassified) / max(len(valid), 1)),
            "processing_failures": int(len(processing_failures)),
            "failure_by_pair": dict(sorted(failure_by_pair.items(), key=lambda x: -x[1])),
            "per_class_errors": per_class_errors,
            "misclassified_sample_ids": misclassified["sample_id"].head(100).tolist(),
        }

    def export_misclassified(self, df: pd.DataFrame, output_path: Path) -> Path:
        valid = df[df["processing_success"] == True]  # noqa: E712
        misclassified = valid[valid["ground_truth_label"] != valid["predicted_label"]]
        ensure_dir(output_path.parent)
        misclassified.to_csv(output_path, index=False)
        return output_path

    def save_report(self, df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
        report = self.analyze(df)
        save_json(report, output_path)
        return report
