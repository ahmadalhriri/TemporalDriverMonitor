"""Automated exploratory data analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from dms.utils.io import save_json


FEATURE_COLUMNS = [
    "ear_avg",
    "mar",
    "eye_opening",
    "mouth_opening",
    "pitch",
    "roll",
    "yaw",
    "gaze_magnitude",
    "brightness",
    "blur",
    "symmetry_score",
]


@dataclass
class EDAReport:
    """Summary statistics from exploratory analysis."""

    feature_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    class_wise_summary: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    separability: dict[str, float] = field(default_factory=dict)
    correlation_matrix: list[list[float]] = field(default_factory=list)
    correlation_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_statistics": self.feature_statistics,
            "class_wise_summary": self.class_wise_summary,
            "separability": self.separability,
            "correlation_matrix": self.correlation_matrix,
            "correlation_features": self.correlation_features,
        }


class ExploratoryAnalyzer:
    """Generate statistical summaries for extracted features."""

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self._features = feature_columns or FEATURE_COLUMNS

    def analyze(self, df: pd.DataFrame) -> EDAReport:
        valid = df[df["processing_success"] == True].copy()  # noqa: E712
        available = [c for c in self._features if c in valid.columns]

        report = EDAReport()
        report.feature_statistics = self._global_statistics(valid, available)
        report.class_wise_summary = self._class_wise_summary(valid, available)
        report.separability = self._separability(valid, available)
        report.correlation_features = available
        if available:
            corr = valid[available].corr(numeric_only=True).fillna(0.0)
            report.correlation_matrix = corr.values.tolist()

        return report

    def save_report(self, df: pd.DataFrame, output_path) -> EDAReport:
        report = self.analyze(df)
        save_json(report.to_dict(), output_path)
        return report

    @staticmethod
    def _global_statistics(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
        stats_map: dict[str, dict[str, float]] = {}
        for col in columns:
            series = df[col].dropna()
            stats_map[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }
        return stats_map

    @staticmethod
    def _class_wise_summary(
        df: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        summary: dict[str, dict[str, dict[str, float]]] = {}
        for label, group in df.groupby("ground_truth_label"):
            summary[str(label)] = {}
            for col in columns:
                series = group[col].dropna()
                summary[str(label)][col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "median": float(series.median()),
                    "count": int(len(series)),
                }
        return summary

    @staticmethod
    def _separability(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        """Cohen's d between alert and yawning for each feature."""
        separability: dict[str, float] = {}
        if "alert" not in df["ground_truth_label"].values:
            return separability
        if "yawning" not in df["ground_truth_label"].values:
            return separability

        alert = df[df["ground_truth_label"] == "alert"]
        yawning = df[df["ground_truth_label"] == "yawning"]

        for col in columns:
            a = alert[col].dropna()
            y = yawning[col].dropna()
            if len(a) < 2 or len(y) < 2:
                continue
            pooled_std = np.sqrt((a.std() ** 2 + y.std() ** 2) / 2.0)
            if pooled_std < 1e-8:
                separability[col] = 0.0
            else:
                separability[col] = float(abs(a.mean() - y.mean()) / pooled_std)

            # Mann-Whitney p-value stored alongside via naming convention in metadata
            _, p_value = stats.mannwhitneyu(a, y, alternative="two-sided")
            separability[f"{col}_mannwhitney_p"] = float(p_value)

        return separability
