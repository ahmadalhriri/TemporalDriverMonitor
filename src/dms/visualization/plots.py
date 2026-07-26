"""Generate EDA and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from dms.evaluation.metrics import EvaluationResult
from dms.utils.io import ensure_dir


class PlotGenerator:
    """Create analysis and evaluation figures."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = ensure_dir(output_dir)
        sns.set_theme(style="whitegrid")

    def generate_all(self, df: pd.DataFrame, evaluation: EvaluationResult | None = None) -> list[Path]:
        paths: list[Path] = []
        valid = df[df["processing_success"] == True]  # noqa: E712

        for feature in ("ear_avg", "mar"):
            if feature in valid.columns:
                paths.append(self.plot_histogram(valid, feature))

        if "ear_avg" in valid.columns and "mar" in valid.columns:
            paths.append(self.plot_scatter(valid, "ear_avg", "mar"))
            paths.append(self.plot_boxplots(valid, ["ear_avg", "mar"]))

        if evaluation is not None:
            paths.append(self.plot_confusion_matrix(evaluation))

        if "mar" in valid.columns:
            paths.append(self.plot_threshold_sweep(valid, "mar"))

        numeric_cols = [
            c
            for c in [
                "ear_avg",
                "mar",
                "eye_opening",
                "mouth_opening",
                "pitch",
                "roll",
                "yaw",
                "brightness",
                "blur",
            ]
            if c in valid.columns
        ]
        if len(numeric_cols) >= 2:
            paths.append(self.plot_correlation_matrix(valid, numeric_cols))

        plt.close("all")
        return paths

    def plot_histogram(self, df: pd.DataFrame, feature: str) -> Path:
        path = self._output_dir / f"hist_{feature}.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in df.groupby("ground_truth_label"):
            sns.histplot(group[feature], label=str(label), kde=True, ax=ax, alpha=0.5)
        ax.set_title(f"{feature} distribution by class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_scatter(self, df: pd.DataFrame, x: str, y: str) -> Path:
        path = self._output_dir / f"scatter_{x}_vs_{y}.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x, y=y, hue="ground_truth_label", alpha=0.4, ax=ax)
        ax.set_title(f"{x} vs {y}")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_boxplots(self, df: pd.DataFrame, features: list[str]) -> Path:
        path = self._output_dir / "boxplots_key_features.png"
        melted = df.melt(
            id_vars=["ground_truth_label"],
            value_vars=features,
            var_name="feature",
            value_name="value",
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=melted, x="feature", y="value", hue="ground_truth_label", ax=ax)
        ax.set_title("Feature box plots by class")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_correlation_matrix(self, df: pd.DataFrame, columns: list[str]) -> Path:
        path = self._output_dir / "correlation_matrix.png"
        corr = df[columns].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature correlation matrix")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_confusion_matrix(self, evaluation: EvaluationResult) -> Path:
        path = self._output_dir / "confusion_matrix.png"
        import numpy as np

        cm = np.array(evaluation.confusion_matrix)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=evaluation.labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — Rule-Based Classifier")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_threshold_sweep(self, df: pd.DataFrame, feature: str) -> Path:
        """Visualize accuracy vs threshold for binary alert vs non-alert."""
        path = self._output_dir / f"threshold_sweep_{feature}.png"
        valid = df.copy()
        thresholds = [float(t) for t in pd.Series(valid[feature]).quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]

        fig, ax = plt.subplots(figsize=(8, 5))
        for label in valid["ground_truth_label"].unique():
            subset = valid[valid["ground_truth_label"] == label][feature]
            ax.axvline(subset.median(), label=f"{label} median", linestyle="--")
        ax.set_title(f"{feature} threshold reference lines")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
