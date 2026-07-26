"""Compute classification metrics and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dms.domain.enums import DriverState


@dataclass
class EvaluationResult:
    """Complete evaluation output."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    f1_per_class: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]
    classification_report: str
    support_per_class: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "classification_report": self.classification_report,
            "support_per_class": self.support_per_class,
        }


class ModelEvaluator:
    """Evaluate predictions against FL3D ground truth."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self._labels = labels or [s.value for s in DriverState]

    def evaluate(self, y_true: list[str], y_pred: list[str]) -> EvaluationResult:
        labels = self._labels
        accuracy = float(accuracy_score(y_true, y_pred))
        precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
        recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))

        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0, labels=labels
        )
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0, labels=labels
        )
        f1_per_class = f1_score(
            y_true, y_pred, average=None, zero_division=0, labels=labels
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

        support = pd.Series(y_true).value_counts().reindex(labels, fill_value=0).astype(int)

        return EvaluationResult(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_per_class={labels[i]: float(precision_per_class[i]) for i in range(len(labels))},
            recall_per_class={labels[i]: float(recall_per_class[i]) for i in range(len(labels))},
            f1_per_class={labels[i]: float(f1_per_class[i]) for i in range(len(labels))},
            confusion_matrix=cm.tolist(),
            labels=labels,
            classification_report=report,
            support_per_class={labels[i]: int(support.iloc[i]) for i in range(len(labels))},
        )

    def evaluate_dataframe(self, df: pd.DataFrame) -> EvaluationResult:
        mask = df["processing_success"] == True  # noqa: E712
        valid = df[mask]
        return self.evaluate(
            valid["ground_truth_label"].tolist(),
            valid["predicted_label"].tolist(),
        )
