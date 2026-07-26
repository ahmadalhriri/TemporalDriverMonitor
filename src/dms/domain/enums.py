"""Enumerations for driver states and dataset splits."""

from enum import Enum


class DriverState(str, Enum):
    """Ground-truth or predicted driver state labels."""

    ALERT = "alert"
    YAWNING = "yawning"
    MICROSLEEP = "microsleep"

    @classmethod
    def from_label(cls, label: str) -> "DriverState":
        normalized = label.strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"Unknown driver state '{label}'. Expected one of: {valid}") from exc


class Split(str, Enum):
    """Predefined FL3D dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"
    ALL = "all"
