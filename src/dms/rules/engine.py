"""Scientific rule engine for driver state classification."""

from __future__ import annotations

from dataclasses import dataclass

from dms.domain.enums import DriverState
from dms.domain.features import FeatureVector
from dms.utils.config import RulesConfig


@dataclass(frozen=True)
class RulePrediction:
    """Output of the rule engine for one sample."""

    predicted_label: DriverState
    matched_rule: str
    rule_scores: dict[str, float]


class RuleEngine:
    """Evaluate physiological rules against extracted features.

    Priority order (configurable):
    1. Microsleep — low EAR + eyes closed
    2. Yawning — high MAR
    3. Alert — default
    """

    def __init__(self, config: RulesConfig) -> None:
        self._config = config
        self._thresholds = config.thresholds
        self._priority = [DriverState.from_label(p) for p in config.priority]

    def predict(self, features: FeatureVector) -> RulePrediction:
        f = features.features
        scores = {
            "microsleep": self._microsleep_score(f),
            "yawning": self._yawning_score(f),
            "alert": 0.0,
        }

        ear_avg = f.get("ear_avg", 1.0)
        eyes_closed = f.get("eyes_closed", 0.0)
        mar = f.get("mar", 0.0)

        mar_threshold = self._thresholds.get("mar_yawning", 0.55)
        ear_microsleep = self._thresholds.get("ear_microsleep", 0.20)
        eyes_closed_ear = self._thresholds.get("eyes_closed_ear", 0.21)

        # Evaluate rules in priority order
        if ear_avg < ear_microsleep and eyes_closed >= 0.5:
            return RulePrediction(
                predicted_label=DriverState.MICROSLEEP,
                matched_rule="ear_microsleep_and_eyes_closed",
                rule_scores=scores,
            )

        if mar > mar_threshold:
            return RulePrediction(
                predicted_label=DriverState.YAWNING,
                matched_rule="mar_above_threshold",
                rule_scores=scores,
            )

        if ear_avg < eyes_closed_ear:
            # Low EAR without full microsleep criteria — still drowsy signal
            scores["alert"] = 0.5
            return RulePrediction(
                predicted_label=DriverState.MICROSLEEP,
                matched_rule="ear_below_threshold",
                rule_scores=scores,
            )

        return RulePrediction(
            predicted_label=DriverState.ALERT,
            matched_rule="default_alert",
            rule_scores=scores,
        )

    @staticmethod
    def _microsleep_score(features: dict[str, float]) -> float:
        ear = features.get("ear_avg", 1.0)
        closed = features.get("eyes_closed", 0.0)
        return float(closed * max(0.0, 0.25 - ear) / 0.25)

    @staticmethod
    def _yawning_score(features: dict[str, float]) -> float:
        mar = features.get("mar", 0.0)
        return float(min(1.0, mar / 0.8))
