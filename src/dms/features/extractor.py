"""Feature extraction orchestrator."""

from __future__ import annotations

from dms.domain.features import FeatureVector
from dms.features.base import FeatureCalculator, FeatureContext
from dms.features.eye import EyeFeatureCalculator
from dms.features.geometric import GeometricFeatureCalculator
from dms.features.gaze import GazeFeatureCalculator
from dms.features.head import HeadPoseFeatureCalculator
from dms.features.mouth import MouthFeatureCalculator
from dms.features.quality import QualityFeatureCalculator
from dms.utils.config import FeaturesConfig


_CALCULATOR_REGISTRY: dict[str, type[FeatureCalculator]] = {
    "eye": EyeFeatureCalculator,
    "mouth": MouthFeatureCalculator,
    "head": HeadPoseFeatureCalculator,
    "gaze": GazeFeatureCalculator,
    "quality": QualityFeatureCalculator,
    "geometric": GeometricFeatureCalculator,
}


class FeatureExtractor:
    """Run enabled feature calculators and assemble a FeatureVector."""

    def __init__(self, config: FeaturesConfig) -> None:
        self._config = config
        self._calculators: list[FeatureCalculator] = []
        for group in config.enabled_groups:
            if group not in _CALCULATOR_REGISTRY:
                raise ValueError(f"Unknown feature group: {group}")
            self._calculators.append(_CALCULATOR_REGISTRY[group]())

    @property
    def enabled_groups(self) -> list[str]:
        return list(self._config.enabled_groups)

    def extract(
        self,
        context: FeatureContext,
        sample_id: str,
        subject_id: str,
    ) -> FeatureVector:
        features: dict[str, float] = {}
        groups_run: list[str] = []

        context.config_values = {
            "ear_blink_threshold": self._config.ear_blink_threshold,
            "mar_yawn_threshold": self._config.mar_yawn_threshold,
        }

        for calculator in self._calculators:
            computed = calculator.compute(context)
            features.update(computed)
            groups_run.append(calculator.group)

        return FeatureVector(
            sample_id=sample_id,
            subject_id=subject_id,
            features=features,
            metadata={"feature_groups": ",".join(groups_run)},
        )
