"""Tests for rule engine."""

from dms.domain.features import FeatureVector
from dms.domain.enums import DriverState
from dms.rules.engine import RuleEngine
from dms.utils.config import RulesConfig


def test_rule_engine_yawning():
    engine = RuleEngine(RulesConfig())
    fv = FeatureVector(
        sample_id="test",
        subject_id="P1",
        features={"mar": 0.8, "ear_avg": 0.3, "eyes_closed": 0.0},
    )
    result = engine.predict(fv)
    assert result.predicted_label == DriverState.YAWNING


def test_rule_engine_alert():
    engine = RuleEngine(RulesConfig())
    fv = FeatureVector(
        sample_id="test",
        subject_id="P1",
        features={"mar": 0.2, "ear_avg": 0.35, "eyes_closed": 0.0},
    )
    result = engine.predict(fv)
    assert result.predicted_label == DriverState.ALERT


def test_rule_engine_microsleep():
    engine = RuleEngine(RulesConfig())
    fv = FeatureVector(
        sample_id="test",
        subject_id="P1",
        features={"mar": 0.2, "ear_avg": 0.15, "eyes_closed": 1.0},
    )
    result = engine.predict(fv)
    assert result.predicted_label == DriverState.MICROSLEEP
