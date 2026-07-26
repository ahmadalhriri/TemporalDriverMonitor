"""Tests for FL3D dataset loader."""

from pathlib import Path

from dms.dataset.loader import FL3DDatasetLoader
from dms.domain.enums import DriverState, Split
from dms.utils.config import load_config


def test_load_train_split_sample():
    config = load_config()
    config.dataset.max_samples = 10
    loader = FL3DDatasetLoader(config)
    samples = loader.load_split(Split.TRAIN)

    assert len(samples) == 10
    sample = samples[0]
    assert sample.image_path.exists()
    assert sample.label in DriverState
    assert sample.subject_id.startswith("P")
    assert sample.split == Split.TRAIN


def test_loader_ignores_landmark_annotations():
    """Loader must only use driver_state — no landmark fields on DriverSample."""
    config = load_config()
    config.dataset.max_samples = 1
    loader = FL3DDatasetLoader(config)
    sample = loader.load_split(Split.TRAIN)[0]
    assert not hasattr(sample, "landmarks")
