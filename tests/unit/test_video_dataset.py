"""Tests for video dataset discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.dataset.video_dataset import VideoDatasetLoader
from dms.utils.config import AppConfig, VideoConfig


def _make_config(root_name: str) -> AppConfig:
    return AppConfig(
        paths={
            "video_data_roots": [root_name],
        },
        video=VideoConfig(
            extensions=[".mov", ".mp4"],
            max_videos=None,
        ),
    )


def test_discover_nested_fold_layout(tmp_path: Path):
    fold = tmp_path / "Fold1_part1"
    inner = fold / "Fold1_part1"
    subject = inner / "01"
    subject.mkdir(parents=True)
    (subject / "0.mov").write_bytes(b"fake")
    (subject / "5.mov").write_bytes(b"fake")
    subj02 = inner / "02"
    subj02.mkdir(parents=True)
    (subj02 / "10.mov").write_bytes(b"fake")

    config = _make_config(str(fold))
    loader = VideoDatasetLoader(config, project_root=tmp_path)
    samples = loader.discover()

    assert len(samples) == 3
    subjects = {s.subject_id for s in samples}
    assert subjects == {"01", "02"}
    stems = {s.video_stem for s in samples}
    assert stems == {"0", "5", "10"}
    assert all(s.fold_name == "Fold1_part1" for s in samples)


def test_discover_flat_subject_layout(tmp_path: Path):
    fold = tmp_path / "Fold2"
    subject = fold / "03"
    subject.mkdir(parents=True)
    (subject / "video.mp4").write_bytes(b"fake")

    config = _make_config(str(fold))
    loader = VideoDatasetLoader(config, project_root=tmp_path)
    samples = loader.discover()

    assert len(samples) == 1
    assert samples[0].subject_id == "03"
    assert samples[0].video_name == "video.mp4"


def test_max_videos_limit(tmp_path: Path):
    fold = tmp_path / "Fold1_part1"
    subject = fold / "01"
    subject.mkdir(parents=True)
    for name in ("0.mov", "1.mov", "2.mov"):
        (subject / name).write_bytes(b"x")

    config = AppConfig(
        paths={"video_data_roots": [str(fold)]},
        video=VideoConfig(max_videos=2),
    )
    loader = VideoDatasetLoader(config, project_root=tmp_path)
    samples = loader.discover()
    assert len(samples) == 2


def test_optional_labels_csv(tmp_path: Path):
    fold = tmp_path / "Fold1_part1"
    subject = fold / "01"
    subject.mkdir(parents=True)
    (subject / "0.mov").write_bytes(b"fake")
    (fold / "labels.csv").write_text("video_path,label\n01/0.mov,alert\n", encoding="utf-8")

    config = _make_config(str(fold))
    loader = VideoDatasetLoader(config, project_root=tmp_path)
    samples = loader.discover()

    assert len(samples) == 1
    assert samples[0].label == "alert"
