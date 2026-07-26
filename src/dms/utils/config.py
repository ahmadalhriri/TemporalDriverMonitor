"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from dms.utils.paths import get_project_root, resolve_path


class PathsConfig(BaseModel):
    data_root: str = "classification_frames"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    outputs_dir: str = "outputs"
    video_data_roots: list[str] = Field(default_factory=list)
    temporal_features_dir: str = "outputs/temporal_features"


class DatasetConfig(BaseModel):
    splits: list[str] = Field(default_factory=lambda: ["train", "val", "test"])
    holdout_split: str = "holdout"
    label_field: str = "driver_state"
    valid_labels: list[str] = Field(
        default_factory=lambda: ["alert", "yawning", "microsleep"]
    )
    annotation_pattern: str = "annotations_{split}.json"
    max_samples: int | None = None


class DetectorConfig(BaseModel):
    provider: str = "mediapipe"
    min_detection_confidence: float = 0.5
    model_selection: int = 0


class LandmarksConfig(BaseModel):
    provider: str = "mediapipe_face_mesh"
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    schema_path: str = "configs/landmarks_mediapipe.yaml"


class FeaturesConfig(BaseModel):
    enabled_groups: list[str] = Field(
        default_factory=lambda: ["eye", "mouth", "head", "gaze", "quality", "geometric"]
    )
    ear_blink_threshold: float = 0.21
    mar_yawn_threshold: float = 0.6


class RulesConfig(BaseModel):
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "mar_yawning": 0.55,
            "ear_microsleep": 0.20,
            "eyes_closed_ear": 0.21,
        }
    )
    priority: list[str] = Field(
        default_factory=lambda: ["microsleep", "yawning", "alert"]
    )


class PipelineConfig(BaseModel):
    batch_size: int = 32
    skip_on_detection_failure: bool = True
    save_misclassified_images: bool = False


class VideoConfig(BaseModel):
    extensions: list[str] = Field(
        default_factory=lambda: [".mov", ".mp4", ".avi", ".mkv", ".webm"]
    )
    subject_dir_pattern: str = r"^\d+$"
    export_formats: list[str] = Field(default_factory=lambda: ["parquet", "csv"])
    skip_existing: bool = True
    max_videos: int | None = None
    use_video_landmark_tracking: bool = True
    labels_filename: str = "labels.csv"


class ComputeConfig(BaseModel):
    device: str = "auto"  # auto | cpu | cuda
    mediapipe_delegate: str = "auto"  # auto | cpu | gpu
    num_workers: int = 1
    frame_buffer_size: int = 4
    log_interval_sec: float = 5.0


class EvaluationConfig(BaseModel):
    primary_split: str = "test"
    generate_failure_analysis: bool = True


class AnalysisConfig(BaseModel):
    generate_plots: bool = True
    plot_formats: list[str] = Field(default_factory=lambda: ["png"])


class ProjectConfig(BaseModel):
    name: str = "dms-phase1"
    seed: int = 42


class AppConfig(BaseModel):
    """Top-level application configuration."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    landmarks: LandmarksConfig = Field(default_factory=LandmarksConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    def data_root_path(self, base: Path | None = None) -> Path:
        return resolve_path(self.paths.data_root, base)

    def features_dir_path(self, base: Path | None = None) -> Path:
        return resolve_path(self.paths.features_dir, base)

    def outputs_dir_path(self, base: Path | None = None) -> Path:
        return resolve_path(self.paths.outputs_dir, base)

    def temporal_features_dir_path(self, base: Path | None = None) -> Path:
        return resolve_path(self.paths.temporal_features_dir, base)

    def landmark_schema_path(self, base: Path | None = None) -> Path:
        return resolve_path(self.landmarks.schema_path, base)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load and validate application config from YAML."""
    root = get_project_root()
    path = resolve_path(config_path or "configs/default.yaml", root)
    raw = load_yaml(path)

    if overrides:
        raw = _deep_merge(raw, overrides)

    return AppConfig.model_validate(raw)


def load_landmark_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load MediaPipe landmark index schema."""
    root = get_project_root()
    path = resolve_path(schema_path or "configs/landmarks_mediapipe.yaml", root)
    return load_yaml(path)
