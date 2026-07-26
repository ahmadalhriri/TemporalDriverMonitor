"""Temporal feature sequence container and schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Stable column order for Phase 2 deep-learning input.
METADATA_COLUMNS: list[str] = [
    "video_id",
    "subject_id",
    "fold_name",
    "video_name",
    "label",
    "frame_index",
    "timestamp_sec",
    "fps",
    "total_frames",
    "frame_width",
    "frame_height",
    "processing_success",
    "detection_confidence",
    "landmark_confidence",
    "error_message",
]

FEATURE_COLUMNS: list[str] = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "eye_opening",
    "eyes_closed",
    "blink_proxy",
    "mar",
    "mouth_opening",
    "mouth_width_norm",
    "nose_to_mouth_ratio",
    "yawning_indicator",
    "pitch",
    "yaw",
    "roll",
    "head_pose_valid",
    "gaze_horizontal",
    "gaze_vertical",
    "gaze_magnitude",
    "gaze_valid",
    "blur",
    "brightness",
    "detection_confidence",
    "face_bbox_area_ratio",
    "eye_distance",
    "eye_distance_norm",
    "face_height_ratio",
    "face_width_ratio",
    "symmetry_score",
    "nose_angle",
    "vertical_alignment",
    "mouth_nose_offset",
]

# detection_confidence appears in quality features and metadata; keep feature column
# name as exported by QualityFeatureCalculator (overwritten in row merge order).
SCHEMA_VERSION = "1.0"


@dataclass
class TemporalSequenceMeta:
    """Run metadata for one exported video sequence."""

    subject_id: str
    fold_name: str
    video_name: str
    video_id: str
    fps: float
    total_frames: int
    processed_frames: int
    successful_frames: int
    failed_frames: int
    dropped_frames: int
    frame_width: int
    frame_height: int
    device: str
    truncated: bool = False
    label: str | None = None
    feature_schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "fold_name": self.fold_name,
            "video_name": self.video_name,
            "video_id": self.video_id,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "successful_frames": self.successful_frames,
            "failed_frames": self.failed_frames,
            "dropped_frames": self.dropped_frames,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "device": self.device,
            "truncated": self.truncated,
            "label": self.label,
            "feature_schema_version": self.feature_schema_version,
            **self.extra,
        }


@dataclass
class TemporalFeatureSequence:
    """Ordered temporal table for one video."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    meta: TemporalSequenceMeta | None = None

    @property
    def column_order(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for col in METADATA_COLUMNS + FEATURE_COLUMNS:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
        for row in self.rows:
            for key in row:
                if key not in seen:
                    ordered.append(key)
                    seen.add(key)
        return ordered
