"""Video sample and per-frame processing result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoSample:
    """One video file belonging to a subject within a dataset fold."""

    video_id: str
    video_path: Path
    subject_id: str
    fold_name: str
    video_name: str
    label: str | None = None

    @property
    def video_stem(self) -> str:
        return Path(self.video_name).stem

    @property
    def subject_dir_name(self) -> str:
        return f"Subject_{self.subject_id}"


@dataclass
class FrameResult:
    """Pipeline output for a single video frame."""

    frame_index: int
    timestamp_sec: float
    processing_success: bool
    features: dict[str, float] = field(default_factory=dict)
    detection_confidence: float = 0.0
    landmark_confidence: float = 0.0
    error_message: str | None = None

    def to_row(
        self,
        sample: VideoSample,
        video_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Flatten to a temporal sequence row."""
        row: dict[str, Any] = {
            "video_id": sample.video_id,
            "subject_id": sample.subject_id,
            "fold_name": sample.fold_name,
            "video_name": sample.video_name,
            "label": sample.label or "",
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "fps": video_meta.get("fps", 0.0),
            "total_frames": video_meta.get("total_frames", 0),
            "frame_width": video_meta.get("frame_width", 0),
            "frame_height": video_meta.get("frame_height", 0),
            "processing_success": self.processing_success,
            "detection_confidence": self.detection_confidence,
            "landmark_confidence": self.landmark_confidence,
            "error_message": self.error_message or "",
        }
        row.update(self.features)
        return row
