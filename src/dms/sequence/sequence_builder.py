"""Build ordered temporal feature tables."""

from __future__ import annotations

from typing import Any

from dms.domain.temporal_sequence import (
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    TemporalFeatureSequence,
    TemporalSequenceMeta,
)
from dms.domain.video_sample import FrameResult, VideoSample
from dms.video.video_reader import VideoMetadata


class SequenceBuilder:
    """Accumulate per-frame rows preserving temporal order."""

    def __init__(self, sample: VideoSample, metadata: VideoMetadata, device: str) -> None:
        self._sample = sample
        self._metadata = metadata
        self._device = device
        self._rows: list[dict[str, Any]] = []
        self._successful = 0
        self._failed = 0
        self._video_meta = {
            "fps": metadata.fps,
            "total_frames": metadata.total_frames,
            "frame_width": metadata.frame_width,
            "frame_height": metadata.frame_height,
        }

    def append(self, result: FrameResult) -> None:
        row = result.to_row(self._sample, self._video_meta)
        for col in FEATURE_COLUMNS:
            row.setdefault(col, float("nan"))
        self._rows.append(row)
        if result.processing_success:
            self._successful += 1
        else:
            self._failed += 1

    def build(
        self,
        dropped_frames: int = 0,
        truncated: bool = False,
    ) -> TemporalFeatureSequence:
        meta = TemporalSequenceMeta(
            subject_id=self._sample.subject_id,
            fold_name=self._sample.fold_name,
            video_name=self._sample.video_name,
            video_id=self._sample.video_id,
            fps=self._metadata.fps,
            total_frames=self._metadata.total_frames,
            processed_frames=len(self._rows),
            successful_frames=self._successful,
            failed_frames=self._failed,
            dropped_frames=dropped_frames,
            frame_width=self._metadata.frame_width,
            frame_height=self._metadata.frame_height,
            device=self._device,
            truncated=truncated,
            label=self._sample.label,
        )
        return TemporalFeatureSequence(rows=self._rows, meta=meta)

    @staticmethod
    def empty_columns() -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for col in METADATA_COLUMNS + FEATURE_COLUMNS:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
        return ordered
