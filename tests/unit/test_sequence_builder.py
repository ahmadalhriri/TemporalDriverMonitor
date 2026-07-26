"""Tests for temporal sequence builder."""

from __future__ import annotations

import math

from dms.domain.video_sample import FrameResult, VideoSample
from dms.sequence.sequence_builder import SequenceBuilder
from dms.video.video_reader import VideoMetadata


def _sample() -> VideoSample:
    return VideoSample(
        video_id="Fold1:01:0",
        video_path=__file__,  # type: ignore[arg-type]
        subject_id="01",
        fold_name="Fold1_part1",
        video_name="0.mov",
        label="alert",
    )


def _metadata() -> VideoMetadata:
    return VideoMetadata(fps=30.0, total_frames=3, frame_width=640, frame_height=480)


def test_sequence_preserves_frame_order():
    builder = SequenceBuilder(_sample(), _metadata(), device="cpu")

    builder.append(
        FrameResult(
            frame_index=0,
            timestamp_sec=0.0,
            processing_success=True,
            features={"ear_avg": 0.3, "mar": 0.1},
            detection_confidence=0.9,
        )
    )
    builder.append(
        FrameResult(
            frame_index=1,
            timestamp_sec=0.033,
            processing_success=False,
            features={"ear_avg": math.nan},
            error_message="Face not detected",
        )
    )

    sequence = builder.build()
    assert len(sequence.rows) == 2
    assert sequence.rows[0]["frame_index"] == 0
    assert sequence.rows[1]["frame_index"] == 1
    assert sequence.rows[0]["ear_avg"] == 0.3
    assert math.isnan(sequence.rows[1]["ear_avg"])
    assert sequence.meta is not None
    assert sequence.meta.successful_frames == 1
    assert sequence.meta.failed_frames == 1
