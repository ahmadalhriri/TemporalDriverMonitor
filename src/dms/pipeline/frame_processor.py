"""Single-frame CV processing without classification."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dms.domain.temporal_sequence import FEATURE_COLUMNS
from dms.domain.video_sample import FrameResult, VideoSample
from dms.features.base import FeatureContext
from dms.features.extractor import FeatureExtractor
from dms.landmarks.processor import LandmarkProcessor
from dms.landmarks.schema import LandmarkSchema
from dms.utils.config import AppConfig
from dms.utils.logging import get_logger

logger = get_logger(__name__)


class FrameProcessor:
    """Process one video frame through the existing CV stack."""

    def __init__(
        self,
        config: AppConfig,
        schema: LandmarkSchema,
        feature_extractor: FeatureExtractor,
        landmark_processor: LandmarkProcessor,
    ) -> None:
        self._config = config
        self._schema = schema
        self._feature_extractor = feature_extractor
        self._landmark_processor = landmark_processor
        self._nan_features = {name: math.nan for name in FEATURE_COLUMNS}

    def process(
        self,
        frame_bgr: NDArray[np.uint8] | None,
        frame_index: int,
        timestamp_sec: float,
        sample: VideoSample,
        detector,
        landmark_extractor,
        read_success: bool = True,
    ) -> FrameResult:
        if not read_success or frame_bgr is None:
            return self._failed_frame(
                frame_index,
                timestamp_sec,
                "Frame read failed or dropped",
            )

        detection = detector.detect(frame_bgr)
        if not detection.detected:
            if self._config.pipeline.skip_on_detection_failure:
                return self._failed_frame(frame_index, timestamp_sec, "Face not detected")
            return self._failed_frame(frame_index, timestamp_sec, "Face not detected")

        timestamp_ms = int(timestamp_sec * 1000)
        if hasattr(landmark_extractor, "extract") and _accepts_timestamp(landmark_extractor.extract):
            landmarks = landmark_extractor.extract(frame_bgr, timestamp_ms=timestamp_ms)
        else:
            landmarks = landmark_extractor.extract(frame_bgr)

        if landmarks is None:
            return self._failed_frame(frame_index, timestamp_sec, "Landmark extraction failed")

        normalized = self._landmark_processor.process(landmarks)
        context = FeatureContext(
            landmarks=landmarks,
            normalized=normalized,
            schema=self._schema,
            image_bgr=frame_bgr,
            detection=detection,
        )

        features = self._feature_extractor.extract(
            context=context,
            sample_id=f"{sample.video_id}:f{frame_index}",
            subject_id=sample.subject_id,
        )

        return FrameResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            processing_success=True,
            features=features.features,
            detection_confidence=detection.confidence,
            landmark_confidence=landmarks.confidence,
        )

    def _failed_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        message: str,
    ) -> FrameResult:
        logger.debug("Frame %d failed: %s", frame_index, message)
        return FrameResult(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            processing_success=False,
            features=dict(self._nan_features),
            detection_confidence=0.0,
            landmark_confidence=0.0,
            error_message=message,
        )


def _accepts_timestamp(func: Any) -> bool:
    import inspect

    try:
        params = inspect.signature(func).parameters
        return "timestamp_ms" in params
    except (TypeError, ValueError):
        return False
