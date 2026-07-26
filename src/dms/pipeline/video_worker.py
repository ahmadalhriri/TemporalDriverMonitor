"""Worker entry point for processing a single video in a separate process."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

from dms.detectors.factory import create_face_detector
from dms.features.extractor import FeatureExtractor
from dms.landmarks.processor import LandmarkProcessor
from dms.landmarks.schema import LandmarkSchema
from dms.pipeline.frame_processor import FrameProcessor
from dms.sequence.sequence_builder import SequenceBuilder
from dms.sequence.sequence_exporter import SequenceExporter
from dms.utils.config import AppConfig, load_landmark_schema
from dms.utils.device import resolve_device
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root
from dms.video.frame_iterator import FrameIterator
from dms.video.landmark_extractor_factory import create_video_landmark_extractor
from dms.video.video_reader import VideoReader


def _emit_progress(queue: Any | None, payload: dict[str, Any]) -> None:
    if queue is None:
        return
    try:
        queue.put(payload)
    except Exception:
        return

logger = get_logger(__name__)


def process_video_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Process one video inside a worker process and export its temporal sequence."""
    config_payload = payload.get("config") or {}
    sample_payload = payload.get("sample") or {}
    project_root = Path(payload.get("project_root") or get_project_root())
    progress_queue = payload.get("progress_queue")

    config = AppConfig.model_validate(config_payload)
    sample = _deserialize_sample(sample_payload)

    if config.video.skip_existing:
        exporter = SequenceExporter(
            config.temporal_features_dir_path(project_root),
            config.video.export_formats,
        )
        if exporter.is_complete(sample):
            return {
                "video_id": sample.video_id,
                "success": True,
                "skipped": True,
                "processed_frames": 0,
                "successful_frames": 0,
                "failed_frames": 0,
                "elapsed_sec": 0.0,
                "error": None,
            }

    schema = LandmarkSchema.from_dict(load_landmark_schema(config.landmarks.schema_path))
    device = resolve_device(config.compute)
    exporter = SequenceExporter(
        config.temporal_features_dir_path(project_root),
        config.video.export_formats,
    )

    detector = create_face_detector(config.detector)
    landmark_extractor = create_video_landmark_extractor(config, device)
    feature_extractor = FeatureExtractor(config.features)
    landmark_processor = LandmarkProcessor(schema)
    frame_processor = FrameProcessor(config, schema, feature_extractor, landmark_processor)

    started_at = time.perf_counter()
    last_progress_emit = 0.0
    frames_processed = 0
    total_frames = 0
    cumulative_inference_ms = 0.0
    try:
        reader = VideoReader(sample.video_path)
        open_result = reader.open()
        if not open_result.success or open_result.metadata is None:
            raise OSError(open_result.error or "Could not open video")

        metadata = open_result.metadata
        total_frames = metadata.total_frames
        builder = SequenceBuilder(sample, metadata, device.device)
        truncated = False

        frame_iter = FrameIterator(reader, metadata, buffer_size=config.compute.frame_buffer_size)
        try:
            for packet in frame_iter:
                frame_started_at = time.perf_counter()
                result = frame_processor.process(
                    frame_bgr=packet.frame_bgr,
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    sample=sample,
                    detector=detector,
                    landmark_extractor=landmark_extractor,
                    read_success=packet.read_success,
                )
                inference_ms = (time.perf_counter() - frame_started_at) * 1000.0
                cumulative_inference_ms += inference_ms
                builder.append(result)
                frames_processed += 1
                now = time.perf_counter()
                if now - last_progress_emit >= 0.2:
                    elapsed_sec = max(now - started_at, 1e-6)
                    fps = frames_processed / elapsed_sec
                    progress_percent = 100.0 if total_frames <= 0 else (frames_processed / total_frames) * 100.0
                    eta_sec = None if total_frames <= 0 or frames_processed <= 0 else (total_frames - frames_processed) / max(fps, 1e-6)
                    _emit_progress(
                        progress_queue,
                        {
                            "pid": os.getpid(),
                            "video_id": sample.video_id,
                            "subject_id": sample.subject_id,
                            "video_name": sample.video_name,
                            "frames_processed": frames_processed,
                            "total_frames": total_frames,
                            "progress_percent": progress_percent,
                            "fps": fps,
                            "avg_inference_ms": cumulative_inference_ms / max(frames_processed, 1),
                            "elapsed_sec": elapsed_sec,
                            "eta_sec": eta_sec,
                            "status": "running",
                        },
                    )
                    last_progress_emit = now

            truncated = frame_iter.dropped_frames > 0
            sequence = builder.build(
                dropped_frames=frame_iter.dropped_frames,
                truncated=truncated,
            )
            exporter.export(sample, sequence)
            elapsed_sec = max(time.perf_counter() - started_at, 1e-6)
            _emit_progress(
                progress_queue,
                {
                    "pid": os.getpid(),
                    "video_id": sample.video_id,
                    "subject_id": sample.subject_id,
                    "video_name": sample.video_name,
                    "frames_processed": len(sequence.rows),
                    "total_frames": total_frames,
                    "progress_percent": 100.0,
                    "fps": len(sequence.rows) / elapsed_sec,
                    "avg_inference_ms": cumulative_inference_ms / max(len(sequence.rows), 1),
                    "elapsed_sec": elapsed_sec,
                    "eta_sec": 0.0,
                    "status": "completed",
                },
            )
            return {
                "video_id": sample.video_id,
                "success": True,
                "skipped": False,
                "processed_frames": len(sequence.rows),
                "successful_frames": builder._successful,
                "failed_frames": builder._failed,
                "elapsed_sec": time.perf_counter() - started_at,
                "error": None,
            }
        finally:
            reader.close()
    except Exception as exc:  # pragma: no cover - exercised via runtime failures
        logger.exception("Worker failed for %s", sample.video_path)
        exporter.export_failure_manifest(sample, str(exc))
        _emit_progress(
            progress_queue,
            {
                "pid": os.getpid(),
                "video_id": sample.video_id,
                "subject_id": sample.subject_id,
                "video_name": sample.video_name,
                "frames_processed": 0,
                "total_frames": total_frames,
                "progress_percent": 0.0,
                "fps": 0.0,
                "avg_inference_ms": 0.0,
                "elapsed_sec": time.perf_counter() - started_at,
                "eta_sec": None,
                "status": "failed",
            },
        )
        return {
            "video_id": sample.video_id,
            "success": False,
            "skipped": False,
            "processed_frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "elapsed_sec": time.perf_counter() - started_at,
            "error": str(exc),
        }
    finally:
        if hasattr(detector, "close"):
            detector.close()
        if hasattr(landmark_extractor, "close"):
            landmark_extractor.close()
        gc.collect()


def _deserialize_sample(payload: dict[str, Any]):
    from dms.domain.video_sample import VideoSample

    return VideoSample(
        video_id=payload.get("video_id", ""),
        video_path=Path(payload.get("video_path", "")),
        subject_id=payload.get("subject_id", ""),
        fold_name=payload.get("fold_name", ""),
        video_name=payload.get("video_name", ""),
        label=payload.get("label"),
    )
