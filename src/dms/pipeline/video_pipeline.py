"""End-to-end video-to-temporal-features pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dms.dataset.video_dataset import VideoDatasetLoader
from dms.detectors.factory import create_face_detector
from dms.features.extractor import FeatureExtractor
from dms.landmarks.processor import LandmarkProcessor
from dms.landmarks.schema import LandmarkSchema
from dms.pipeline.frame_processor import FrameProcessor
from dms.pipeline.video_executor import VideoExecutor
from dms.sequence.sequence_builder import SequenceBuilder
from dms.sequence.sequence_exporter import SequenceExporter
from dms.utils.config import AppConfig, load_config, load_landmark_schema
from dms.utils.device import resolve_device
from dms.utils.io import ensure_dir, save_json
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root
from dms.utils.progress import ProgressTracker
from dms.video.frame_iterator import FrameIterator
from dms.video.landmark_extractor_factory import create_video_landmark_extractor
from dms.video.video_reader import VideoReader

logger = get_logger(__name__)


@dataclass
class VideoPipelineResult:
    """Summary of a video pipeline run."""

    run_dir: Path
    videos_discovered: int
    videos_processed: int
    videos_skipped: int
    videos_failed: int
    total_frames: int


class VideoPipelineRunner:
    """Convert videos into temporal feature sequences."""

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._root = project_root or get_project_root()
        self._schema = LandmarkSchema.from_dict(
            load_landmark_schema(config.landmarks.schema_path)
        )
        self._device = resolve_device(config.compute)

    def run(self) -> VideoPipelineResult:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(self._config.outputs_dir_path(self._root) / "video_runs" / run_id)
        save_json(self._config.model_dump(), run_dir / "config_snapshot.json")

        loader = VideoDatasetLoader(self._config, self._root)
        videos = loader.discover()

        output_root = ensure_dir(self._config.temporal_features_dir_path(self._root))
        exporter = SequenceExporter(output_root, self._config.video.export_formats)

        progress = ProgressTracker(
            device_label=self._device.description,
            total_videos=len(videos),
            log_interval_sec=self._config.compute.log_interval_sec,
        )

        processed = 0
        skipped = 0
        failed = 0
        total_frames = 0

        worker_count = self._config.compute.num_workers
        if worker_count <= 1:
            with create_face_detector(self._config.detector) as detector, create_video_landmark_extractor(
                self._config, self._device
            ) as landmark_extractor:
                landmark_processor = LandmarkProcessor(self._schema)
                feature_extractor = FeatureExtractor(self._config.features)
                frame_processor = FrameProcessor(
                    self._config,
                    self._schema,
                    feature_extractor,
                    landmark_processor,
                )

                for index, sample in enumerate(videos, start=1):
                    if self._config.video.skip_existing and exporter.is_complete(sample):
                        logger.info(
                            "[%d/%d] Skipping existing export: %s",
                            index,
                            len(videos),
                            sample.video_name,
                        )
                        skipped += 1
                        continue

                    logger.info(
                        "[%d/%d] Processing %s (subject %s)",
                        index,
                        len(videos),
                        sample.video_path,
                        sample.subject_id,
                    )

                    try:
                        frame_count = self._process_video(
                            sample=sample,
                            detector=detector,
                            landmark_extractor=landmark_extractor,
                            frame_processor=frame_processor,
                            exporter=exporter,
                            progress=progress,
                        )
                        processed += 1
                        total_frames += frame_count
                    except OSError as exc:
                        logger.error("Failed to process video %s: %s", sample.video_path, exc)
                        exporter.export_failure_manifest(sample, str(exc))
                        failed += 1

                    progress.complete_video()
        else:
            executor = VideoExecutor(self._config, self._root)
            summaries = executor.run(videos)
            for summary in summaries:
                if summary.skipped:
                    skipped += 1
                    continue
                if summary.success:
                    processed += 1
                    total_frames += summary.processed_frames
                else:
                    failed += 1
                progress.complete_video()

        summary = progress.summary()
        summary.update(
            {
                "videos_discovered": len(videos),
                "videos_processed": processed,
                "videos_skipped": skipped,
                "videos_failed": failed,
            }
        )
        save_json(summary, run_dir / "run_summary.json")

        logger.info(
            "Video pipeline complete: processed=%d skipped=%d failed=%d frames=%d",
            processed,
            skipped,
            failed,
            total_frames,
        )

        return VideoPipelineResult(
            run_dir=run_dir,
            videos_discovered=len(videos),
            videos_processed=processed,
            videos_skipped=skipped,
            videos_failed=failed,
            total_frames=total_frames,
        )

    def _process_video(
        self,
        sample,
        detector,
        landmark_extractor,
        frame_processor: FrameProcessor,
        exporter: SequenceExporter,
        progress: ProgressTracker,
    ) -> int:
        reader = VideoReader(sample.video_path)
        open_result = reader.open()
        if not open_result.success or open_result.metadata is None:
            raise OSError(open_result.error or "Could not open video")

        metadata = open_result.metadata
        if progress.total_frames_target == 0:
            progress.total_frames_target = metadata.total_frames * progress.total_videos

        builder = SequenceBuilder(sample, metadata, self._device.device)
        truncated = False

        try:
            frame_iter = FrameIterator(
                reader,
                metadata,
                buffer_size=self._config.compute.frame_buffer_size,
            )

            for packet in frame_iter:
                start = time.perf_counter()
                result = frame_processor.process(
                    frame_bgr=packet.frame_bgr,
                    frame_index=packet.frame_index,
                    timestamp_sec=packet.timestamp_sec,
                    sample=sample,
                    detector=detector,
                    landmark_extractor=landmark_extractor,
                    read_success=packet.read_success,
                )
                builder.append(result)
                progress.record_frame(time.perf_counter() - start)

            truncated = frame_iter.dropped_frames > 0
            sequence = builder.build(
                dropped_frames=frame_iter.dropped_frames,
                truncated=truncated,
            )
            exporter.export(sample, sequence)
            return len(sequence.rows)
        finally:
            reader.close()


def run_video_from_config(config_path: str | Path | None = None) -> VideoPipelineResult:
    config = load_config(config_path)
    runner = VideoPipelineRunner(config)
    return runner.run()
