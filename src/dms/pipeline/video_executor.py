"""Video-level multiprocessing execution engine for the DMS video pipeline."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.domain.video_sample import VideoSample
from dms.pipeline.video_dashboard import VideoDashboard
from dms.pipeline.video_worker import process_video_worker
from dms.utils.config import AppConfig
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root

logger = get_logger(__name__)


@dataclass
class VideoExecutionSummary:
    """Summary for one worker execution result."""

    video_id: str
    success: bool
    skipped: bool
    processed_frames: int
    successful_frames: int
    failed_frames: int
    elapsed_sec: float
    error: str | None = None


def resolve_worker_count(config: AppConfig) -> int:
    """Resolve a safe worker count for this machine and configuration."""
    requested = int(getattr(config.compute, "num_workers", 1) or 1)
    return max(1, min(requested, 4))


class VideoExecutor:
    """Execute video work across multiple processes while preserving per-video ordering."""

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._project_root = project_root or get_project_root()
        self._worker_count = resolve_worker_count(config)

    def run(self, videos: list[VideoSample]) -> list[VideoExecutionSummary]:
        if self._worker_count == 1 or not videos:
            return [self._run_sequential(videos)]

        mp.set_start_method("spawn", force=False)

        results: list[VideoExecutionSummary] = []
        pending: dict[Any, tuple[VideoSample, Any]] = {}
        remaining = list(videos)
        dashboard = VideoDashboard()
        dashboard.set_total_videos(len(videos))
        logger.info("Starting video executor with %d workers for %d videos", self._worker_count, len(videos))

        self._progress_queue = mp.Manager().Queue()

        with ProcessPoolExecutor(max_workers=self._worker_count) as executor:
            while remaining or pending:
                while remaining and len(pending) < self._worker_count:
                    sample = remaining.pop(0)
                    payload = {
                        "config": self._config.model_dump(),
                        "sample": {
                            "video_id": sample.video_id,
                            "video_path": str(sample.video_path),
                            "subject_id": sample.subject_id,
                            "fold_name": sample.fold_name,
                            "video_name": sample.video_name,
                            "label": sample.label,
                        },
                        "project_root": str(self._project_root),
                        "progress_queue": self._progress_queue,
                    }
                    future = executor.submit(process_video_worker, payload)
                    pending[future] = (sample, future)

                if not pending:
                    break

                while not self._progress_queue.empty():
                    event = self._progress_queue.get_nowait()
                    dashboard.handle_event(event)
                    dashboard.refresh()

                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED, timeout=0.5)
                for future in done:
                    sample, _ = pending.pop(future)
                    try:
                        outcome = future.result()
                        results.append(_coerce_summary(outcome))
                        dashboard.mark_video_completed(1)
                        dashboard.refresh()
                        logger.info(
                            "Completed video %s with success=%s processed_frames=%d",
                            sample.video_id,
                            outcome.get("success"),
                            outcome.get("processed_frames", 0),
                        )
                    except Exception as exc:  # pragma: no cover - runtime path
                        logger.exception("Worker failed for %s: %s", sample.video_id, exc)
                        dashboard.mark_video_completed(1)
                        results.append(
                            VideoExecutionSummary(
                                video_id=sample.video_id,
                                success=False,
                                skipped=False,
                                processed_frames=0,
                                successful_frames=0,
                                failed_frames=0,
                                elapsed_sec=0.0,
                                error=str(exc),
                            )
                        )

        while not self._progress_queue.empty():
            event = self._progress_queue.get_nowait()
            dashboard.handle_event(event)
            dashboard.refresh()

        dashboard.finish()
        return results

    def _run_sequential(self, videos: list[VideoSample]) -> VideoExecutionSummary:
        raise RuntimeError("sequential execution should be handled by video_pipeline")


def _coerce_summary(outcome: dict[str, Any]) -> VideoExecutionSummary:
    return VideoExecutionSummary(
        video_id=str(outcome.get("video_id", "")),
        success=bool(outcome.get("success", False)),
        skipped=bool(outcome.get("skipped", False)),
        processed_frames=int(outcome.get("processed_frames", 0)),
        successful_frames=int(outcome.get("successful_frames", 0)),
        failed_frames=int(outcome.get("failed_frames", 0)),
        elapsed_sec=float(outcome.get("elapsed_sec", 0.0)),
        error=outcome.get("error"),
    )
