"""Processing progress and throughput statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from dms.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProgressTracker:
    """Track frame/video processing throughput and ETA."""

    device_label: str = "cpu"
    total_videos: int = 0
    videos_completed: int = 0
    total_frames_target: int = 0
    frames_processed: int = 0
    inference_time_sec: float = 0.0
    _start_time: float = field(default_factory=time.perf_counter)
    _last_log_time: float = field(default_factory=time.perf_counter)
    log_interval_sec: float = 5.0

    def record_frame(self, inference_sec: float) -> None:
        self.frames_processed += 1
        self.inference_time_sec += inference_sec
        now = time.perf_counter()
        if now - self._last_log_time >= self.log_interval_sec:
            self._log_stats(scope="frame")
            self._last_log_time = now

    def complete_video(self) -> None:
        self.videos_completed += 1
        self._log_stats(scope="video")

    @property
    def processing_fps(self) -> float:
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        return self.frames_processed / elapsed

    @property
    def avg_inference_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return (self.inference_time_sec / self.frames_processed) * 1000.0

    @property
    def eta_sec(self) -> float | None:
        if self.total_frames_target <= 0 or self.frames_processed == 0:
            return None
        remaining = max(self.total_frames_target - self.frames_processed, 0)
        return remaining / max(self.processing_fps, 1e-6)

    def _log_stats(self, scope: str) -> None:
        eta = self.eta_sec
        eta_str = f"{eta:.0f}s" if eta is not None else "n/a"
        logger.info(
            "[%s] device=%s | fps=%.1f | avg_infer=%.1fms | frames=%d | videos=%d/%d | eta=%s",
            scope,
            self.device_label,
            self.processing_fps,
            self.avg_inference_ms,
            self.frames_processed,
            self.videos_completed,
            self.total_videos,
            eta_str,
        )

    def summary(self) -> dict[str, float | int | str]:
        elapsed = time.perf_counter() - self._start_time
        return {
            "device": self.device_label,
            "processing_fps": round(self.processing_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "frames_processed": self.frames_processed,
            "videos_completed": self.videos_completed,
            "elapsed_sec": round(elapsed, 2),
        }
