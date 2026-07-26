"""Real-time console dashboard for video worker progress."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from dms.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class VideoDashboard:
    """Render a live progress dashboard for the video pipeline."""

    def __init__(self, enabled: bool | None = None, refresh_interval_sec: float = 1.0) -> None:
        self._enabled = enabled if enabled is not None else self._should_enable()
        self._refresh_interval_sec = refresh_interval_sec
        self._last_render_time = 0.0
        self._worker_state: dict[int, dict[str, Any]] = {}
        self._worker_order: list[int] = []
        self._total_videos = 0
        self._completed_videos = 0
        self._started_at = time.perf_counter()
        self._overall_processed_frames = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_total_videos(self, total_videos: int) -> None:
        self._total_videos = max(0, total_videos)

    def mark_video_completed(self, count: int = 1) -> None:
        self._completed_videos = max(0, self._completed_videos + count)

    def handle_event(self, event: dict[str, Any]) -> None:
        if not self._enabled:
            return

        pid = int(event.get("pid", 0))
        if pid <= 0:
            return

        if pid not in self._worker_state:
            self._worker_state[pid] = {
                "worker_id": len(self._worker_order) + 1,
                "pid": pid,
                "status": "idle",
                "subject_id": "-",
                "video_name": "-",
                "frames_processed": 0,
                "total_frames": 0,
                "progress_percent": 0.0,
                "fps": 0.0,
                "avg_inference_ms": 0.0,
                "elapsed_sec": 0.0,
                "eta_sec": None,
                "video_id": "",
            }
            self._worker_order.append(pid)

        state = self._worker_state[pid]
        previous_frames = int(state.get("frames_processed", 0))
        new_frames = int(event.get("frames_processed", previous_frames))
        if new_frames >= previous_frames and event.get("status", "running") == "running":
            self._overall_processed_frames += max(0, new_frames - previous_frames)

        state.update(
            {
                "status": event.get("status", "running"),
                "subject_id": event.get("subject_id", state.get("subject_id", "-")),
                "video_name": event.get("video_name", state.get("video_name", "-")),
                "frames_processed": new_frames,
                "total_frames": int(event.get("total_frames", state.get("total_frames", 0))),
                "progress_percent": float(event.get("progress_percent", state.get("progress_percent", 0.0))),
                "fps": float(event.get("fps", state.get("fps", 0.0))),
                "avg_inference_ms": float(event.get("avg_inference_ms", state.get("avg_inference_ms", 0.0))),
                "elapsed_sec": float(event.get("elapsed_sec", state.get("elapsed_sec", 0.0))),
                "eta_sec": event.get("eta_sec"),
                "video_id": event.get("video_id", state.get("video_id", "")),
            }
        )

        if event.get("status") in {"completed", "failed"}:
            state["status"] = "idle"
            state["video_name"] = "-"
            state["subject_id"] = "-"
            state["video_id"] = ""
            state["frames_processed"] = 0
            state["total_frames"] = 0
            state["progress_percent"] = 0.0
            state["fps"] = 0.0
            state["avg_inference_ms"] = 0.0
            state["elapsed_sec"] = 0.0
            state["eta_sec"] = None

    def refresh(self) -> None:
        if not self._enabled:
            return

        now = time.perf_counter()
        if now - self._last_render_time < self._refresh_interval_sec:
            return
        self._last_render_time = now
        self._render()

    def finish(self) -> None:
        if not self._enabled:
            return
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        if not self._enabled:
            return

        self._clear_console()
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("DMS Video Processing Dashboard")
        lines.append("=" * 72)
        lines.append("")

        for pid in self._worker_order:
            state = self._worker_state[pid]
            lines.append(f"Worker #{state['worker_id']} (PID {pid})")
            lines.append(f"  Subject        : {state['subject_id']}")
            lines.append(f"  Video         : {state['video_name']}")
            lines.append(
                f"  Frames        : {state['frames_processed']:,} / {state['total_frames']:,}"
            )
            progress_percent = state["progress_percent"]
            bar = self._build_bar(progress_percent)
            lines.append(f"  Progress      : {bar} {progress_percent:5.1f}%")
            lines.append(f"  FPS           : {state['fps']:.1f}")
            lines.append(f"  Inference     : {state['avg_inference_ms']:.1f} ms/frame")
            elapsed = self._format_duration(state["elapsed_sec"])
            eta = self._format_duration(state["eta_sec"]) if state["eta_sec"] is not None else "n/a"
            lines.append(f"  Elapsed       : {elapsed}")
            lines.append(f"  ETA           : {eta}")
            lines.append("-" * 72)

        if not self._worker_order:
            lines.append("Waiting for workers to start...")
            lines.append("-" * 72)

        lines.append("")
        lines.append("Overall Pipeline")
        lines.append(f"  Videos completed : {self._completed_videos} / {self._total_videos}")
        remaining = max(self._total_videos - self._completed_videos, 0)
        lines.append(f"  Videos remaining : {remaining}")
        lines.append(f"  Total processed frames : {self._overall_processed_frames:,}")
        average_fps = self._compute_average_fps()
        lines.append(f"  Average FPS : {average_fps:.1f}")
        lines.append(f"  Active workers : {self._active_worker_count()} / {len(self._worker_order)}")
        cpu_usage = self._cpu_usage_percent()
        ram_usage = self._ram_usage()
        lines.append(f"  CPU Usage : {cpu_usage:.0f}%")
        lines.append(f"  RAM Usage : {ram_usage}")
        lines.append("=" * 72)

        sys.stdout.write("\n".join(lines) + ("\n" if final else "\r"))
        sys.stdout.flush()

    def _build_bar(self, percent: float) -> str:
        width = 28
        filled = int(round(max(0.0, min(percent, 100.0)) / 100.0 * width))
        return "█" * filled + "░" * (width - filled)

    def _active_worker_count(self) -> int:
        return sum(1 for state in self._worker_state.values() if state.get("status") == "running")

    def _compute_average_fps(self) -> float:
        fps_values = [state.get("fps", 0.0) for state in self._worker_state.values() if state.get("status") == "running"]
        if not fps_values:
            return 0.0
        return sum(fps_values) / len(fps_values)

    def _cpu_usage_percent(self) -> float:
        if psutil is None:
            return 0.0
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    def _ram_usage(self) -> str:
        if psutil is None:
            return "n/a"
        try:
            usage = psutil.virtual_memory()
            used_gb = usage.used / (1024**3)
            total_gb = usage.total / (1024**3)
            return f"{used_gb:.1f} / {total_gb:.1f} GB"
        except Exception:
            return "n/a"

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        total_seconds = max(int(seconds), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _should_enable() -> bool:
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    def _clear_console(self) -> None:
        if not self._enabled:
            return
        if os.name == "nt":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
