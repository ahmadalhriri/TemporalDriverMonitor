"""OpenCV-based streaming video reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from dms.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Video stream properties."""

    fps: float
    total_frames: int
    frame_width: int
    frame_height: int


@dataclass
class VideoOpenResult:
    """Result of attempting to open a video."""

    success: bool
    capture: cv2.VideoCapture | None = None
    metadata: VideoMetadata | None = None
    error: str | None = None


class VideoReader:
    """Stream frames from a video file without loading it entirely into memory."""

    def __init__(self, video_path: Path) -> None:
        self._video_path = video_path
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    @property
    def metadata(self) -> VideoMetadata | None:
        return self._metadata

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> VideoOpenResult:
        if not self._video_path.exists():
            return VideoOpenResult(success=False, error=f"Video not found: {self._video_path}")

        capture = cv2.VideoCapture(str(self._video_path), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            return VideoOpenResult(
                success=False,
                error=f"Failed to open video (codec/unreadable): {self._video_path}",
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0
            logger.warning("Invalid FPS for %s; defaulting to %.1f", self._video_path.name, fps)

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self._capture = capture
        self._metadata = VideoMetadata(
            fps=fps,
            total_frames=total_frames,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return VideoOpenResult(success=True, capture=capture, metadata=self._metadata)

    def read_frame(self) -> tuple[bool, object | None]:
        if self._capture is None:
            return False, None
        return self._capture.read()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoReader:
        result = self.open()
        if not result.success:
            raise OSError(result.error or "Failed to open video")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
