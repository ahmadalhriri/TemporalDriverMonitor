"""Video I/O module."""

from __future__ import annotations

from dms.video.frame_iterator import FrameIterator, FramePacket
from dms.video.video_reader import VideoMetadata, VideoOpenResult, VideoReader

__all__ = [
    "FrameIterator",
    "FramePacket",
    "VideoMetadata",
    "VideoOpenResult",
    "VideoReader",
]
