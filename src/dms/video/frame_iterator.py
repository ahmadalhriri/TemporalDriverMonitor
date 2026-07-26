"""Frame-by-frame iteration with timestamps and optional buffering."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from dms.utils.logging import get_logger
from dms.video.video_reader import VideoMetadata, VideoReader

logger = get_logger(__name__)


@dataclass(frozen=True)
class FramePacket:
    """One frame from a video stream."""

    frame_index: int
    timestamp_sec: float
    frame_bgr: NDArray[np.uint8] | None
    read_success: bool


class FrameIterator:
    """Yield ordered FramePacket objects from a streaming VideoReader."""

    def __init__(
        self,
        reader: VideoReader,
        metadata: VideoMetadata,
        buffer_size: int = 0,
    ) -> None:
        self._reader = reader
        self._metadata = metadata
        self._buffer_size = max(buffer_size, 0)
        self.dropped_frames = 0

    def __iter__(self) -> Iterator[FramePacket]:
        if self._buffer_size <= 1:
            yield from self._iter_sequential()
            return
        yield from self._iter_buffered()

    def _iter_sequential(self) -> Iterator[FramePacket]:
        fps = self._metadata.fps
        frame_index = 0

        while True:
            ok, frame = self._reader.read_frame()
            if not ok:
                break
            if frame is None:
                self.dropped_frames += 1
                logger.warning("Dropped frame at index %d", frame_index)
                yield FramePacket(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    frame_bgr=None,
                    read_success=False,
                )
                frame_index += 1
                continue

            yield FramePacket(
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                frame_bgr=frame,
                read_success=True,
            )
            frame_index += 1

    def _iter_buffered(self) -> Iterator[FramePacket]:
        fps = self._metadata.fps
        frame_queue: queue.Queue[tuple[int, bool, NDArray[np.uint8] | None] | None] = queue.Queue(
            maxsize=self._buffer_size
        )
        stop_token = object()

        def _reader_thread() -> None:
            index = 0
            while True:
                ok, frame = self._reader.read_frame()
                if not ok:
                    break
                try:
                    frame_queue.put((index, True, frame), timeout=30.0)
                except queue.Full:
                    logger.warning("Frame buffer full; dropping frame %d", index)
                index += 1
            frame_queue.put(stop_token)  # type: ignore[arg-type]

        thread = threading.Thread(target=_reader_thread, daemon=True)
        thread.start()

        while True:
            item = frame_queue.get()
            if item is stop_token:
                break
            frame_index, read_ok, frame = item
            if not read_ok or frame is None:
                self.dropped_frames += 1
                yield FramePacket(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    frame_bgr=None,
                    read_success=False,
                )
                continue

            yield FramePacket(
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                frame_bgr=frame,
                read_success=True,
            )

        thread.join(timeout=5.0)
