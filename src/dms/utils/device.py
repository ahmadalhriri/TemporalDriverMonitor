"""Compute device detection and resolution."""

from __future__ import annotations

from dataclasses import dataclass

from dms.utils.config import ComputeConfig
from dms.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedDevice:
    """Resolved runtime compute settings."""

    device: str  # cpu | cuda
    mediapipe_delegate: str  # cpu | gpu
    cuda_available: bool
    description: str


def _probe_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        pass

    try:
        import cv2

        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (ImportError, AttributeError):
        return False


def _probe_mediapipe_gpu() -> bool:
    try:
        import mediapipe as mp

        if hasattr(mp, "tasks") and hasattr(mp.tasks, "python"):
            base_options = mp.tasks.python.BaseOptions
            if hasattr(base_options, "Delegate"):
                return hasattr(base_options.Delegate, "GPU")
    except ImportError:
        pass
    return False


def resolve_device(config: ComputeConfig) -> ResolvedDevice:
    """Resolve device settings from config and runtime probes."""
    cuda_available = _probe_cuda()
    mp_gpu_available = _probe_mediapipe_gpu()

    device = config.device.lower()
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    elif device == "cuda" and not cuda_available:
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    delegate = config.mediapipe_delegate.lower()
    if delegate == "auto":
        delegate = "gpu" if mp_gpu_available and device == "cuda" else "cpu"
    elif delegate == "gpu" and not mp_gpu_available:
        logger.warning("MediaPipe GPU delegate unavailable; using CPU delegate")
        delegate = "cpu"

    description = f"device={device}, mediapipe_delegate={delegate}"
    logger.info("Compute backend: %s (cuda_available=%s)", description, cuda_available)

    return ResolvedDevice(
        device=device,
        mediapipe_delegate=delegate,
        cuda_available=cuda_available,
        description=description,
    )
