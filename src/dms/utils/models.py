"""Model asset management and automatic downloader utility."""

from __future__ import annotations

import os
from pathlib import Path
import urllib.request

from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root

logger = get_logger(__name__)

MODEL_URLS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def ensure_model_asset(filename: str, models_dir: Path | None = None) -> Path:
    """Ensure specified model asset exists locally, downloading if necessary.

    Args:
        filename: Target model filename (e.g. 'face_landmarker.task').
        models_dir: Directory where models are stored. Defaults to <project_root>/models.

    Returns:
        Absolute Path to the validated model asset.
    """
    if models_dir is None:
        models_dir = get_project_root() / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / filename

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    if filename not in MODEL_URLS:
        raise ValueError(f"Unknown model asset '{filename}'. Available: {list(MODEL_URLS.keys())}")

    url = MODEL_URLS[filename]
    logger.info("Downloading model asset '%s' from %s...", filename, url)
    
    temp_path = target_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(target_path)
        logger.info("Successfully downloaded '%s' (%d bytes)", filename, target_path.stat().st_size)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model asset '{filename}': {exc}") from exc

    return target_path
