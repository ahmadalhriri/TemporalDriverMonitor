"""Discover videos from fold-based dataset layouts."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from dms.domain.video_sample import VideoSample
from dms.utils.config import AppConfig
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root, resolve_path

logger = get_logger(__name__)


class VideoDatasetLoader:
    """Discover subject videos under configurable fold roots."""

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._root = project_root or get_project_root()
        self._subject_pattern = re.compile(config.video.subject_dir_pattern)
        self._extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                            for ext in config.video.extensions}

    def discover(self) -> list[VideoSample]:
        roots = self._config.paths.video_data_roots
        if not roots:
            logger.warning("No video_data_roots configured")
            return []

        samples: list[VideoSample] = []
        max_videos = self._config.video.max_videos

        for root_spec in roots:
            root_path = resolve_path(root_spec, self._root)
            if not root_path.exists():
                logger.warning("Video data root not found: %s", root_path)
                continue

            fold_name = root_path.name
            dataset_root = self._resolve_dataset_root(root_path)
            labels = self._load_labels(dataset_root)

            for subject_dir in sorted(dataset_root.iterdir()):
                if not subject_dir.is_dir():
                    continue
                if not self._subject_pattern.match(subject_dir.name):
                    continue

                subject_id = subject_dir.name
                for video_path in sorted(subject_dir.iterdir()):
                    if not video_path.is_file():
                        continue
                    if video_path.suffix.lower() not in self._extensions:
                        continue

                    rel_key = f"{subject_id}/{video_path.name}"
                    label = labels.get(rel_key) or labels.get(video_path.name)

                    video_id = f"{fold_name}:{subject_id}:{video_path.stem}"
                    samples.append(
                        VideoSample(
                            video_id=video_id,
                            video_path=video_path.resolve(),
                            subject_id=subject_id,
                            fold_name=fold_name,
                            video_name=video_path.name,
                            label=label,
                        )
                    )

                    if max_videos is not None and len(samples) >= max_videos:
                        logger.info("Reached max_videos limit (%d)", max_videos)
                        return samples

        logger.info("Discovered %d videos across %d root(s)", len(samples), len(roots))
        return samples

    def _resolve_dataset_root(self, root_path: Path) -> Path:
        """Handle nested layouts such as Fold1_part1/Fold1_part1/01/."""
        subdirs = [p for p in root_path.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            return root_path

        only_subdir = subdirs[0]
        nested_subjects = [
            p for p in only_subdir.iterdir()
            if p.is_dir() and self._subject_pattern.match(p.name)
        ]
        if nested_subjects:
            logger.debug("Using nested dataset root: %s", only_subdir)
            return only_subdir

        return root_path

    def _load_labels(self, dataset_root: Path) -> dict[str, str]:
        labels_path = dataset_root / self._config.video.labels_filename
        if not labels_path.exists():
            parent_labels = dataset_root.parent / self._config.video.labels_filename
            if parent_labels.exists():
                labels_path = parent_labels
            else:
                return {}

        labels: dict[str, str] = {}
        try:
            with labels_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    return labels
                path_field = "video_path" if "video_path" in reader.fieldnames else reader.fieldnames[0]
                label_field = "label" if "label" in reader.fieldnames else reader.fieldnames[-1]
                for row in reader:
                    key = row.get(path_field, "").strip().replace("\\", "/")
                    value = row.get(label_field, "").strip()
                    if key and value:
                        labels[key] = value
        except OSError as exc:
            logger.warning("Failed to read labels file %s: %s", labels_path, exc)

        logger.info("Loaded %d optional labels from %s", len(labels), labels_path)
        return labels
