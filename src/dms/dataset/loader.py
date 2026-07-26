"""Load FL3D images and ground-truth labels (ignores annotation landmarks)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from dms.domain.enums import DriverState, Split
from dms.domain.sample import DriverSample
from dms.utils.config import AppConfig
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root, resolve_path

logger = get_logger(__name__)

_SUBJECT_PATTERN = re.compile(r"(P\d+_\d+)")


class FL3DDatasetLoader:
    """Load FL3D dataset samples from split annotation files."""

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._root = project_root or get_project_root()
        self._data_root = config.data_root_path(self._root)

    def load_split(self, split: Split | str) -> list[DriverSample]:
        split_enum = split if isinstance(split, Split) else Split(str(split))
        annotation_path = self._annotation_path(split_enum)
        logger.info("Loading split '%s' from %s", split_enum.value, annotation_path)

        with annotation_path.open("r", encoding="utf-8") as handle:
            annotations: dict[str, dict] = json.load(handle)

        samples: list[DriverSample] = []
        max_samples = self._config.dataset.max_samples

        for raw_path, payload in annotations.items():
            if max_samples is not None and len(samples) >= max_samples:
                break

            sample = self._parse_entry(raw_path, payload, split_enum)
            if sample is not None:
                samples.append(sample)

        logger.info("Loaded %d samples for split '%s'", len(samples), split_enum.value)
        return samples

    def load_splits(self, splits: list[str] | None = None) -> dict[Split, list[DriverSample]]:
        split_names = splits or self._config.dataset.splits
        return {Split(name): self.load_split(name) for name in split_names}

    def _annotation_path(self, split: Split) -> Path:
        filename = self._config.dataset.annotation_pattern.format(split=split.value)
        return self._data_root / filename if (self._data_root / filename).exists() else resolve_path(
            self._data_root / filename, self._root
        )

    def _parse_entry(
        self,
        raw_path: str,
        payload: dict,
        split: Split,
    ) -> DriverSample | None:
        label_field = self._config.dataset.label_field
        if label_field not in payload:
            logger.warning("Missing label field '%s' for %s", label_field, raw_path)
            return None

        label_str = str(payload[label_field])
        if label_str not in self._config.dataset.valid_labels:
            logger.warning("Skipping sample with invalid label '%s': %s", label_str, raw_path)
            return None

        image_path = self._resolve_image_path(raw_path)
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            return None

        subject_id = self._extract_subject_id(raw_path, image_path)
        sample_id = f"{split.value}:{subject_id}:{image_path.name}"

        return DriverSample(
            sample_id=sample_id,
            image_path=image_path,
            subject_id=subject_id,
            split=split,
            label=DriverState.from_label(label_str),
        )

    def _resolve_image_path(self, raw_path: str) -> Path:
        normalized = raw_path.replace("\\", "/").lstrip("./")
        if normalized.startswith("classification_frames/"):
            normalized = normalized[len("classification_frames/") :]

        candidate = self._data_root / normalized
        if candidate.exists():
            return candidate.resolve()

        # Fallback: search by filename under data root
        filename = Path(normalized).name
        matches = list(self._data_root.rglob(filename))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            subject_hint = Path(normalized).parent.name
            for match in matches:
                if subject_hint in str(match):
                    return match.resolve()

        return candidate.resolve()

    @staticmethod
    def _extract_subject_id(raw_path: str, image_path: Path) -> str:
        for text in (raw_path, str(image_path)):
            match = _SUBJECT_PATTERN.search(text)
            if match:
                return match.group(1)
        return image_path.parent.name
