"""Export temporal feature sequences to disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dms.domain.temporal_sequence import SCHEMA_VERSION, TemporalFeatureSequence
from dms.domain.video_sample import VideoSample
from dms.utils.io import ensure_dir, save_dataframe
from dms.utils.logging import get_logger

logger = get_logger(__name__)


class SequenceExporter:
    """Write one temporal sequence per video."""

    def __init__(self, output_root: Path, export_formats: list[str] | None = None) -> None:
        self._output_root = output_root
        self._formats = [fmt.lower().lstrip(".") for fmt in (export_formats or ["parquet"])]

    def output_paths(self, sample: VideoSample) -> dict[str, Path]:
        video_dir = (
            self._output_root
            / sample.fold_name
            / sample.subject_dir_name
        )
        stem = sample.video_stem
        return {
            fmt: video_dir / f"{stem}.{fmt}"
            for fmt in self._formats
        }

    def is_complete(self, sample: VideoSample) -> bool:
        paths = self.output_paths(sample)
        parquet_path = paths.get("parquet")
        meta_path = self._meta_path(sample)
        if parquet_path is None or not parquet_path.exists() or not meta_path.exists():
            return False
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            processed = meta.get("processed_frames", 0)
            total = meta.get("total_frames", 0)
            if processed <= 0:
                return False
            if total <= 0:
                return True
            return processed >= total
        except (OSError, json.JSONDecodeError):
            return False

    def export(self, sample: VideoSample, sequence: TemporalFeatureSequence) -> dict[str, Path]:
        paths = self.output_paths(sample)
        df = pd.DataFrame(sequence.rows, columns=sequence.column_order if sequence.rows else None)

        written: dict[str, Path] = {}
        for fmt, path in paths.items():
            save_dataframe(df, path)
            written[fmt] = path
            logger.info("Exported %s (%d rows)", path, len(df))

        meta_path = self._export_meta(sample, sequence)
        written["meta"] = meta_path

        schema_path = ensure_dir(paths[next(iter(paths))].parent) / "feature_schema.json"
        if not schema_path.exists():
            self._write_schema(schema_path, sequence.column_order)

        return written

    def export_failure_manifest(self, sample: VideoSample, error: str) -> Path:
        path = (
            self._output_root
            / sample.fold_name
            / sample.subject_dir_name
            / f"{sample.video_stem}_FAILED.json"
        )
        ensure_dir(path.parent)
        payload = {
            "video_id": sample.video_id,
            "video_path": str(sample.video_path),
            "error": error,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def _meta_path(self, sample: VideoSample) -> Path:
        return (
            self._output_root
            / sample.fold_name
            / sample.subject_dir_name
            / f"{sample.video_stem}.meta.json"
        )

    def _export_meta(self, sample: VideoSample, sequence: TemporalFeatureSequence) -> Path:
        path = self._meta_path(sample)
        ensure_dir(path.parent)
        meta_dict = sequence.meta.to_dict() if sequence.meta else {}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(meta_dict, handle, indent=2, default=str)
        return path

    @staticmethod
    def _write_schema(path: Path, columns: list[str]) -> None:
        payload = {
            "feature_schema_version": SCHEMA_VERSION,
            "columns": columns,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
