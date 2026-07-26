"""CLI entry point for the DMS Phase 1 pipeline."""

from __future__ import annotations

import argparse
import sys

from dms.pipeline.runner import run_from_config
from dms.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DMS Phase 1 — FL3D feature extraction and rule-based evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to process (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples per split (debug mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    overrides = {}
    if args.max_samples is not None:
        overrides = {"dataset": {"max_samples": args.max_samples}}

    from dms.utils.config import load_config

    config = load_config(args.config, overrides=overrides if overrides else None)
    from dms.pipeline.runner import PipelineRunner

    runner = PipelineRunner(config)
    result = runner.run(splits=args.splits)
    print(f"Processed {result.samples_processed} samples")
    print(f"Feature dataset: {result.feature_dataset_path_parquet}")
    print(f"Run directory: {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
