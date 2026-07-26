"""CLI entry point for the video temporal feature pipeline."""

from __future__ import annotations

import argparse
import sys

from dms.pipeline.video_pipeline import run_video_from_config
from dms.utils.logging import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DMS Video Pipeline — extract temporal feature sequences from videos",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/video_default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos to process (debug mode)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    overrides = {}
    if args.max_videos is not None:
        overrides = {"video": {"max_videos": args.max_videos}}

    from dms.utils.config import load_config

    config = load_config(args.config, overrides=overrides if overrides else None)

    from dms.pipeline.video_pipeline import VideoPipelineRunner

    runner = VideoPipelineRunner(config)
    result = runner.run()

    print(f"Discovered {result.videos_discovered} videos")
    print(f"Processed {result.videos_processed} | Skipped {result.videos_skipped} | Failed {result.videos_failed}")
    print(f"Total frames exported: {result.total_frames}")
    print(f"Run directory: {result.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
