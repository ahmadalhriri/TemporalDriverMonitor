from dms.pipeline.video_executor import resolve_worker_count
from dms.utils.config import AppConfig, ComputeConfig


def test_resolve_worker_count_uses_configured_value() -> None:
    config = AppConfig(compute=ComputeConfig(num_workers=4))

    assert resolve_worker_count(config) == 4


def test_resolve_worker_count_clamps_to_one() -> None:
    config = AppConfig(compute=ComputeConfig(num_workers=0))

    assert resolve_worker_count(config) == 1
