"""Project path resolution utilities."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to project root or an explicit base."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    root = base if base is not None else get_project_root()
    return (root / candidate).resolve()
