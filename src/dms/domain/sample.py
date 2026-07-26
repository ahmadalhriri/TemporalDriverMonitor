"""Annotated driver sample from FL3D (labels only — landmarks ignored)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dms.domain.enums import DriverState, Split


@dataclass(frozen=True)
class DriverSample:
    """One FL3D image with ground-truth label. No annotation landmarks are stored."""

    sample_id: str
    image_path: Path
    subject_id: str
    split: Split
    label: DriverState

    @property
    def image_name(self) -> str:
        return self.image_path.name
