"""In-memory dataset index."""

from __future__ import annotations

from collections import defaultdict

from dms.domain.enums import DriverState, Split
from dms.domain.sample import DriverSample


class DatasetRegistry:
    """Queryable index of driver samples."""

    def __init__(self, samples: list[DriverSample] | None = None) -> None:
        self._samples: dict[str, DriverSample] = {}
        if samples:
            for sample in samples:
                self.add(sample)

    def add(self, sample: DriverSample) -> None:
        self._samples[sample.sample_id] = sample

    def extend(self, samples: list[DriverSample]) -> None:
        for sample in samples:
            self.add(sample)

    @property
    def samples(self) -> list[DriverSample]:
        return list(self._samples.values())

    def __len__(self) -> int:
        return len(self._samples)

    def filter(
        self,
        split: Split | None = None,
        label: DriverState | None = None,
        subject_id: str | None = None,
    ) -> list[DriverSample]:
        results = self.samples
        if split is not None:
            results = [s for s in results if s.split == split]
        if label is not None:
            results = [s for s in results if s.label == label]
        if subject_id is not None:
            results = [s for s in results if s.subject_id == subject_id]
        return results

    def group_by_split(self) -> dict[Split, list[DriverSample]]:
        grouped: dict[Split, list[DriverSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.split].append(sample)
        return dict(grouped)

    def group_by_subject(self) -> dict[str, list[DriverSample]]:
        grouped: dict[str, list[DriverSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.subject_id].append(sample)
        return dict(grouped)
