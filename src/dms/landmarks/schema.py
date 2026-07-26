"""Landmark index schema loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LandmarkSchema:
    """Semantic mapping of MediaPipe Face Mesh indices."""

    schema_id: str
    num_landmarks: int
    left_eye: tuple[int, ...]
    right_eye: tuple[int, ...]
    mouth_left_corner: int
    mouth_right_corner: int
    mouth_upper_lip: int
    mouth_lower_lip: int
    head_pose_indices: tuple[int, ...]
    left_iris: int | None
    right_iris: int | None
    symmetry_pairs: tuple[tuple[int, int], ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LandmarkSchema":
        mouth = data["mouth"]
        head_pose = data["head_pose"]
        iris = data.get("iris", {})
        symmetry = data.get("symmetry_pairs", [])

        return cls(
            schema_id=str(data["schema_id"]),
            num_landmarks=int(data["num_landmarks"]),
            left_eye=tuple(int(i) for i in data["left_eye"]),
            right_eye=tuple(int(i) for i in data["right_eye"]),
            mouth_left_corner=int(mouth["left_corner"]),
            mouth_right_corner=int(mouth["right_corner"]),
            mouth_upper_lip=int(mouth["upper_lip"]),
            mouth_lower_lip=int(mouth["lower_lip"]),
            head_pose_indices=tuple(
                int(head_pose[k])
                for k in ("nose_tip", "chin", "left_eye_outer", "right_eye_outer", "left_mouth", "right_mouth")
            ),
            left_iris=int(iris["left"]) if iris.get("left") is not None else None,
            right_iris=int(iris["right"]) if iris.get("right") is not None else None,
            symmetry_pairs=tuple((int(a), int(b)) for a, b in symmetry),
        )

    def get_indices(self, group: str) -> tuple[int, ...]:
        mapping = {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
        }
        if group not in mapping:
            raise KeyError(f"Unknown landmark group: {group}")
        return mapping[group]
