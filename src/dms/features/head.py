"""Head pose features via solvePnP."""

from __future__ import annotations

import cv2
import numpy as np

from dms.features.base import FeatureCalculator, FeatureContext


# Generic 3D face model for head pose (millimeters, approximate)
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye outer
        [43.3, 32.7, -26.0],    # right eye outer
        [-28.9, -28.9, -24.1],  # left mouth
        [28.9, -28.9, -24.1],   # right mouth
    ],
    dtype=np.float64,
)


class HeadPoseFeatureCalculator(FeatureCalculator):
    group = "head"

    def compute(self, context: FeatureContext) -> dict[str, float]:
        points = context.points
        indices = context.schema.head_pose_indices
        image_points = np.array([points[i] for i in indices], dtype=np.float64)

        height, width = context.image_bgr.shape[:2]
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            _MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return {"pitch": 0.0, "roll": 0.0, "yaw": 0.0, "head_pose_valid": 0.0}

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "head_pose_valid": 1.0,
        }
