"""MediaPipe Face Mesh landmark extraction supporting legacy solutions and modern Tasks API."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from dms.domain.landmarks import LandmarkSet
from dms.landmarks.base import LandmarkExtractor
from dms.landmarks.schema import LandmarkSchema
from dms.utils.config import LandmarksConfig
from dms.utils.models import ensure_model_asset


class MediaPipeFaceMeshExtractor(LandmarkExtractor):
    """Extract 468+ landmarks using MediaPipe Face Mesh or Tasks FaceLandmarker."""

    def __init__(self, config: LandmarksConfig, schema: LandmarkSchema) -> None:
        self._schema = schema
        self._config = config
        self._is_tasks_api = False
        self._mesh = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=config.max_num_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        else:
            self._is_tasks_api = True
            model_path = ensure_model_asset("face_landmarker.task")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=config.max_num_faces,
                min_face_detection_confidence=config.min_detection_confidence,
                min_face_presence_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)

    def extract(self, image_bgr: NDArray[np.uint8]) -> LandmarkSet | None:
        height, width = image_bgr.shape[:2]

        if not self._is_tasks_api:
            image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
            results = self._mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return None

            face_landmarks = results.multi_face_landmarks[0]
            points = np.array(
                [[lm.x * width, lm.y * height] for lm in face_landmarks.landmark],
                dtype=np.float64,
            )
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self._mesh.detect(mp_image)

            if not results.face_landmarks:
                return None

            face_landmarks = results.face_landmarks[0]
            points = np.array(
                [[lm.x * width, lm.y * height] for lm in face_landmarks],
                dtype=np.float64,
            )

        return LandmarkSet(
            points=points,
            image_width=width,
            image_height=height,
            schema_id=self._schema.schema_id,
            confidence=1.0,
        )

    def close(self) -> None:
        if self._mesh is not None and hasattr(self._mesh, "close"):
            self._mesh.close()

