"""Video-optimized MediaPipe Face Mesh extractor (tracking mode)."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray

from dms.domain.landmarks import LandmarkSet
from dms.landmarks.base import LandmarkExtractor
from dms.landmarks.factory import create_landmark_extractor
from dms.landmarks.schema import LandmarkSchema
from dms.utils.config import AppConfig, LandmarksConfig, load_landmark_schema
from dms.utils.device import ResolvedDevice
from dms.utils.logging import get_logger
from dms.utils.models import ensure_model_asset

logger = get_logger(__name__)


class VideoMediaPipeFaceMeshExtractor(LandmarkExtractor):
    """Face Mesh configured for video streams (static_image_mode=False)."""

    def __init__(
        self,
        config: LandmarksConfig,
        schema: LandmarkSchema,
        delegate: str = "cpu",
    ) -> None:
        self._schema = schema
        self._config = config
        self._delegate = delegate.lower()
        self._is_tasks_api = False
        self._mesh = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=config.max_num_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            logger.info("Video Face Mesh initialized (legacy API, tracking mode)")
        else:
            self._is_tasks_api = True
            model_path = ensure_model_asset("face_landmarker.task")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options_kwargs: dict = {"model_asset_path": str(model_path)}
            if self._delegate == "gpu":
                try:
                    base_options_kwargs["delegate"] = python.BaseOptions.Delegate.GPU
                    logger.info("Video Face Landmarker using GPU delegate")
                except (AttributeError, ValueError):
                    logger.warning("GPU delegate unavailable; using CPU for Face Landmarker")

            base_options = python.BaseOptions(**base_options_kwargs)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=config.max_num_faces,
                min_face_detection_confidence=config.min_detection_confidence,
                min_face_presence_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
                running_mode=vision.RunningMode.VIDEO,
            )
            self._mesh = vision.FaceLandmarker.create_from_options(options)
            self._timestamp_ms = 0
            logger.info("Video Face Landmarker initialized (Tasks API, VIDEO mode)")

    def extract(self, image_bgr: NDArray[np.uint8], timestamp_ms: int | None = None) -> LandmarkSet | None:
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
            ts = timestamp_ms if timestamp_ms is not None else self._timestamp_ms
            self._timestamp_ms = ts + 33
            results = self._mesh.detect_for_video(mp_image, ts)

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


def create_video_landmark_extractor(
    config: AppConfig,
    device: ResolvedDevice,
) -> LandmarkExtractor:
    """Create a landmark extractor suited for sequential video frames."""
    schema = LandmarkSchema.from_dict(load_landmark_schema(config.landmarks.schema_path))
    landmarks_config = config.landmarks

    if not config.video.use_video_landmark_tracking:
        return create_landmark_extractor(landmarks_config)

    provider = landmarks_config.provider.lower()
    if provider == "mediapipe_face_mesh":
        return VideoMediaPipeFaceMeshExtractor(
            landmarks_config,
            schema,
            delegate=device.mediapipe_delegate,
        )

    return create_landmark_extractor(landmarks_config)
