"""The production perception backend: YOLOv8 pose plus instance segmentation.

Two checkpoints run per image because they answer different questions. The pose
model gives joints, which carry lengths and the torso frame. The segmentation
model gives per-instance masks, which carry breadths. Neither substitutes for
the other: a skeleton cannot tell you a waist, and a mask cannot tell you where
the waist *is*.

Ultralytics and OpenCV are optional extras. Importing this module without them
is fine; :meth:`YoloPerception.available` reports false and the registry falls
back, so a slim deployment that only needs the geometric path installs neither.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from arcbody import keypoints as kp
from arcbody.config import PerceptionSettings
from arcbody.errors import BackendUnavailableError
from arcbody.perception.base import PerceptionResult, largest_component, rank_observations
from arcbody.perception.view import classify_view
from arcbody.types import BoundingBox, PersonObservation

logger = logging.getLogger(__name__)

#: COCO class index for "person". The seg model is trained on all 80 classes and
#: will happily hand back chairs.
PERSON_CLASS = 0


class YoloPerception:
    """Ultralytics-backed detection, pose and segmentation.

    Models are loaded lazily on first use: constructing the backend must stay
    cheap so the registry can probe availability during startup without paying
    for two checkpoint loads it may never need.
    """

    name = "yolo"

    def __init__(self, settings: PerceptionSettings) -> None:
        self.settings = settings
        self._pose: Any | None = None
        self._seg: Any | None = None

    # -- lifecycle --------------------------------------------------------

    def available(self) -> bool:
        try:
            import importlib.util
        except ImportError:  # pragma: no cover - importlib is always present
            return False
        return all(
            importlib.util.find_spec(module) is not None
            for module in ("ultralytics", "cv2")
        )

    def _load(self) -> tuple[Any, Any]:
        if self._pose is not None and self._seg is not None:
            return self._pose, self._seg
        if not self.available():
            raise BackendUnavailableError(
                "the yolo backend needs the 'yolo' extra: pip install 'arcbody[yolo]'",
                backend=self.name,
            )
        from ultralytics import YOLO

        logger.info(
            "loading yolo checkpoints pose=%s seg=%s device=%s",
            self.settings.pose_weights,
            self.settings.seg_weights,
            self.settings.device,
        )
        self._pose = YOLO(self.settings.pose_weights)
        self._seg = YOLO(self.settings.seg_weights)
        return self._pose, self._seg

    def warm_up(self) -> None:
        """Load and run once, so the first real request is not the slow one."""
        pose, seg = self._load()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        pose.predict(blank, verbose=False, device=self.settings.device)
        seg.predict(blank, verbose=False, device=self.settings.device)

    # -- inference --------------------------------------------------------

    def analyse(self, image: np.ndarray) -> PerceptionResult:
        pose_model, seg_model = self._load()
        height, width = image.shape[:2]

        pose_results = pose_model.predict(
            image, verbose=False, device=self.settings.device, classes=[PERSON_CLASS]
        )
        masks = self._person_masks(seg_model, image, (height, width))

        observations: list[PersonObservation] = []
        for result in pose_results:
            boxes = getattr(result, "boxes", None)
            poses = getattr(result, "keypoints", None)
            if boxes is None or poses is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            points = poses.data.cpu().numpy()
            for index in range(len(xyxy)):
                score = float(scores[index])
                if score < self.settings.min_detection_score:
                    continue
                bbox = BoundingBox(*(float(v) for v in xyxy[index])).clipped(width, height)
                if bbox.area <= 0:
                    continue
                observations.append(
                    PersonObservation(
                        bbox=bbox,
                        keypoints=self._normalise_keypoints(points[index]),
                        image_size=(width, height),
                        detection_score=score,
                        mask=self._match_mask(masks, bbox),
                        backend=self.name,
                    )
                )

        for observation in observations:
            observation.view, _ = classify_view(
                observation.keypoints, self.settings.min_keypoint_score
            )
        return PerceptionResult(
            observations=rank_observations(observations, (width, height)),
            backend=self.name,
        )

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _normalise_keypoints(raw: np.ndarray) -> np.ndarray:
        """Coerce ultralytics output to the canonical ``(17, 3)`` layout.

        The pose head emits ``(x, y, conf)`` per joint in COCO order already, so
        this is a shape and dtype guard rather than a remapping — but it is the
        one place a future backend swap would need to reorder, so it stays
        explicit.
        """
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 2 and array.shape == (kp.NUM_KEYPOINTS, 3):
            return array
        if array.ndim == 2 and array.shape == (kp.NUM_KEYPOINTS, 2):
            points = kp.empty()
            points[:, :2] = array
            points[:, 2] = 1.0
            return points
        raise ValueError(f"unexpected keypoint payload of shape {array.shape}")

    def _person_masks(
        self, seg_model: Any, image: np.ndarray, shape: tuple[int, int]
    ) -> list[np.ndarray]:
        """Full-resolution boolean masks for every detected person."""
        import cv2

        height, width = shape
        results = seg_model.predict(
            image, verbose=False, device=self.settings.device, classes=[PERSON_CLASS]
        )
        masks: list[np.ndarray] = []
        for result in results:
            data = getattr(result, "masks", None)
            if data is None:
                continue
            for plane in data.data.cpu().numpy():
                # Ultralytics returns masks at the model's letterboxed
                # resolution; nearest-neighbour keeps the silhouette edge crisp,
                # and that edge is exactly what the breadths are read from.
                resized = cv2.resize(
                    plane.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                )
                mask = largest_component(resized.astype(bool))
                if mask.any():
                    masks.append(mask)
        return masks

    @staticmethod
    def _match_mask(masks: list[np.ndarray], bbox: BoundingBox) -> np.ndarray | None:
        """Pick the mask that best fills a pose detection's box.

        Pose and segmentation run as separate models and their detections are
        not index-aligned, so the two are paired by overlap. A mask that covers
        less than a fifth of the box is a different person standing behind this
        one, and pairing it would measure the wrong body.
        """
        if not masks:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in bbox.as_tuple())
        if x2 <= x1 or y2 <= y1:
            return None
        best_mask, best_overlap = None, 0.0
        box_area = float((x2 - x1) * (y2 - y1))
        for mask in masks:
            inside = float(mask[y1:y2, x1:x2].sum())
            overlap = inside / max(box_area, 1.0)
            # Penalise masks that spill far outside the box.
            spill = float(mask.sum()) - inside
            overlap -= 0.5 * spill / max(box_area, 1.0)
            if overlap > best_overlap:
                best_mask, best_overlap = mask, overlap
        return best_mask if best_overlap >= 0.20 else None
