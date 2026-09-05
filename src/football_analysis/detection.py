from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


class ObjectTracker(Protocol):
    def reset(self) -> None: ...

    def track(self, frame: np.ndarray) -> list[TrackedObject]: ...


class UltralyticsTracker:
    """Replaceable local tracker supporting generic COCO or football-specific YOLO weights."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.18,
        image_size: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.model = YOLO(self.model_path)
        self.names = {int(key): str(value).lower() for key, value in self.model.names.items()}
        self.device = self._choose_device()

    @staticmethod
    def _choose_device() -> str:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def reset(self) -> None:
        # Ultralytics constructs tracker state when track() is first called.
        predictors = getattr(self.model, "predictor", None)
        if predictors is not None and hasattr(predictors, "trackers"):
            for tracker in predictors.trackers:
                tracker.reset()

    @staticmethod
    def _canonical_label(raw: str) -> str | None:
        label = raw.lower().replace("_", " ").strip()
        if label in {"ball", "football", "sports ball", "soccer ball"}:
            return "ball"
        if label in {"referee", "official", "linesman"}:
            return "official"
        if label in {"goalkeeper", "goal keeper"}:
            return "goalkeeper"
        if label in {"person", "player"}:
            return "player"
        return None

    def track(self, frame: np.ndarray) -> list[TrackedObject]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        if boxes.xyxy is None or boxes.cls is None:
            return []
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(len(boxes)))
        classes = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        tracked: list[TrackedObject] = []
        for track_id, class_id, confidence, bbox in zip(ids, classes, confidences, coordinates):
            canonical = self._canonical_label(self.names.get(class_id, str(class_id)))
            if canonical is None:
                continue
            tracked.append(
                TrackedObject(
                    track_id=int(track_id),
                    label=canonical,
                    confidence=float(confidence),
                    bbox=tuple(float(value) for value in bbox),
                )
            )
        return tracked


class CompositeFootballTracker:
    """Merge football-specific roles with generic player and ball coverage."""

    def __init__(
        self,
        person_model_path: str | Path,
        ball_model_path: str | Path,
        confidence: float = 0.18,
        image_size: int = 640,
    ) -> None:
        self.people = UltralyticsTracker(person_model_path, confidence, image_size)
        self.generic = UltralyticsTracker(ball_model_path, max(0.03, confidence * 0.2), max(1280, image_size))
        self.device = self.people.device

    def reset(self) -> None:
        self.people.reset()
        self.generic.reset()

    @staticmethod
    def _iou(left: TrackedObject, right: TrackedObject) -> float:
        lx1, ly1, lx2, ly2 = left.bbox
        rx1, ry1, rx2, ry2 = right.bbox
        intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
            0.0, min(ly2, ry2) - max(ly1, ry1)
        )
        union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
        return intersection / union if union > 0 else 0.0

    def track(self, frame: np.ndarray) -> list[TrackedObject]:
        football_objects = [item for item in self.people.track(frame) if item.label != "ball"]
        generic_objects = self.generic.track(frame)
        generic_people = [item for item in generic_objects if item.label == "player"]
        merged_people: list[TrackedObject] = []

        for generic in generic_people:
            matches = [item for item in football_objects if self._iou(generic, item) >= 0.45]
            role = max(matches, key=lambda item: self._iou(generic, item)).label if matches else "player"
            merged_people.append(
                TrackedObject(generic.track_id, role, generic.confidence, generic.bbox)
            )

        # Keep football-model observations that the generic model did not cover.
        for football in football_objects:
            if not any(self._iou(football, generic) >= 0.45 for generic in generic_people):
                merged_people.append(
                    TrackedObject(1_000_000 + football.track_id, football.label, football.confidence, football.bbox)
                )

        balls = [item for item in generic_objects if item.label == "ball"]
        return merged_people + balls
