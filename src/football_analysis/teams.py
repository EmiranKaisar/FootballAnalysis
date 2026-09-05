from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import TrackedObject
from .domain import Team


def hex_to_bgr(value: str) -> np.ndarray:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError("Team color must use #RRGGBB format")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return np.array([blue, green, red], dtype=np.uint8)


def bgr_to_hex(color: np.ndarray) -> str:
    blue, green, red = (int(round(value)) for value in color)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _jersey_color(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(x1 + 0.2 * (x2 - x1))))
    right = max(left + 1, min(width, int(x2 - 0.2 * (x2 - x1))))
    top = max(0, min(height - 1, int(y1 + 0.12 * (y2 - y1))))
    bottom = max(top + 1, min(height, int(y1 + 0.52 * (y2 - y1))))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Suppress common grass pixels while retaining black kits.
    mask = ~(((hsv[..., 0] >= 30) & (hsv[..., 0] <= 95) & (hsv[..., 1] > 45)) | (hsv[..., 2] < 10))
    pixels = crop[mask]
    if len(pixels) < 8:
        pixels = crop.reshape(-1, 3)
    return np.median(pixels, axis=0).astype(np.uint8)


def suggest_team_colors(samples: list[tuple[np.ndarray, list[TrackedObject]]]) -> tuple[str, str] | None:
    colors: list[np.ndarray] = []
    for frame, objects in samples:
        for player in (item for item in objects if item.label == "player"):
            color = _jersey_color(frame, player.bbox)
            if color is not None:
                colors.append(color)
    if len(colors) < 4:
        return None
    data = np.float32(colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, _, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return bgr_to_hex(centers[0]), bgr_to_hex(centers[1])


@dataclass
class TeamClassifier:
    team_colors: dict[Team, str]

    def __post_init__(self) -> None:
        reference = np.uint8([[hex_to_bgr(self.team_colors[Team.A]), hex_to_bgr(self.team_colors[Team.B])]])
        self._reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[0].astype(float)

    def classify(self, frame: np.ndarray, player: TrackedObject) -> tuple[Team, float] | None:
        color = _jersey_color(frame, player.bbox)
        if color is None:
            return None
        lab = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        distances = np.linalg.norm(self._reference_lab - lab, axis=1)
        selected = int(np.argmin(distances))
        separation = abs(float(distances[0] - distances[1]))
        confidence = min(1.0, separation / 55.0)
        return (Team.A if selected == 0 else Team.B, confidence)
