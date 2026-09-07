from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class PreflightReport:
    metadata: VideoMetadata | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sampled_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    probable_cuts: int = 0

    @property
    def suitable(self) -> bool:
        return not self.errors


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def inspect_video(path: str | Path, sample_count: int = 12) -> PreflightReport:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return PreflightReport(None, errors=["Video file does not exist."])
    if path.suffix.lower() != ".mp4":
        return PreflightReport(None, errors=["Version one accepts MP4 files only."])

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return PreflightReport(None, errors=["OpenCV could not decode this MP4 file."])

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    metadata = VideoMetadata(path, width, height, fps, frame_count, duration)
    report = PreflightReport(metadata)

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        report.errors.append("The video has invalid or unreadable stream metadata.")
    if duration > 120.5:
        report.errors.append(f"Clip duration is {duration:.1f}s; the limit is 120s.")
    if duration < 30:
        report.errors.append("Clip must contain at least 30 seconds of play.")
    if width < 640 or height < 360:
        report.errors.append(f"Resolution {width}×{height} is below the supported minimum 640×360.")
    if fps and not 20 <= fps <= 60:
        report.warnings.append(f"Frame rate {fps:.2f} FPS is outside the preferred 20–60 FPS range.")

    if frame_count > 0:
        indices = np.linspace(0, max(0, frame_count - 1), num=min(sample_count, frame_count), dtype=int)
        histograms: list[np.ndarray] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                report.sampled_frames.append(frame)
                histograms.append(_histogram(frame))
        for previous, current in zip(histograms, histograms[1:]):
            correlation = cv2.compareHist(previous, current, cv2.HISTCMP_CORREL)
            if correlation < 0.25:
                report.probable_cuts += 1
        if report.probable_cuts:
            report.warnings.append(
                f"Detected {report.probable_cuts} probable viewpoint cut(s); version one expects continuous footage."
            )
    capture.release()
    if len(report.sampled_frames) < min(3, sample_count):
        report.errors.append("Too few frames could be decoded for analysis.")
    return report
