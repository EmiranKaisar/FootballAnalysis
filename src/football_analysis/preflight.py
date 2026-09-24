from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import cv2
import numpy as np

MIN_DURATION_SECONDS = 30.0
MAX_DURATION_SECONDS = 20 * 60.0
DURATION_TOLERANCE_SECONDS = 0.5
MAX_FILE_SIZE_BYTES = 4 * 1024**3
PREFLIGHT_SAMPLE_INTERVAL_SECONDS = 30.0
MIN_PREFLIGHT_SAMPLES = 12
MAX_PREFLIGHT_SAMPLES = 48


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


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _adaptive_sample_count(duration_seconds: float, frame_count: int) -> int:
    duration_samples = ceil(max(0.0, duration_seconds) / PREFLIGHT_SAMPLE_INTERVAL_SECONDS) + 1
    return min(frame_count, MAX_PREFLIGHT_SAMPLES, max(MIN_PREFLIGHT_SAMPLES, duration_samples))


def inspect_video(
    path: str | Path,
    sample_count: int | None = None,
    *,
    retain_sampled_frames: bool = False,
) -> PreflightReport:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return PreflightReport(None, errors=["Video file does not exist."])
    if path.suffix.lower() != ".mp4":
        return PreflightReport(None, errors=["Version one accepts MP4 files only."])
    file_size = _file_size(path)
    if file_size > MAX_FILE_SIZE_BYTES:
        return PreflightReport(
            None,
            errors=[f"File size is {file_size / 1024**3:.2f} GB; the limit is 4 GB."],
        )

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
    if duration > MAX_DURATION_SECONDS + DURATION_TOLERANCE_SECONDS:
        report.errors.append(f"Clip duration is {duration:.1f}s; the limit is 20 minutes.")
    if duration < MIN_DURATION_SECONDS:
        report.errors.append("Clip must contain at least 30 seconds of play.")
    if width < 640 or height < 360:
        report.errors.append(f"Resolution {width}×{height} is below the supported minimum 640×360.")
    if fps and not 20 <= fps <= 60:
        report.warnings.append(f"Frame rate {fps:.2f} FPS is outside the preferred 20–60 FPS range.")
    if duration > 120:
        report.warnings.append(
            "Long clips must remain within one match half and must not include an attacking-direction change."
        )

    if frame_count > 0:
        resolved_sample_count = (
            max(1, min(sample_count, frame_count))
            if sample_count is not None
            else _adaptive_sample_count(duration, frame_count)
        )
        indices = np.linspace(0, max(0, frame_count - 1), num=resolved_sample_count, dtype=int)
        histograms: list[np.ndarray] = []
        decoded_samples = 0
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                decoded_samples += 1
                if retain_sampled_frames:
                    report.sampled_frames.append(frame)
                histograms.append(_histogram(frame))
        for previous, current in zip(histograms, histograms[1:]):
            correlation = cv2.compareHist(previous, current, cv2.HISTCMP_CORREL)
            if correlation < 0.25:
                report.probable_cuts += 1
        if report.probable_cuts:
            report.warnings.append(
                f"Detected {report.probable_cuts} probable viewpoint cut(s); continuous footage is expected."
            )
        if decoded_samples < min(3, resolved_sample_count):
            report.errors.append("Too few frames could be decoded for analysis.")
    capture.release()
    return report
