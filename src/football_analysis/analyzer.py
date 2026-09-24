from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from threading import Event as CancelEvent
from time import monotonic
from typing import Callable

import cv2

from .detection import CompositeFootballTracker, ObjectTracker, TrackedObject, UltralyticsTracker
from .domain import AnalysisResult, Team
from .events import EventEngine, PlayerState
from .preflight import inspect_video
from .teams import TeamClassifier

ProgressCallback = Callable[[str, float, str], None]
LOGICAL_UNIT_SECONDS = 120.0
MAX_LOGICAL_UNITS = 10


@dataclass(frozen=True)
class AnalysisConfig:
    team_names: dict[Team, str]
    team_colors: dict[Team, str]
    attacks: dict[Team, str]
    model_path: str = "yolo11n.pt"
    ball_model_path: str | None = None
    target_fps: float = 8.0
    confidence: float = 0.18
    image_size: int = 640


def _notify(callback: ProgressCallback | None, stage: str, progress: float, message: str) -> None:
    if callback:
        callback(stage, min(1.0, max(0.0, progress)), message)


def _format_clock(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _device_label(tracker: ObjectTracker) -> str:
    device = str(getattr(tracker, "device", "custom")).lower()
    if device.startswith("mps"):
        return "Metal"
    if device.startswith("cuda"):
        return "CUDA"
    if device.startswith("cpu"):
        return "CPU"
    return device or "custom"


def analyze_video(
    path: str | Path,
    config: AnalysisConfig,
    *,
    tracker: ObjectTracker | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
) -> AnalysisResult:
    path = Path(path)
    _notify(progress, "preflight", 0.02, "Inspecting video")
    preflight = inspect_video(path)
    if not preflight.suitable or preflight.metadata is None:
        raise ValueError("; ".join(preflight.errors))
    metadata = preflight.metadata

    _notify(progress, "model", 0.05, "Loading local detection model")
    if tracker is None:
        tracker = (
            CompositeFootballTracker(
                config.model_path,
                config.ball_model_path,
                config.confidence,
                config.image_size,
            )
            if config.ball_model_path
            else UltralyticsTracker(config.model_path, config.confidence, config.image_size)
        )
    tracker.reset()
    classifier = TeamClassifier(config.team_colors)

    capture = cv2.VideoCapture(str(path))
    source_fps = metadata.fps
    frame_step = max(1, round(source_fps / min(source_fps, config.target_fps)))
    analysis_fps = source_fps / frame_step
    engine = EventEngine(metadata.width, metadata.height, analysis_fps, config.attacks)
    result = AnalysisResult(
        source_name=path.name,
        duration_seconds=metadata.duration_seconds,
        team_names=config.team_names,
        source_frames=metadata.frame_count,
        warnings=list(preflight.warnings),
    )

    frame_index = 0
    ball_seen = 0
    frames_with_players = 0
    last_detected_ball: TrackedObject | None = None
    last_ball_timestamp: float | None = None
    analysis_started = monotonic()
    total_units = min(
        MAX_LOGICAL_UNITS,
        max(1, ceil(metadata.duration_seconds / LOGICAL_UNIT_SECONDS)),
    )
    device_label = _device_label(tracker)
    _notify(
        progress,
        "analysis",
        0.08,
        f"00:00/{_format_clock(metadata.duration_seconds)} · unit 1/{total_units} · "
        f"starting · {device_label}",
    )
    try:
        while capture.isOpened():
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Analysis cancelled")
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step:
                frame_index += 1
                continue

            timestamp = frame_index / source_fps
            result.analyzed_timestamps.append(timestamp)
            objects = tracker.track(frame)
            players: list[PlayerState] = []
            for tracked in objects:
                if tracked.label not in {"player", "goalkeeper"}:
                    continue
                if tracked.label == "goalkeeper":
                    defending_side = "left" if tracked.center[0] < metadata.width / 2 else "right"
                    goalkeeper_team = next(
                        team for team, direction in config.attacks.items() if direction != defending_side
                    )
                    classification = (goalkeeper_team, 0.65)
                else:
                    classification = classifier.classify(frame, tracked)
                if classification is not None:
                    team, team_confidence = classification
                    players.append(PlayerState(tracked, team, team_confidence))
            balls = [item for item in objects if item.label == "ball"]
            detected_ball: TrackedObject | None = max(balls, key=lambda item: item.confidence) if balls else None
            ball_seen += int(detected_ball is not None)
            if detected_ball is not None:
                ball = detected_ball
                last_detected_ball = detected_ball
                last_ball_timestamp = timestamp
            elif (
                last_detected_ball is not None
                and last_ball_timestamp is not None
                and timestamp - last_ball_timestamp <= 0.5
            ):
                # Bridge only a short detector gap so one missed frame does not erase control.
                ball = TrackedObject(
                    last_detected_ball.track_id,
                    "ball",
                    last_detected_ball.confidence * 0.6,
                    last_detected_ball.bbox,
                )
            else:
                ball = None
            frames_with_players += int(len(players) >= 4)
            engine.update(timestamp, players, ball)
            result.analyzed_frames += 1

            ratio = frame_index / max(1, metadata.frame_count)
            if result.analyzed_frames % 10 == 0:
                wall_elapsed = max(0.001, monotonic() - analysis_started)
                throughput = result.analyzed_frames / wall_elapsed
                remaining_frames = max(0, metadata.frame_count - frame_index) / frame_step
                eta_seconds = remaining_frames / throughput if throughput > 0 else 0.0
                unit = min(total_units, int(timestamp // LOGICAL_UNIT_SECONDS) + 1)
                message = (
                    f"{_format_clock(timestamp)}/{_format_clock(metadata.duration_seconds)} · "
                    f"unit {unit}/{total_units} · {throughput:.1f} fps · {device_label} · "
                    f"ETA {_format_clock(eta_seconds)}"
                )
                _notify(progress, "analysis", 0.08 + 0.88 * ratio, message)
            frame_index += 1
    finally:
        capture.release()

    engine.finish(metadata.duration_seconds)
    result.events = engine.events
    result.controlled_seconds = engine.possession_seconds
    result.eligible_seconds = engine.eligible_seconds
    result.intervals = engine.intervals
    result.warnings.append(
        "Shot-on-target detection currently confirms visible goalkeeper saves; "
        "goals without a tracked terminal ball may remain unknown."
    )
    if result.analyzed_frames:
        ball_coverage = ball_seen / result.analyzed_frames
        player_coverage = frames_with_players / result.analyzed_frames
        if ball_coverage < 0.15:
            result.warnings.append(
                f"Ball detection coverage was only {100 * ball_coverage:.1f}%; event estimates are unreliable."
            )
        if player_coverage < 0.70:
            result.warnings.append(
                f"At least four players were detected in only {100 * player_coverage:.1f}% of analyzed frames."
            )
    _notify(progress, "complete", 1.0, "Analysis complete")
    return result
