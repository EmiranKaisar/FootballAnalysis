"""Run a small local detection diagnostic without analyzing the full clip."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from football_analysis.detection import CompositeFootballTracker, UltralyticsTracker
from football_analysis.preflight import inspect_video
from football_analysis.teams import suggest_team_colors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/yolo-football-player-detection.pt"))
    parser.add_argument("--ball-model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    report = inspect_video(args.video, sample_count=args.samples)
    if not report.suitable or report.metadata is None:
        raise SystemExit("; ".join(report.errors))
    tracker = (
        CompositeFootballTracker(args.model, args.ball_model)
        if args.ball_model.exists()
        else UltralyticsTracker(args.model)
    )
    counts = {"player": 0, "ball": 0, "official": 0}
    frames_with_ball = 0
    samples = []
    started = time.perf_counter()
    for frame in report.sampled_frames:
        objects = tracker.track(frame)
        samples.append((frame, objects))
        frames_with_ball += int(any(item.label == "ball" for item in objects))
        for label in counts:
            counts[label] += sum(item.label == label for item in objects)
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "video": str(args.video),
                "metadata": {
                    "width": report.metadata.width,
                    "height": report.metadata.height,
                    "fps": report.metadata.fps,
                    "duration_seconds": report.metadata.duration_seconds,
                },
                "model": str(args.model),
                "device": tracker.device,
                "sample_frames": len(samples),
                "seconds": round(elapsed, 3),
                "counts": counts,
                "frames_with_ball": frames_with_ball,
                "suggested_team_colors": suggest_team_colors(samples),
                "warnings": report.warnings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
