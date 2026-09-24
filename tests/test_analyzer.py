from pathlib import Path

import numpy as np

from football_analysis.analyzer import AnalysisConfig, analyze_video
from football_analysis.domain import Team
from football_analysis.preflight import PreflightReport, VideoMetadata


class FakeCapture:
    def __init__(self, frame_count: int) -> None:
        self.frame_count = frame_count
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        self.position += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FakeTracker:
    device = "cpu"

    def __init__(self) -> None:
        self.reset_count = 0
        self.track_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def track(self, frame):
        self.track_count += 1
        return []


def test_twenty_minute_analysis_keeps_one_tracker_and_reports_ten_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "long.mp4"
    path.write_bytes(b"placeholder")
    metadata = VideoMetadata(path, 640, 360, 1.0, 1_200, 1_200.0)
    monkeypatch.setattr(
        "football_analysis.analyzer.inspect_video",
        lambda _: PreflightReport(metadata),
    )
    capture = FakeCapture(metadata.frame_count)
    monkeypatch.setattr("football_analysis.analyzer.cv2.VideoCapture", lambda _: capture)
    tracker = FakeTracker()
    updates: list[tuple[str, float, str]] = []
    config = AnalysisConfig(
        team_names={Team.A: "A", Team.B: "B"},
        team_colors={Team.A: "#ff0000", Team.B: "#0000ff"},
        attacks={Team.A: "right", Team.B: "left"},
    )

    result = analyze_video(path, config, tracker=tracker, progress=lambda *args: updates.append(args))

    assert result.analyzed_frames == 1_200
    assert tracker.reset_count == 1
    assert tracker.track_count == 1_200
    assert capture.released is True
    assert any("unit 10/10" in message and "CPU" in message for _, _, message in updates)
