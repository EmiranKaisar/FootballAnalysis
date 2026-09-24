from pathlib import Path

import numpy as np

from football_analysis.preflight import MAX_FILE_SIZE_BYTES, inspect_video


class FakeCapture:
    def __init__(self, *, fps: float = 30.0, frame_count: int = 900) -> None:
        self.fps = fps
        self.frame_count = frame_count
        self.read_count = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            5: self.fps,
            7: float(self.frame_count),
            3: 640.0,
            4: 360.0,
        }[property_id]

    def set(self, property_id: int, value: float) -> None:
        pass

    def read(self):
        self.read_count += 1
        return True, np.zeros((360, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    report = inspect_video(tmp_path / "missing.mp4")

    assert report.suitable is False
    assert "does not exist" in report.errors[0]


def test_non_mp4_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "clip.mov"
    path.write_bytes(b"not a video")

    report = inspect_video(path)

    assert report.suitable is False
    assert "MP4" in report.errors[0]


def test_clip_shorter_than_thirty_seconds_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "short.mp4"
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "football_analysis.preflight.cv2.VideoCapture",
        lambda _: FakeCapture(frame_count=870),
    )

    report = inspect_video(path)

    assert report.suitable is False
    assert "at least 30 seconds" in report.errors[0]


def test_exactly_twenty_minutes_is_accepted_with_bounded_sampling(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "twenty-minutes.mp4"
    path.write_bytes(b"placeholder")
    capture = FakeCapture(frame_count=36_000)
    monkeypatch.setattr("football_analysis.preflight.cv2.VideoCapture", lambda _: capture)

    report = inspect_video(path)

    assert report.suitable is True
    assert report.metadata is not None
    assert report.metadata.duration_seconds == 1_200.0
    assert capture.read_count == 41
    assert report.sampled_frames == []


def test_clip_beyond_duration_tolerance_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "too-long.mp4"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "football_analysis.preflight.cv2.VideoCapture",
        lambda _: FakeCapture(frame_count=36_030),
    )

    report = inspect_video(path)

    assert report.suitable is False
    assert any("20 minutes" in error for error in report.errors)


def test_file_over_four_gigabytes_is_rejected(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "large.mp4"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr("football_analysis.preflight._file_size", lambda _: MAX_FILE_SIZE_BYTES + 1)
    monkeypatch.setattr(
        "football_analysis.preflight.cv2.VideoCapture",
        lambda _: FakeCapture(),
    )

    report = inspect_video(path)

    assert report.suitable is False
    assert any("limit is 4 GB" in error for error in report.errors)


def test_frames_are_retained_only_when_requested(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "colors.mp4"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "football_analysis.preflight.cv2.VideoCapture",
        lambda _: FakeCapture(),
    )

    report = inspect_video(path, sample_count=6, retain_sampled_frames=True)

    assert len(report.sampled_frames) == 6
