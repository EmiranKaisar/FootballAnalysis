from pathlib import Path

from football_analysis.preflight import inspect_video


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

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {
                5: 30.0,
                7: 870.0,
                3: 1280.0,
                4: 720.0,
            }
            return values[property_id]

        def set(self, property_id: int, value: float) -> None:
            pass

        def read(self):
            import numpy as np

            return True, np.zeros((720, 1280, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr("football_analysis.preflight.cv2.VideoCapture", lambda _: FakeCapture())

    report = inspect_video(path)

    assert report.suitable is False
    assert "at least 30 seconds" in report.errors[0]
