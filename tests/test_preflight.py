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

