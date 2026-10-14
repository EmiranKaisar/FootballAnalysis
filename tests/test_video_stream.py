from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from football_analysis.video_stream import LocalVideoServer


def test_local_video_stream_supports_seeking_and_rejects_unknown_files(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0123456789")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.mp4"
    outside.write_bytes(b"private")
    server = LocalVideoServer(tmp_path)
    try:
        url = server.url_for(video)
        with urlopen(Request(url, headers={"Range": "bytes=3-5"})) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 3-5/10"
            assert response.read() == b"345"
        with urlopen(Request(url, method="HEAD")) as response:
            assert response.headers["Content-Length"] == "10"
            assert response.read() == b""
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={"Range": "bytes=20-"}))
        assert error.value.code == 416
        with pytest.raises(HTTPError) as error:
            urlopen(url.rsplit("/", 1)[0] + "/unknown")
        assert error.value.code == 404
        with pytest.raises(ValueError):
            server.url_for(outside)
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
