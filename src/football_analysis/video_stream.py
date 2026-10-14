"""Serve selected private clips to the local browser without loading them into RAM."""

from __future__ import annotations

import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


class LocalVideoServer:
    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir.resolve()
        self.paths: dict[str, Path] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.serve_video(send_body=True)

            def do_HEAD(self) -> None:
                self.serve_video(send_body=False)

            def serve_video(self, *, send_body: bool) -> None:
                path = server.paths.get(self.path.removeprefix("/"))
                if path is None or not path.is_file():
                    self.send_error(404)
                    return

                size = path.stat().st_size
                start, end = 0, size - 1
                requested_range = self.headers.get("Range")
                if requested_range:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range)
                    if not match or not any(match.groups()):
                        self.send_range_error(size)
                        return
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), end) if last else end
                    else:
                        start = max(0, size - int(last))
                    if start > end or start >= size:
                        self.send_range_error(size)
                        return

                self.send_response(206 if requested_range else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "private, no-store")
                self.send_header("Content-Length", str(end - start + 1))
                if requested_range:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not send_body:
                    return
                try:
                    with path.open("rb") as source:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Seeking cancels the browser's previous range request.

            def send_range_error(self, size: int) -> None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = self
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        Thread(target=self.httpd.serve_forever, daemon=True, name="local-video-stream").start()

    def url_for(self, path: Path) -> str:
        path = path.resolve()
        if not path.is_file() or not path.is_relative_to(self.input_dir):
            raise ValueError("Video must be an existing file in the private input directory.")
        token = next((key for key, value in self.paths.items() if value == path), None)
        if token is None:
            token = secrets.token_urlsafe(24)
            self.paths[token] = path
        return f"http://127.0.0.1:{self.httpd.server_port}/{token}"
