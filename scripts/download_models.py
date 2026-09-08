"""Download and verify every model required by the local application."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODELS = [
    {
        "source_type": "huggingface",
        "repo_id": "martinjolif/yolo-football-player-detection",
        "revision": "5e83fafa8d564243001ce8e063612a618a138fbe",
        "filename": "yolo-football-player-detection.pt",
        "path": "models/yolo-football-player-detection.pt",
        "expected_sha256": "69c652bfa9814ef882c439617f04b8fd5749b6b8455aaa3c36110bc2e802aadd",
        "license": "AGPL-3.0",
        "purpose": "player, goalkeeper, referee, and fallback ball detection",
    },
    {
        "source_type": "url",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "revision": "v8.3.0",
        "filename": "yolo11n.pt",
        "path": "yolo11n.pt",
        "expected_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "license": "AGPL-3.0",
        "purpose": "supplementary ball detection",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to(model: dict[str, str], destination: Path) -> None:
    if model["source_type"] == "huggingface":
        cached = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            revision=model["revision"],
        )
        shutil.copyfile(cached, destination)
        return

    with urllib.request.urlopen(model["url"]) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def fetch_model(model: dict[str, str]) -> dict[str, str | int]:
    target = ROOT / model["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = model["expected_sha256"]

    if target.exists() and sha256(target) == expected:
        print(f"Verified existing model: {model['path']}")
    else:
        if target.exists():
            print(f"Replacing model with unexpected checksum: {model['path']}")
        else:
            print(f"Downloading model: {model['path']}")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".download",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            _download_to(model, temporary_path)
            actual = sha256(temporary_path)
            if actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {model['filename']}: expected {expected}, received {actual}."
                )
            os.replace(temporary_path, target)
            temporary_path = None
            print(f"Downloaded and verified: {model['path']}")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return {
        **model,
        "sha256": sha256(target),
        "size_bytes": target.stat().st_size,
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        manifest = [fetch_model(model) for model in MODELS]
    except Exception as error:
        raise SystemExit(f"Model setup failed: {error}") from error
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nAll required models are ready.")


if __name__ == "__main__":
    main()
