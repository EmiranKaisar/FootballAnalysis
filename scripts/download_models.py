"""Download the approved prototype checkpoint and record its exact hash."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODELS = [
    {
        "repo_id": "martinjolif/yolo-football-player-detection",
        "revision": "main",
        "filename": "yolo-football-player-detection.pt",
        "license": "AGPL-3.0",
        "purpose": "player, goalkeeper, referee, and ball detection",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for model in MODELS:
        target = MODEL_DIR / model["filename"]
        if not target.exists():
            cached = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                revision=model["revision"],
            )
            shutil.copyfile(cached, target)
        manifest.append(
            {**model, "path": str(target.relative_to(ROOT)), "sha256": sha256(target), "size_bytes": target.stat().st_size}
        )
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
