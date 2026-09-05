# Local Football Analysis Demo

A private, on-device proof of concept that estimates team-level shots, shots on target, passes, pass success, and possession from short tactical-camera football clips.

## Scope

- MP4 clips up to two minutes
- Fixed or smoothly moving wide-angle tactical camera
- Team-level estimates; no player identification
- Local processing after the initial model download
- JSON and CSV results; no annotated-video rendering

See [`CONTEXT.md`](./CONTEXT.md) for the precise football terms and `docs/adr/` for architectural decisions.

## Setup

Python 3.11 or 3.12 is recommended. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/download_models.py
```

The model-download script fetches the approved football-specific player checkpoint and records its hash and provenance in `models/manifest.json`. Model weights are ignored by Git. The demo uses the generic COCO `yolo11n.pt` checkpoint for ball detection because it performed better on the uploaded clip during local diagnostics. Both detectors are isolated behind a replaceable interface.

Shot-on-target is deliberately strict in this version: a visible goalkeeper save confirms it; outcomes without sufficient terminal evidence remain unknown rather than being guessed.

## Run

Put private clips under `data/private/input/`; this directory is ignored by Git. Then run:

```bash
source .venv/bin/activate
streamlit run app.py
```

Optional local diagnostic:

```bash
python scripts/diagnose_clip.py data/private/input/Match.mp4
```

All footage and derived data stay on the local machine. The results are estimates and must not be treated as official match statistics.

## Test

```bash
source .venv/bin/activate
pytest
```
