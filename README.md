# Local Football Analysis Demo

A private, on-device proof of concept that estimates team-level shots, shots on target, passes, pass success, and possession from short tactical-camera football clips.

## Scope

- MP4 clips from 30 seconds through two minutes
- Fixed or smoothly moving wide-angle tactical camera
- Team-level estimates; no player identification
- Local processing after the initial model download
- Interactive full-clip and range-filtered reports; no annotated-video rendering

See [`DESIGN.md`](./DESIGN.md) for the complete product design, [`CONTEXT.md`](./CONTEXT.md) for precise domain terms, and `docs/adr/` for architectural decisions.

## Requirements

- Python 3.12 (other Python versions are not supported)
- An internet connection for dependency installation and the first model download
- A local MP4 clip between 30 seconds and two minutes

## Setup on macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/download_models.py
python scripts/check_environment.py
```

## Setup on Windows

Install Python 3.12 from [python.org](https://www.python.org/downloads/), open PowerShell in the project directory, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_models.py
python scripts/check_environment.py
```

If this repository already has a `.venv` created with another Python version, delete that environment and recreate it with Python 3.12 before installing dependencies.

The single model-download command fetches both required checkpoints:

- `models/yolo-football-player-detection.pt` for players, goalkeepers, referees, and fallback ball observations
- `yolo11n.pt` for supplementary ball detection

The downloader pins immutable model releases, verifies their SHA-256 checksums, reuses valid existing files, and records local provenance in `models/manifest.json`. Model weights and the generated manifest are ignored by Git. After the first successful download, analysis does not require internet access.

The environment check confirms Python, installed packages, both model files, and the analysis device. CPU operation is supported; compatible Macs may report Metal (MPS), while a compatible Windows PyTorch installation may report CUDA.

Both detectors are isolated behind a replaceable interface.

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

## License

Original project code is available under the [MIT License](./LICENSE). Third-party packages and model weights keep their respective licenses and are not relicensed by this project. In particular, the current Ultralytics runtime and YOLO model weights are distributed under AGPL-3.0 terms by their maintainers. Review those terms before redistribution, proprietary use, or commercial use.
