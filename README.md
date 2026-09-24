# Local Football Analysis Demo

A private, on-device proof of concept that estimates team-level shots, shots on target, passes, pass success, and possession from continuous tactical-camera football clips.

## Scope

- MP4 clips from 30 seconds through 20 minutes, up to 4 GB
- Fixed or smoothly moving wide-angle tactical camera
- Team-level estimates; no player identification
- Local processing after the initial model download
- Interactive full-clip and range-filtered reports; no annotated-video rendering

See [`DESIGN.md`](./DESIGN.md) for the complete product design, [`CONTEXT.md`](./CONTEXT.md) for precise domain terms, and `docs/adr/` for architectural decisions.

## Twenty-minute analysis

The video is not physically split or independently analyzed. OpenCV decodes one source sequentially, and the models receive one sampled frame at a time. The application organizes the work into ten logical two-minute progress units while preserving one continuous tracker and event engine across every boundary.

At the current target of approximately eight analyzed frames per second, a 20-minute clip produces roughly 9,600 analyzed frames and approximately 19,200 inference calls when both reference detectors are enabled. Logical work units improve progress reporting, cancellation, diagnostics, and resource control; they do not reduce the total inference work.

The 20-minute implementation:

- accepts clips from 30 seconds through 20 minutes;
- requires one continuous segment from a single match half, without a halftime or attacking-direction change;
- retains the current approximately 8 FPS analysis target until benchmark and accuracy evidence supports changing it;
- allows only one active inference job per local server;
- reports elapsed clip time, logical work-unit count, measured throughput, device, and estimated time remaining;
- uses duration-aware, memory-bounded preflight sampling;
- accepts files up to 4 GB and configures Streamlit's upload/message limits accordingly;
- preserves tracker, pending-event, ball-history, and possession state across logical boundaries;
- keeps the completed full-clip evidence available for immediate range filtering without reanalysis; and
- does not promise resume after a Streamlit or machine restart in the first version.

Large browser uploads may temporarily require substantial memory and an additional local copy. Placing a clip directly in `data/private/input/` is the preferred path for multi-gigabyte files. Before copying an upload, the application verifies free space for the file plus a 512 MB reserve, writes atomically, and removes inactive uploaded session files after seven days. Completed job records are bounded to the most recent 16 per local server process.

## Requirements

- Python 3.12 (other Python versions are not supported)
- An internet connection for dependency installation and the first model download
- A local MP4 clip from 30 seconds through 20 minutes and no larger than 4 GB

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

### Referenced models and repositories

The default configuration references:

- [`martinjolif/yolo-football-player-detection`](https://huggingface.co/martinjolif/yolo-football-player-detection) for football-specific player, goalkeeper, referee, and ball classes. The downloader pins model revision [`5e83fafa8d564243001ce8e063612a618a138fbe`](https://huggingface.co/martinjolif/yolo-football-player-detection/tree/5e83fafa8d564243001ce8e063612a618a138fbe).
- The official [`yolo11n.pt` checkpoint from the Ultralytics assets v8.3.0 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) for supplementary generic person and sports-ball coverage.
- The [`ultralytics/ultralytics`](https://github.com/ultralytics/ultralytics) repository and Python package for YOLO inference and ByteTrack-based tracking.

These are reference defaults, not a requirement that every installation use the same trained weights.

### Using alternative models

An alternative Ultralytics-compatible model can be used without changing application code:

1. Store the custom weights under a different local filename, such as `models/custom-football-model.pt`.
2. Start the app and enter that path in **Local model path**.
3. Either leave **Local ball model path** empty and use one model that covers all required objects, or provide a second compatible model for generic player and ball coverage.

Do not overwrite the two default model files. Running `python scripts/download_models.py` verifies their pinned checksums and restores a default file whose contents do not match.

Compatible models must provide bounding boxes and confidence scores. Ultralytics supplies stable ByteTrack IDs across frames. Model class names are normalized as follows:

| Analysis object | Accepted model class names |
| --- | --- |
| Player | `person`, `player` |
| Goalkeeper | `goalkeeper`, `goal keeper` |
| Ball | `ball`, `football`, `sports ball`, `soccer ball` |
| Official | `referee`, `official`, `linesman` |

Other classes are ignored. Team identity does not need to be part of the detector; the application classifies detected player crops using the user-confirmed jersey colors.

A model served through another runtime, such as ONNX Runtime or TorchVision, requires a small adapter implementing the `ObjectTracker` protocol in `src/football_analysis/detection.py`. Its `track(frame)` method must return `TrackedObject` values with a stable track ID, one of the canonical labels above, a confidence score, and an `(x1, y1, x2, y2)` pixel bounding box. `analyze_video()` accepts an injected tracker for testing, but selecting a non-Ultralytics backend from the interface requires additional provider wiring.

The environment check confirms Python, installed packages, both model files, and the analysis device. CPU operation is supported; compatible Macs may report Metal (MPS), while a compatible Windows PyTorch installation may report CUDA.

Both detectors are isolated behind a replaceable interface.

Shot-on-target is deliberately strict in this version: a visible goalkeeper save confirms it; outcomes without sufficient terminal evidence remain unknown rather than being guessed.

## Run

Put private clips under `data/private/input/`; this directory is ignored by Git. Then run:

```bash
source .venv/bin/activate
streamlit run app.py
```

To stop the server, return to the terminal where it is running and press **Control+C** (`Ctrl+C`). If that terminal does not respond, open another terminal in the project directory and run:

```bash
pkill -f "streamlit run app.py"
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

Original project code is available under the [MIT License](./LICENSE). Third-party packages, replacement models, and default model weights keep their respective licenses and are not relicensed by this project. The referenced football model declares AGPL-3.0, and Ultralytics offers its runtime and YOLO models under AGPL-3.0 or separate enterprise terms. Review the license of every selected model and runtime before redistribution, proprietary use, or commercial use.
