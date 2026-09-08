# Local model files

After creating the supported Python 3.12 environment, run `python scripts/download_models.py` from the repository root to fetch both required detectors. Checkpoints (`*.pt`) are ignored by Git; `manifest.json` records the pinned source, license, size, and verified SHA-256 of each exact local artifact.

The prototype uses `martinjolif/yolo-football-player-detection`, a YOLO11m detector for player, goalkeeper, referee, and ball classes. Its upstream model card declares AGPL-3.0. The application currently supplements its weak ball recall with the official COCO-pretrained `yolo11n.pt`, selected after a local diagnostic on the private clip. Review this and all model/data licenses before any proprietary or commercial use.
