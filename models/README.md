# Local model files

Run `python scripts/download_models.py` to fetch the approved prototype detector. Checkpoints (`*.pt`) are ignored by Git; `manifest.json` records the source, license, size, and SHA-256 of the exact local artifact.

The prototype uses `martinjolif/yolo-football-player-detection`, a YOLO11m detector for player, goalkeeper, referee, and ball classes. Its upstream model card declares AGPL-3.0. The application currently supplements its weak ball recall with the official COCO-pretrained `yolo11n.pt`, selected after a local diagnostic on the private clip. Review this and all model/data licenses before any proprietary or commercial use.
