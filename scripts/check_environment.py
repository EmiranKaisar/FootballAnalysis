"""Check the supported Python runtime, dependencies, models, and accelerator."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

from download_models import MODELS, ROOT, sha256

PACKAGES = {
    "cv2": "opencv-python",
    "huggingface_hub": "huggingface-hub",
    "lap": "lap",
    "numpy": "numpy",
    "streamlit": "streamlit",
    "torch": "torch",
    "ultralytics": "ultralytics",
}


def main() -> None:
    failures: list[str] = []
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info[:2] == (3, 12):
        print(f"[OK] Python {version}")
    else:
        failures.append(f"Python {version} is unsupported; install Python 3.12 and recreate .venv.")

    for module_name, distribution_name in PACKAGES.items():
        try:
            importlib.import_module(module_name)
            package_version = importlib.metadata.version(distribution_name)
            print(f"[OK] {distribution_name} {package_version}")
        except Exception as error:
            failures.append(f"{distribution_name} is unavailable: {error}")

    for model in MODELS:
        path = ROOT / model["path"]
        if not path.is_file():
            failures.append(f"Missing model {model['path']}; run: python scripts/download_models.py")
        elif sha256(path) != model["expected_sha256"]:
            failures.append(f"Model checksum is invalid: {model['path']}; run the downloader again.")
        else:
            print(f"[OK] Model {model['path']}")

    try:
        import torch

        if torch.backends.mps.is_available():
            accelerator = "Metal (MPS)"
        elif torch.cuda.is_available():
            accelerator = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            accelerator = "CPU"
        print(f"[OK] Analysis device: {accelerator}")
    except Exception:
        pass

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"[ERROR] {failure}")
        raise SystemExit(1)
    print("\nEnvironment is ready. Start the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
