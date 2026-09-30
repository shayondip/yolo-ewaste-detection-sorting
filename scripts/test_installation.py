"""Environment sanity check. Run this first:  python scripts/test_installation.py"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED = [("ultralytics", "ultralytics"), ("torch", "torch"), ("cv2", "opencv-python"), ("numpy", "numpy"),
            ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("yaml", "pyyaml"), ("PIL", "Pillow"),
            ("streamlit", "streamlit"), ("pytest", "pytest")]


def main() -> int:
    failures = 0
    print(f"Python {platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 11):
        print("  [WARN] Python 3.11+ recommended")
    for module, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module)
            print(f"  [ OK ] {pip_name:<14} {getattr(mod, '__version__', '')}")
        except ImportError:
            failures += 1
            print(f"  [FAIL] {pip_name:<14} missing -> pip install {pip_name}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [ OK ] CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  [INFO] No CUDA GPU detected - CPU will be used (training will be slow).")
    except ImportError:
        pass

    try:
        from src.config import load_config
        from src.utils import check_data_yaml_matches_config, read_data_yaml
        cfg = load_config()
        print(f"  [ OK ] config loaded: {cfg.num_classes} classes {cfg.class_names}")
        check_data_yaml_matches_config(cfg, read_data_yaml(cfg.data_yaml_path))
        print("  [ OK ] dataset/data.yaml matches config")
        n = sum(len(list(cfg.split_dir(s).glob('*.*'))) for s in cfg.get("dataset.splits", {}) if cfg.split_dir(s).is_dir())
        print(f"  [INFO] image files currently in dataset/: {n}" + ("  (see docs/dataset.md)" if n == 0 else ""))
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"  [FAIL] project config: {exc}")

    try:
        import cv2
        cap = cv2.VideoCapture(0)
        print("  [INFO] webcam 0: " + ("available" if cap.isOpened() else "not available (optional)"))
        cap.release()
    except Exception:  # noqa: BLE001
        pass

    print("\n" + ("All required checks passed." if not failures else f"{failures} check(s) failed."))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
