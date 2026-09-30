"""Shared helpers: logging, device selection, paths, model loading, sources, data.yaml."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.errors import (
    ConfigError,
    DatasetError,
    MissingDependencyError,
    ModelNotFoundError,
    SourceError,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.config import Config

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


# ------------------------------------------------------------------ logging / io
def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=_json_default)
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def list_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Sorted image files in a directory."""
    d = Path(directory)
    if not d.is_dir():
        return []
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted(p for p in it if p.is_file() and is_image_file(p))


# ------------------------------------------------------------------------ device
def _is_gpu_request(req: str) -> bool:
    """True for '0', '0,1', 'cuda', 'cuda:0'."""
    return req.replace("cuda:", "").replace("cuda", "0").replace(",", "").isdigit()


def select_device(requested: str | int = "auto") -> str:
    """Pick the compute device and print the choice.

    ``auto`` -> CUDA GPU if available, otherwise CPU. An explicit GPU request that
    cannot be satisfied falls back to CPU with a warning (never crashes).
    """
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError(
            "PyTorch is not installed. Install it from https://pytorch.org/get-started/locally/ "
            "or run `pip install -r requirements.txt`."
        ) from exc

    req = str(requested).strip().lower()
    cuda_ok = torch.cuda.is_available()

    if req in {"", "auto"}:
        device = "0" if cuda_ok else "cpu"
    elif req == "cpu":
        device = "cpu"
    elif _is_gpu_request(req):
        if cuda_ok:
            device = req.replace("cuda:", "").replace("cuda", "0")
        else:
            logger.warning("GPU '%s' requested but CUDA is unavailable - falling back to CPU.", req)
            device = "cpu"
    else:
        device = req  # e.g. 'mps' - let Ultralytics validate it

    if device == "cpu":
        print("Device: CPU")
    elif cuda_ok and device.replace(",", "").isdigit():
        print(f"Device: CUDA GPU ({torch.cuda.get_device_name(int(device.split(',')[0]))})")
    else:
        print(f"Device: {device}")
    return device


# ------------------------------------------------------------------------- model
def resolve_weights(cfg: "Config", override: str | Path | None = None) -> tuple[str, bool]:
    """Decide which weights to load.

    Returns ``(weights, is_finetuned)``. Priority: explicit override -> trained weights
    from the config -> the pretrained base weights (with a loud warning).
    """
    if override:
        p = Path(override)
        if p.exists():
            return str(p), True
        rooted = cfg.resolve(override)
        return (str(rooted) if rooted.exists() else str(override)), True

    trained = cfg.resolve(cfg.get("model.trained_weights", "outputs/train/ewaste/weights/best.pt"))
    if trained.exists():
        return str(trained), True

    base = str(cfg.get("model.weights", "yolo11n.pt"))
    logger.warning(
        "Trained weights not found at %s. Falling back to base weights '%s', which are NOT "
        "trained on e-waste classes. Run `python src/train.py` first or pass --model.",
        trained, base,
    )
    return base, False


def load_yolo(weights: str | Path):
    """Load an Ultralytics YOLO model with friendly errors."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise MissingDependencyError(
            "The 'ultralytics' package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    p = Path(weights)
    bare_name = p.parent == Path(".")  # e.g. 'yolo11n.pt' -> Ultralytics may auto-download
    if not p.exists() and not bare_name:
        raise ModelNotFoundError(
            f"Model file not found: {p}\nTrain a model first (`python src/train.py`) or pass an "
            "existing checkpoint with --model."
        )
    try:
        return YOLO(str(weights))
    except Exception as exc:  # noqa: BLE001 - surface any loader failure clearly
        raise ModelNotFoundError(
            f"Could not load model '{weights}': {exc}\n"
            "If this is a download, check your internet connection; otherwise check the file path."
        ) from exc


# --------------------------------------------------------------------- data.yaml
def read_data_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetError(
            f"Dataset config not found: {path}\nRun `python scripts/prepare_dataset.py write-yaml`."
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise DatasetError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DatasetError(f"{path} must contain a YAML mapping.")
    return data


def build_data_yaml(cfg: "Config") -> dict[str, Any]:
    """Build the Ultralytics dataset dict from the project config (single source of truth)."""
    splits = cfg.get("dataset.splits", {})
    return {
        "path": ".",
        "train": splits.get("train", "images/train"),
        "val": splits.get("val", "images/val"),
        "test": splits.get("test", "images/test"),
        "nc": cfg.num_classes,
        "names": {c.id: c.name for c in cfg.classes},
    }


def write_data_yaml(cfg: "Config", path: Path | None = None) -> Path:
    """Write ``dataset/data.yaml`` from the config."""
    out = path or cfg.data_yaml_path
    ensure_dir(out.parent)
    header = (
        "# Generated from configs/config.yaml by:  python scripts/prepare_dataset.py write-yaml\n"
        "# Do not edit class names by hand - edit configs/config.yaml and regenerate.\n"
    )
    body = yaml.safe_dump(build_data_yaml(cfg), sort_keys=False)
    out.write_text(header + body, encoding="utf-8")
    return out


def check_data_yaml_matches_config(cfg: "Config", data: dict[str, Any]) -> None:
    """Raise if data.yaml class names/count disagree with configs/config.yaml."""
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise DatasetError("data.yaml has no valid 'names' entry.")
    got = [str(names[k]) for k in sorted(names, key=int)]
    if got != cfg.class_names or int(data.get("nc", len(got))) != cfg.num_classes:
        raise DatasetError(
            "dataset/data.yaml class names do not match configs/config.yaml.\n"
            f"  data.yaml: {got}\n  config   : {cfg.class_names}\n"
            "Run `python scripts/prepare_dataset.py write-yaml` to regenerate it."
        )


def write_runtime_data_yaml(cfg: "Config") -> Path:
    """Validate data.yaml and write a copy with an absolute dataset ``path``.

    Ultralytics resolves a relative ``path`` against its own settings directory, which
    is unreliable. Writing an absolute path makes training work from any directory
    on Windows and Linux.
    """
    data = read_data_yaml(cfg.data_yaml_path)
    check_data_yaml_matches_config(cfg, data)
    if not cfg.dataset_root.is_dir():
        raise DatasetError(f"Dataset folder not found: {cfg.dataset_root}. See docs/dataset.md.")
    data["path"] = cfg.dataset_root.as_posix()
    out = ensure_dir(cfg.resolve("outputs/.cache")) / "data_runtime.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


# ----------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Source:
    """Parsed inference source."""

    kind: str  # 'image' | 'directory' | 'video' | 'webcam'
    value: Any  # Path for files/dirs, int for webcam index


def parse_source(source: str | int) -> Source:
    """Classify a --source argument (image, directory, video or webcam index)."""
    if isinstance(source, int) or str(source).strip().isdigit():
        return Source("webcam", int(source))
    p = Path(str(source)).expanduser()
    if not p.exists():
        raise SourceError(f"Source not found: {p}")
    if p.is_dir():
        if not list_images(p):
            raise SourceError(f"No supported images found in directory: {p}")
        return Source("directory", p)
    if is_image_file(p):
        return Source("image", p)
    if is_video_file(p):
        return Source("video", p)
    raise SourceError(
        f"Unsupported file type '{p.suffix}'. Images: {sorted(IMAGE_EXTS)}; videos: {sorted(VIDEO_EXTS)}"
    )


def open_capture(src: Source):
    """Open a cv2.VideoCapture for a video file or webcam, raising friendly errors."""
    try:
        import cv2
    except ImportError as exc:
        raise MissingDependencyError(
            "opencv-python is not installed. Run `pip install -r requirements.txt`."
        ) from exc
    cap = cv2.VideoCapture(src.value if src.kind == "webcam" else str(src.value))
    if not cap.isOpened():
        cap.release()
        if src.kind == "webcam":
            raise SourceError(
                f"Could not open webcam {src.value}. Check that it is connected, not used by "
                "another app, and that camera permissions are granted. Try another index (--source 1)."
            )
        raise SourceError(f"Could not open video: {src.value}. The file may be corrupted or use an "
                          "unsupported codec.")
    return cap


def require_ok(condition: bool, message: str) -> None:
    """Raise ConfigError with ``message`` if the condition is false."""
    if not condition:
        raise ConfigError(message)
