"""Configuration loading and validation.

`configs/config.yaml` is the single source of truth for class names, recycling
categories, thresholds, colors, sorting rules and all training/inference settings.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_CONFIG_VAR = "EWASTE_CONFIG"

REQUIRED_SECTIONS = (
    "model", "dataset", "training", "inference", "sorting", "categories", "classes",
)


def normalize_name(name: str) -> str:
    """Normalize a class/category name: 'Mobile Phone' -> 'mobile_phone'."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class ClassSpec:
    """One detectable class as defined in the configuration."""

    id: int
    name: str
    category: str
    color: tuple[int, int, int]  # RGB
    min_confidence: float


class Config:
    """Validated, read-only view of the YAML configuration."""

    def __init__(self, raw: dict[str, Any], path: Path | None = None, root: Path = PROJECT_ROOT):
        self.raw = raw
        self.path = path
        self.root = Path(root)
        self._validate()
        self.categories: dict[str, dict[str, Any]] = {
            normalize_name(k): dict(v) for k, v in raw["categories"].items()
        }
        self.classes: list[ClassSpec] = self._build_classes()

    # ------------------------------------------------------------------ access
    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level config section (empty dict if absent)."""
        return self.raw.get(name) or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value using dot notation, e.g. ``get('training.epochs')``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, path_str: str | os.PathLike[str]) -> Path:
        """Resolve a config path: absolute stays as is, relative is under the repo root."""
        p = Path(path_str)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    @property
    def class_by_name(self) -> dict[str, ClassSpec]:
        return {c.name: c for c in self.classes}

    @property
    def class_by_id(self) -> dict[int, ClassSpec]:
        return {c.id: c for c in self.classes}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_root(self) -> Path:
        return self.resolve(self.get("dataset.root", "dataset"))

    @property
    def data_yaml_path(self) -> Path:
        return self.resolve(self.get("dataset.data_yaml", "dataset/data.yaml"))

    def split_dir(self, split: str, kind: str = "images") -> Path:
        """Directory of a split. ``kind`` is 'images' or 'labels'."""
        splits = self.get("dataset.splits", {})
        if split not in splits:
            raise ConfigError(f"Unknown split '{split}'. Defined splits: {sorted(splits)}")
        rel = str(splits[split])
        if kind == "labels":
            rel = rel.replace("images", "labels", 1)
        return self.dataset_root / rel

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a validated copy with dotted-key overrides (``**{'training.epochs': 5}``).

        ``None`` values are ignored so argparse defaults can be passed straight through.
        """
        raw = copy.deepcopy(self.raw)
        for dotted, value in overrides.items():
            if value is None:
                continue
            node = raw
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return Config(raw, self.path, self.root)

    # -------------------------------------------------------------- validation
    def _validate(self) -> None:
        raw = self.raw
        if not isinstance(raw, dict):
            raise ConfigError("Config root must be a YAML mapping.")
        missing = [s for s in REQUIRED_SECTIONS if s not in raw]
        if missing:
            raise ConfigError(f"Config is missing required section(s): {', '.join(missing)}")

        def num(key: str, lo: float, hi: float | None = None) -> None:
            val = self.get(key)
            if val is None:
                return
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ConfigError(f"'{key}' must be a number, got {val!r}")
            if val < lo or (hi is not None and val > hi):
                bound = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
                raise ConfigError(f"'{key}' must be in {bound}, got {val}")

        for key in ("inference.conf", "inference.iou", "evaluation.conf", "evaluation.iou",
                    "sorting.default_min_confidence"):
            num(key, 0.0, 1.0)
        num("inference.imgsz", 1)
        num("training.imgsz", 1)
        num("training.epochs", 1)
        num("training.workers", 0)
        num("training.lr0", 1e-12)
        batch = self.get("training.batch")
        if batch is not None and (not isinstance(batch, int) or batch == 0 or batch < -1):
            raise ConfigError(f"'training.batch' must be a positive integer or -1, got {batch!r}")

        if not isinstance(raw["classes"], list) or not raw["classes"]:
            raise ConfigError("'classes' must be a non-empty list.")
        if not isinstance(raw["categories"], dict) or not raw["categories"]:
            raise ConfigError("'categories' must be a non-empty mapping.")
        for cat, info in raw["categories"].items():
            if not isinstance(info, dict) or not info.get("destination"):
                raise ConfigError(f"Category '{cat}' needs a 'destination'.")

    def _build_classes(self) -> list[ClassSpec]:
        default_min = float(self.get("sorting.default_min_confidence", 0.35))
        specs: list[ClassSpec] = []
        seen: set[str] = set()
        for idx, entry in enumerate(self.raw["classes"]):
            if not isinstance(entry, dict) or "name" not in entry or "category" not in entry:
                raise ConfigError(f"classes[{idx}] must define 'name' and 'category'.")
            name = normalize_name(entry["name"])
            category = normalize_name(entry["category"])
            if name in seen:
                raise ConfigError(f"Duplicate class name '{name}' in config.")
            seen.add(name)
            if category not in self.categories:
                raise ConfigError(
                    f"Class '{name}' references unknown category '{category}'. "
                    f"Defined categories: {sorted(self.categories)}"
                )
            color = entry.get("color", [0, 255, 0])
            if (not isinstance(color, (list, tuple)) or len(color) != 3
                    or not all(isinstance(c, int) and 0 <= c <= 255 for c in color)):
                raise ConfigError(f"Class '{name}': color must be [R, G, B] with 0-255 ints.")
            min_conf = float(entry.get("min_confidence", default_min))
            if not 0.0 <= min_conf <= 1.0:
                raise ConfigError(f"Class '{name}': min_confidence must be within [0, 1].")
            specs.append(ClassSpec(idx, name, category, tuple(color), min_conf))  # type: ignore[arg-type]
        return specs


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the YAML config.

    Resolution order: explicit ``path`` -> ``$EWASTE_CONFIG`` -> ``configs/config.yaml``.
    """
    chosen = path or os.environ.get(ENV_CONFIG_VAR) or DEFAULT_CONFIG_PATH
    cfg_path = Path(chosen)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc
    if raw is None:
        raise ConfigError(f"Config file is empty: {cfg_path}")
    return Config(raw, cfg_path)
