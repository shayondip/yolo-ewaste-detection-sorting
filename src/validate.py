"""Validate a trained model on a dataset split and save machine- and human-readable metrics.

Usage:
    python src/validate.py
    python src/validate.py --split val --model outputs/train/ewaste/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config
from src.errors import DatasetError, EwasteError
from src.utils import (ensure_dir, list_images, load_yolo, resolve_weights, save_json, select_device,
                       setup_logging, write_runtime_data_yaml)

logger = logging.getLogger("validate")


def extract_metrics(results: Any, cfg: Config) -> dict[str, Any]:
    """Convert an Ultralytics DetMetrics object into a plain dict."""
    box = results.box
    names = {int(k): v for k, v in results.names.items()}
    per_class: dict[str, Any] = {n: None for n in cfg.class_names}
    for i, class_idx in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[names[int(class_idx)]] = {
            "precision": float(p), "recall": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)}
    out: dict[str, Any] = {
        "precision": float(box.mp), "recall": float(box.mr),
        "mAP50": float(box.map50), "mAP50-95": float(box.map),
        "per_class": per_class,
        "classes_without_ground_truth": [n for n, v in per_class.items() if v is None],
        "speed_ms_per_image": {k: float(v) for k, v in getattr(results, "speed", {}).items()},
    }
    cm = getattr(results, "confusion_matrix", None)
    if cm is not None:
        out["confusion_matrix"] = {"labels": [names[i] for i in sorted(names)] + ["background"],
                                   "matrix": cm.matrix.tolist(),
                                   "orientation": "rows = predicted, columns = true"}
    return out


def run_validation(cfg: Config, weights: str | None, split: str, device: str | None = None,
                   conf: float | None = None, iou: float | None = None) -> dict[str, Any]:
    """Run Ultralytics validation on a split and return metrics (also includes metadata)."""
    if not list_images(cfg.split_dir(split, "images")):
        raise DatasetError(f"No images found for split '{split}' in {cfg.split_dir(split, 'images')}. "
                           "See docs/dataset.md.")
    ev = cfg.section("evaluation")
    dev = select_device(device or cfg.get("inference.device", "auto"))
    model_path, is_finetuned = resolve_weights(cfg, weights)
    model = load_yolo(model_path)
    data_yaml = write_runtime_data_yaml(cfg)
    val_dir = ensure_dir(cfg.resolve(ev.get("val_project_dir", "outputs/val")))

    results = model.val(
        data=str(data_yaml), split=split, imgsz=int(cfg.get("inference.imgsz", 640)),
        batch=int(ev.get("batch", 16)), conf=float(conf if conf is not None else ev.get("conf", 0.001)),
        iou=float(iou if iou is not None else ev.get("iou", 0.6)), device=dev, plots=True,
        project=str(val_dir), name=split, exist_ok=True, verbose=False,
    )
    metrics = extract_metrics(results, cfg)
    metrics["meta"] = {"split": split, "weights": str(model_path), "device": dev,
                       "finetuned_weights": is_finetuned, "ultralytics_output_dir": str(results.save_dir)}
    return metrics


def format_summary(metrics: dict[str, Any]) -> str:
    """Human-readable metrics summary."""
    m = metrics["meta"]
    L = ["=" * 64, f"EVALUATION SUMMARY  (split: {m['split']})", "=" * 64,
         f"Weights : {m['weights']}", f"Device  : {m['device']}", "",
         f"Precision : {metrics['precision']:.4f}", f"Recall    : {metrics['recall']:.4f}",
         f"mAP@50    : {metrics['mAP50']:.4f}", f"mAP@50-95 : {metrics['mAP50-95']:.4f}", "",
         f"{'class':<16}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}"]
    for name, v in metrics["per_class"].items():
        L.append(f"{name:<16}" + (f"{v['precision']:>8.3f}{v['recall']:>8.3f}{v['mAP50']:>8.3f}{v['mAP50-95']:>10.3f}"
                                   if v else "   (no ground-truth instances in this split)"))
    if not m["finetuned_weights"]:
        L += ["", "WARNING: base (non-fine-tuned) weights were used; these numbers are not meaningful."]
    L.append("=" * 64)
    return "\n".join(L)


def save_metrics(cfg: Config, metrics: dict[str, Any]) -> Path:
    """Write metrics.json (latest), metrics_<split>.json and a text summary."""
    out = ensure_dir(cfg.resolve(cfg.get("evaluation.output_dir", "outputs/metrics")))
    split = metrics["meta"]["split"]
    save_json(metrics, out / "metrics.json")
    save_json(metrics, out / f"metrics_{split}.json")
    (out / f"summary_{split}.txt").write_text(format_summary(metrics), encoding="utf-8")
    return out / "metrics.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="Weights path (default: trained weights from config)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        metrics = run_validation(cfg, args.model, args.split, args.device, args.conf, args.iou)
        path = save_metrics(cfg, metrics)
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print(format_summary(metrics))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
