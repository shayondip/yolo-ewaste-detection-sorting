"""Full evaluation report: metrics + confusion matrix + PR curves + training curves + example detections.

Usage:
    python src/evaluate.py                       # split from config (default: test)
    python src/evaluate.py --split val --examples 20
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import Config, load_config
from src.dataset_audit import audit_dataset
from src.errors import EwasteError
from src.inference import Detector
from src.utils import ensure_dir, list_images, setup_logging
from src.validate import format_summary, run_validation, save_metrics
from src.visualization import (draw_detections, plot_bar_counts, plot_class_distribution,
                               plot_confusion_matrix, plot_training_curves)

logger = logging.getLogger("evaluate")

# Plots that Ultralytics writes during validation and that we simply copy next to ours.
ULTRALYTICS_PLOTS = ["BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                     "confusion_matrix.png", "confusion_matrix_normalized.png"]


def save_detection_examples(cfg: Config, weights: str | None, split: str, n: int, out_dir: Path,
                            device: str | None) -> int:
    """Run the Detector on the first ``n`` images of a split and save annotated results."""
    images = list_images(cfg.split_dir(split, "images"))[:n]
    if not images:
        return 0
    detector = Detector(cfg, weights=weights, device=device)
    ensure_dir(out_dir)
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), draw_detections(frame, detector.predict_frame(frame), cfg))
    return len(images)


def evaluate(cfg: Config, weights: str | None, split: str, examples: int, device: str | None) -> dict[str, Any]:
    """Run validation and generate all evaluation artifacts from real outputs."""
    metrics = run_validation(cfg, weights, split, device)
    save_metrics(cfg, metrics)

    viz = ensure_dir(cfg.resolve(cfg.get("evaluation.visualization_dir", "outputs/visualizations")))
    cm = metrics.get("confusion_matrix")
    if cm:
        plot_confusion_matrix(cm["matrix"], cm["labels"], viz / f"confusion_matrix_{split}.png")
        plot_confusion_matrix(cm["matrix"], cm["labels"], viz / f"confusion_matrix_{split}_normalized.png", normalize=True)

    per_class_ap = {n: v["mAP50"] for n, v in metrics["per_class"].items() if v}
    if per_class_ap:
        plot_bar_counts(per_class_ap, f"Per-class mAP@50 ({split})", "mAP@50", viz / f"per_class_map50_{split}.png")

    counts = audit_dataset(cfg.dataset_root, cfg.class_names, dict(cfg.get("dataset.splits", {})))
    plot_class_distribution(counts["splits"][split]["class_counts"], cfg, viz / f"class_distribution_{split}.png",
                            f"Ground-truth class distribution ({split})")

    ul_dir = Path(metrics["meta"]["ultralytics_output_dir"])
    for name in ULTRALYTICS_PLOTS:
        if (ul_dir / name).exists():
            shutil.copy2(ul_dir / name, viz / f"ultralytics_{split}_{name}")

    train_csv = cfg.resolve(cfg.get("training.project_dir", "outputs/train")) / cfg.get("training.run_name", "ewaste") / "results.csv"
    if train_csv.exists():
        plot_training_curves(train_csv, viz / "training_curves.png")

    n = save_detection_examples(cfg, weights, split, examples, viz / f"examples_{split}", device)
    logger.info("Saved %d example detections", n)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained model and generate a report.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default=None, choices=["train", "val", "test"], help="Default: evaluation.split")
    ap.add_argument("--examples", type=int, default=None, help="Number of annotated example images")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        split = args.split or cfg.get("evaluation.split", "test")
        examples = args.examples if args.examples is not None else int(cfg.get("evaluation.num_examples", 12))
        metrics = evaluate(cfg, args.model, split, examples, args.device)
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print(format_summary(metrics))
    print(f"\nArtifacts: {cfg.resolve(cfg.get('evaluation.output_dir'))}  and  "
          f"{cfg.resolve(cfg.get('evaluation.visualization_dir'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
