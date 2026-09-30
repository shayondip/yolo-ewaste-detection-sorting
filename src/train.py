"""Train (fine-tune) a pretrained Ultralytics YOLO detector on the e-waste dataset.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch 8 --device cpu
    python src/train.py --model yolo11s.pt --name exp_small
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
from src.dataset_audit import run_audit
from src.errors import DatasetError, EwasteError
from src.utils import (ensure_dir, load_yolo, save_json, select_device, setup_logging,
                       write_runtime_data_yaml)

logger = logging.getLogger("train")

# Column names in Ultralytics' results.csv (after whitespace stripping)
CSV_METRICS = {
    "precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)", "mAP50-95": "metrics/mAP50-95(B)",
    "train_box_loss": "train/box_loss", "train_cls_loss": "train/cls_loss", "train_dfl_loss": "train/dfl_loss",
    "val_box_loss": "val/box_loss", "val_cls_loss": "val/cls_loss", "val_dfl_loss": "val/dfl_loss",
}


def summarize_training(save_dir: Path) -> dict[str, Any]:
    """Read results.csv from a finished run: final-epoch and best-epoch metrics."""
    from src.visualization import read_results_csv

    df = read_results_csv(save_dir / "results.csv")
    last = df.iloc[-1]
    summary: dict[str, Any] = {"epochs_completed": int(len(df)),
                               "final_epoch": {k: float(last[v]) for k, v in CSV_METRICS.items() if v in df}}
    if CSV_METRICS["mAP50-95"] in df:
        best = df.loc[df[CSV_METRICS["mAP50-95"]].idxmax()]
        summary["best_epoch"] = {"epoch": int(best.get("epoch", -1)),
                                 **{k: float(best[v]) for k, v in CSV_METRICS.items() if v in df}}
    return summary


def train(cfg: Config, weights: str | None = None, skip_audit: bool = False, resume: bool = False) -> dict[str, Any]:
    """Run training and return a summary dict (also saved to outputs/metrics)."""
    tr = cfg.section("training")
    device = select_device(tr.get("device", "auto"))
    data_yaml = write_runtime_data_yaml(cfg)

    if not skip_audit:
        report = run_audit(cfg, save=True)
        train_images = report["splits"].get("train", {}).get("images", 0)
        val_images = report["splits"].get("val", {}).get("images", 0)
        if train_images == 0 or val_images == 0:
            raise DatasetError(
                f"Training needs images in both train ({train_images}) and val ({val_images}) splits. "
                "See docs/dataset.md and scripts/prepare_dataset.py.")
        if not report["summary"]["passed"]:
            raise DatasetError(
                f"Dataset audit found {report['summary']['errors']} error(s). See "
                "outputs/metrics/dataset_audit.txt, fix them, or re-run with --skip-audit.")

    base = weights or cfg.get("model.weights", "yolo11n.pt")
    model = load_yolo(base)
    project = ensure_dir(cfg.resolve(tr.get("project_dir", "outputs/train")))
    logger.info("Fine-tuning '%s' for %s epochs on device %s", base, tr.get("epochs"), device)

    results = model.train(
        data=str(data_yaml), epochs=int(tr["epochs"]), imgsz=int(tr["imgsz"]), batch=int(tr["batch"]),
        lr0=float(tr["lr0"]), lrf=float(tr.get("lrf", 0.01)), patience=int(tr.get("patience", 50)),
        workers=int(tr["workers"]), device=device, seed=int(tr.get("seed", 0)),
        project=str(project), name=str(tr.get("run_name", "ewaste")),
        exist_ok=bool(tr.get("exist_ok", True)), pretrained=True, resume=resume, plots=True,
    )
    save_dir = Path(getattr(results, "save_dir", None) or model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    summary = summarize_training(save_dir)
    summary.update({"base_weights": str(base), "device": device, "save_dir": str(save_dir),
                    "best_weights": str(best), "classes": cfg.class_names})
    out_dir = ensure_dir(cfg.resolve(cfg.get("evaluation.output_dir", "outputs/metrics")))
    save_json(summary, out_dir / "training_summary.json")

    try:
        from src.visualization import plot_training_curves
        plot_training_curves(save_dir / "results.csv",
                             cfg.resolve(cfg.get("evaluation.visualization_dir", "outputs/visualizations"))
                             / "training_curves.png")
    except Exception as exc:  # noqa: BLE001 - plotting must never fail a finished training run
        logger.warning("Could not create training curves: %s", exc)

    _print_summary(summary)
    return summary


def _print_summary(s: dict[str, Any]) -> None:
    fe = s["final_epoch"]
    print("\n" + "=" * 60 + "\nTRAINING FINISHED\n" + "=" * 60)
    print(f"Epochs completed : {s['epochs_completed']}")
    print(f"Precision        : {fe.get('precision', float('nan')):.4f}")
    print(f"Recall           : {fe.get('recall', float('nan')):.4f}")
    print(f"mAP@50           : {fe.get('mAP50', float('nan')):.4f}")
    print(f"mAP@50-95        : {fe.get('mAP50-95', float('nan')):.4f}")
    print("Train loss (box/cls/dfl): "
          f"{fe.get('train_box_loss', float('nan')):.4f} / {fe.get('train_cls_loss', float('nan')):.4f} / "
          f"{fe.get('train_dfl_loss', float('nan')):.4f}")
    print("Val loss   (box/cls/dfl): "
          f"{fe.get('val_box_loss', float('nan')):.4f} / {fe.get('val_cls_loss', float('nan')):.4f} / "
          f"{fe.get('val_dfl_loss', float('nan')):.4f}")
    print(f"Best weights     : {s['best_weights']}")
    print("Metrics on the validation split. For an unbiased number run: python src/evaluate.py")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO on the e-waste dataset.")
    ap.add_argument("--config", default=None, help="Config YAML (default configs/config.yaml)")
    ap.add_argument("--model", default=None, help="Pretrained weights (overrides model.weights)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto | cpu | 0 | 0,1")
    ap.add_argument("--name", default=None, help="Run name (folder under outputs/train)")
    ap.add_argument("--skip-audit", action="store_true", help="Skip the pre-training dataset audit")
    ap.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config).with_overrides(**{
            "training.epochs": args.epochs, "training.batch": args.batch, "training.imgsz": args.imgsz,
            "training.lr0": args.lr0, "training.workers": args.workers, "training.device": args.device,
            "training.run_name": args.name,
        })
        train(cfg, args.model, args.skip_audit, args.resume)
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Training interrupted. Use --resume to continue.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
