"""Benchmark inference speed on CPU and (if available) GPU. All numbers are measured, never invented.

Reports per device: latency (mean/median/p95), FPS, plus model size and parameter count.
Timing covers the full Ultralytics predict call (pre-process + inference + NMS) on a
single image at a time (batch = 1), after warm-up runs.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model outputs/train/ewaste/weights/best.pt --runs 100
    python scripts/benchmark.py --images dataset/images/test --devices cpu
"""

from __future__ import annotations

import argparse
import logging
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config, load_config
from src.errors import EwasteError, MissingDependencyError
from src.utils import list_images, load_yolo, resolve_weights, save_json, setup_logging

logger = logging.getLogger("benchmark")


def _load_inputs(images_dir: str | None, imgsz: int, n: int) -> tuple[list[np.ndarray], str]:
    """Real images if provided, else a fixed-seed synthetic image (labelled as such in the report)."""
    if images_dir:
        import cv2
        frames = [f for f in (cv2.imread(str(p)) for p in list_images(images_dir)[:n]) if f is not None]
        if frames:
            return frames, f"{len(frames)} real image(s) from {images_dir}"
        logger.warning("No readable images in %s - falling back to a synthetic image.", images_dir)
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)], "synthetic random image (latency only)"


def bench_device(model: Any, device: str, frames: list[np.ndarray], imgsz: int, warmup: int, runs: int) -> dict[str, Any]:
    import torch

    def sync() -> None:
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    for i in range(warmup):
        model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
    times: list[float] = []
    for i in range(runs):
        sync()
        t0 = time.perf_counter()
        model.predict(frames[i % len(frames)], imgsz=imgsz, device=device, verbose=False)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    mean = statistics.fmean(times)
    return {
        "device": "cpu" if device == "cpu" else f"cuda:{device} ({torch.cuda.get_device_name(int(device))})",
        "runs": runs, "warmup": warmup,
        "latency_ms": {"mean": round(mean, 2), "median": round(statistics.median(times), 2),
                       "p95": round(times[min(len(times) - 1, int(0.95 * len(times)))], 2),
                       "min": round(times[0], 2), "max": round(times[-1], 2)},
        "fps": round(1000 / mean, 2),
    }


def run(cfg: Config, weights: str | None, images: str | None, imgsz: int, warmup: int, runs: int,
        devices: list[str]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise MissingDependencyError("PyTorch is required. Run `pip install -r requirements.txt`.") from exc

    path, finetuned = resolve_weights(cfg, weights)
    model = load_yolo(path)
    frames, input_desc = _load_inputs(images, imgsz, max(runs, 1))
    n_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else None
    size_mb = round(Path(path).stat().st_size / 1e6, 2) if Path(path).exists() else None

    results = []
    for dev in devices:
        if dev != "cpu" and not torch.cuda.is_available():
            logger.warning("Skipping GPU benchmark: CUDA is not available on this machine.")
            results.append({"device": "cuda", "skipped": "CUDA not available"})
            continue
        logger.info("Benchmarking on %s ...", dev)
        results.append(bench_device(model, dev, frames, imgsz, warmup, runs))

    return {
        "model": str(path), "finetuned_weights": finetuned, "model_size_mb": size_mb, "parameters": n_params,
        "imgsz": imgsz, "batch_size": 1, "input": input_desc,
        "system": {"python": platform.python_version(), "os": platform.platform(),
                   "processor": platform.processor() or "unknown", "torch": torch.__version__,
                   "cuda_available": torch.cuda.is_available()},
        "results": results,
    }


def print_report(rep: dict[str, Any]) -> None:
    print("\n" + "=" * 64 + "\nBENCHMARK\n" + "=" * 64)
    print(f"Model      : {rep['model']}  ({rep['model_size_mb']} MB, {rep['parameters']:,} params)")
    print(f"Input      : {rep['input']} @ imgsz={rep['imgsz']}, batch=1")
    print(f"System     : {rep['system']['os']} | torch {rep['system']['torch']}")
    for r in rep["results"]:
        if "skipped" in r:
            print(f"\n[{r['device'].upper()}] skipped - {r['skipped']}")
            continue
        l = r["latency_ms"]
        print(f"\n[{r['device']}]  {r['runs']} runs after {r['warmup']} warm-up")
        print(f"  latency ms  mean {l['mean']} | median {l['median']} | p95 {l['p95']} | min {l['min']} | max {l['max']}")
        print(f"  FPS         {r['fps']}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU vs GPU inference benchmark.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--images", default=None, help="Folder of real images to cycle through")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--devices", nargs="+", default=["cpu", "0"], help="e.g. cpu 0 (GPU 0 skipped if no CUDA)")
    ap.add_argument("--output", default=None, help="JSON path (default outputs/metrics/benchmark.json)")
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        imgsz = args.imgsz or int(cfg.get("inference.imgsz", 640))
        rep = run(cfg, args.model, args.images, imgsz, args.warmup, args.runs, args.devices)
        out = Path(args.output) if args.output else cfg.resolve(cfg.get("evaluation.output_dir")) / "benchmark.json"
        save_json(rep, out)
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print_report(rep)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
