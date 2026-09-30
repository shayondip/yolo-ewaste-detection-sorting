"""Export the trained model for deployment (ONNX by default).

Usage:
    python scripts/export.py                                  # ONNX from trained weights
    python scripts/export.py --format onnx --dynamic --simplify
    python scripts/export.py --format openvino                # other Ultralytics formats also work
    python scripts/export.py --model path/to/best.pt --imgsz 640

Ultralytics writes the exported model next to the .pt file. Export needs extra packages
for some formats (e.g. `pip install onnx onnxruntime` for ONNX) - Ultralytics will try to
install them automatically, or tell you what is missing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.errors import EwasteError
from src.utils import load_yolo, resolve_weights, save_json, setup_logging

logger = logging.getLogger("export")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a trained YOLO model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="Weights (default: trained weights from config)")
    ap.add_argument("--format", default="onnx", help="onnx | openvino | torchscript | engine | tflite ...")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--dynamic", action="store_true", help="Dynamic input shapes (ONNX)")
    ap.add_argument("--simplify", action="store_true", help="Simplify the ONNX graph")
    ap.add_argument("--half", action="store_true", help="FP16 export (GPU)")
    ap.add_argument("--device", default="cpu", help="Export device (default cpu)")
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        weights, finetuned = resolve_weights(cfg, args.model)
        if not finetuned:
            logger.warning("Exporting NON-fine-tuned base weights (no trained model found).")
        model = load_yolo(weights)
        imgsz = args.imgsz or int(cfg.get("inference.imgsz", 640))
        try:
            out = model.export(format=args.format, imgsz=imgsz, dynamic=args.dynamic,
                               simplify=args.simplify, half=args.half, device=args.device)
        except Exception as exc:  # noqa: BLE001
            logger.error("Export to '%s' failed: %s\nInstall the format's dependencies (see Ultralytics "
                         "export docs) and retry.", args.format, exc)
            return 2
        info = {"source_weights": str(weights), "format": args.format, "imgsz": imgsz, "exported_file": str(out),
                "class_names": cfg.class_names}
        save_json(info, Path(str(out)).parent / "export_info.json")
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print(f"\nExported: {out}\nClass order (for your runtime): {cfg.class_names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
