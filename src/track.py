"""Track e-waste objects across video frames, assign IDs and count each object once.

Uses the Ultralytics built-in trackers (ByteTrack / BoT-SORT, set in configs/config.yaml).

Usage:
    python src/track.py --source path/to/video.mp4
    python src/track.py --source 0 --conf 0.4        # webcam
    python src/track.py --source video.mp4 --tracker botsort.yaml

Limitation: tracker identity switches can make the unique count slightly too high;
occlusion and fast motion increase this. Treat counts as estimates.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.errors import EwasteError, SourceError
from src.inference import Detector
from src.predict import run_stream
from src.utils import parse_source, save_json, setup_logging

logger = logging.getLogger("track")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-object tracking + unique counting + sorting.")
    ap.add_argument("--source", required=True, help="Video path or webcam index")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--tracker", default=None, help="bytetrack.yaml | botsort.yaml (default from config)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        src = parse_source(args.source)
        if src.kind not in {"video", "webcam"}:
            raise SourceError("Tracking needs a video file or webcam index (not an image or folder).")
        out_dir = Path(args.output_dir) if args.output_dir else cfg.resolve(cfg.get("tracking.output_dir"))
        detector = Detector(cfg, args.model, args.conf, args.iou, args.device)
        if args.tracker:
            detector.tracker = args.tracker
        result = run_stream(detector, src, out_dir, True, args.show or src.kind == "webcam",
                            not args.no_save, args.max_frames)
        summary = detector.sorter.summary()
        result.update({"model": detector.weights, "device": detector.device, "tracker": detector.tracker,
                       "sorting_summary": summary})
        stem = Path(args.source).stem if src.kind == "video" else "webcam"
        json_path = save_json(result, out_dir / f"{stem}_track_summary.json")
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print("\nUnique objects per class:", summary["by_class"])
    print("Unique objects per category:", summary["by_category"])
    print(f"Total unique objects: {summary['total_objects']}  (tracks seen: {summary['unique_tracks']})")
    print(f"Saved results to: {json_path}")
    print(cfg.get("project.disclaimer", "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
