"""Run detection + sorting on an image, image folder, video file or webcam.

Usage:
    python src/predict.py --source path/to/image.jpg
    python src/predict.py --source path/to/images_folder/
    python src/predict.py --source path/to/video.mp4
    python src/predict.py --source 0                      # webcam (press 'q' to quit)
    python src/predict.py --source img.jpg --conf 0.4 --model outputs/train/ewaste/weights/best.pt

Note: video/webcam counts here are per-detection (an object seen in 100 frames counts 100x).
For unique-object counting use `python src/track.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import Config, load_config
from src.errors import EwasteError, SourceError
from src.inference import Detector
from src.utils import (Source, ensure_dir, list_images, open_capture, parse_source, save_json,
                       setup_logging)
from src.visualization import draw_detections, draw_overlay_text

logger = logging.getLogger("predict")


def predict_images(detector: Detector, images: list[Path], out_dir: Path) -> dict[str, Any]:
    """Detect on image files, save annotated copies, return a JSON-ready result."""
    ensure_dir(out_dir)
    per_image: list[dict[str, Any]] = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Skipping unreadable/corrupted image: %s", path)
            continue
        dets = detector.predict_frame(frame)
        out_path = out_dir / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), draw_detections(frame, dets, detector.cfg))
        per_image.append({"image": str(path), "output": str(out_path),
                          "inference_ms": round(detector.last_inference_ms, 1), "detections": dets})
        logger.info("%s: %d object(s) -> %s", path.name, len(dets), out_path)
        for d in dets:
            print(f"  Detected: {d['class_name']} | Confidence: {d['confidence']:.2f} | "
                  f"Category: {d['category']} | Decision: {d['decision']}")
    if not per_image:
        raise SourceError("No readable images were processed.")
    return {"type": "images", "processed": len(per_image), "results": per_image}


def run_stream(detector: Detector, src: Source, out_dir: Path, use_tracking: bool = False,
               show: bool = False, save: bool = True, max_frames: int | None = None) -> dict[str, Any]:
    """Process a video file or webcam frame by frame (detection or tracking mode)."""
    cap = open_capture(src)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(src.value).stem if src.kind == "video" else f"webcam{src.value}_{datetime.now():%Y%m%d_%H%M%S}"
    suffix = "track" if use_tracking else "pred"
    writer = None
    out_path: Path | None = None
    if save:
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}_{suffix}.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
        if not writer.isOpened():
            logger.warning("Could not create output video %s - continuing without saving.", out_path)
            writer, out_path = None, None

    detector.reset()
    frames = 0
    t_start = time.perf_counter()
    infer_total = 0.0
    fps_smooth = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if src.kind == "webcam" and frames == 0:
                    raise SourceError("Webcam opened but returned no frames.")
                break
            t0 = time.perf_counter()
            dets = detector.track_frame(frame) if use_tracking else detector.predict_frame(frame)
            infer_total += detector.last_inference_ms
            annotated = draw_detections(frame, dets, detector.cfg)
            dt = time.perf_counter() - t0
            fps_smooth = (1 / dt) if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1 / max(dt, 1e-6))
            total = detector.sorter.summary()["total_objects"]
            label = f"FPS {fps_smooth:4.1f} | {'unique objects' if use_tracking else 'detections'}: {total}"
            draw_overlay_text(annotated, label)
            if writer is not None:
                writer.write(annotated)
            frames += 1
            if show:
                try:
                    cv2.imshow("E-Waste Detection (press q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    logger.warning("No display available (headless OpenCV?) - disabling live preview.")
                    show = False
            if max_frames and frames >= max_frames:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = max(time.perf_counter() - t_start, 1e-9)
    return {"type": "track" if use_tracking else "video", "frames": frames,
            "avg_fps_end_to_end": round(frames / elapsed, 2),
            "avg_inference_ms": round(infer_total / frames, 2) if frames else None,
            "output_video": str(out_path) if out_path else None,
            "counting_mode": "unique_tracked_objects" if use_tracking else "per_detection_not_unique"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E-waste detection + sorting on images, video or webcam.")
    ap.add_argument("--source", required=True, help="Image / folder / video path, or webcam index (e.g. 0)")
    ap.add_argument("--model", default=None, help="Weights path (default: trained weights from config)")
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold (default from config)")
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-dir", default=None, help="Default: inference.output_dir from config")
    ap.add_argument("--show", action="store_true", help="Show a live window (always on for webcam)")
    ap.add_argument("--no-save", action="store_true", help="Do not write annotated video output")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        if args.conf is not None and not 0.0 <= args.conf <= 1.0:
            raise SourceError("--conf must be between 0 and 1.")
        src = parse_source(args.source)
        out_dir = Path(args.output_dir) if args.output_dir else cfg.resolve(cfg.get("inference.output_dir"))
        detector = Detector(cfg, args.model, args.conf, args.iou, args.device, args.imgsz)

        if src.kind in {"image", "directory"}:
            images = [src.value] if src.kind == "image" else list_images(src.value)
            result = predict_images(detector, images, out_dir)
            stem = Path(args.source).stem or "images"
        else:
            result = run_stream(detector, src, out_dir, False, args.show or src.kind == "webcam",
                                not args.no_save, args.max_frames)
            stem = f"{Path(str(args.source)).stem}" if src.kind == "video" else "webcam"
        result.update({"model": detector.weights, "device": detector.device, "conf": detector.conf,
                       "sorting_summary": detector.sorter.summary()})
        json_path = save_json(result, out_dir / f"{stem}_summary.json")
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    print(f"\nSorting summary: {result['sorting_summary']['by_category']}")
    print(f"Saved results to: {json_path}")
    print(cfg.get("project.disclaimer", "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
