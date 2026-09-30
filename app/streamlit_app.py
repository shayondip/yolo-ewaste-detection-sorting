"""Streamlit dashboard for the E-Waste Detection and Sorting prototype.

Run:  streamlit run app/streamlit_app.py

Uses exactly the same `Detector` / `EwasteSorter` code as the CLI tools.
Live webcam: browsers cannot stream frames to Streamlit without extra components, so this
dashboard offers webcam *snapshots* (st.camera_input). For real-time webcam use
`python src/predict.py --source 0` or `python src/track.py --source 0`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.errors import EwasteError
from src.inference import Detector
from src.utils import load_json, resolve_weights
from src.visualization import (draw_detections, plot_category_distribution, plot_class_distribution,
                               plot_confusion_matrix, plot_detection_statistics, plot_training_curves)

st.set_page_config(page_title="E-Waste Detection & Sorting", page_icon="♻️", layout="wide")


@st.cache_resource(show_spinner="Loading configuration...")
def get_config():
    return load_config()


@st.cache_resource(show_spinner="Loading model...")
def get_detector(weights: str, device: str) -> Detector:
    return Detector(get_config(), weights=weights or None, device=device)


def discover_models(cfg) -> list[str]:
    """Trained weights from config, any run under outputs/train, and models/*.pt."""
    found: list[Path] = [cfg.resolve(cfg.get("model.trained_weights"))]
    found += sorted(cfg.resolve(cfg.get("training.project_dir", "outputs/train")).glob("*/weights/*.pt"))
    found += sorted(cfg.resolve("models").glob("*.pt")) if cfg.resolve("models").exists() else []
    options = [str(p) for p in dict.fromkeys(found) if p.exists()]
    return options + [str(cfg.get("model.weights"))]  # base weights last (with warning)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detections_table(dets: list[dict]) -> pd.DataFrame:
    if not dets:
        return pd.DataFrame(columns=["class", "confidence", "category", "decision", "bbox"])
    return pd.DataFrame([{"track_id": d.get("track_id"), "class": d["class_name"], "confidence": d["confidence"],
                          "category": d["category"], "decision": d["decision"],
                          "bbox": [round(v) for v in d["bbox"]]} for d in dets])


def process_image(detector: Detector, frame: np.ndarray) -> dict:
    detector.reset()
    dets = detector.predict_frame(frame)
    return {"preview": bgr_to_rgb(draw_detections(frame, dets, detector.cfg)), "detections": dets,
            "summary": detector.sorter.summary(), "fps": 1000 / max(detector.last_inference_ms, 1e-6),
            "frames": 1, "mode": "image"}


def process_video(detector: Detector, path: str, use_tracking: bool, stride: int, max_frames: int) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise EwasteError("Could not open the uploaded video (unsupported codec or corrupted file).")
    detector.reset()
    holder, bar, info = st.empty(), st.progress(0.0), st.empty()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    all_dets: list[dict] = []
    preview, frames, t0 = None, 0, time.perf_counter()
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if (frames - 1) % stride:
            continue
        dets = detector.track_frame(frame) if use_tracking else detector.predict_frame(frame)
        all_dets.extend(dets)
        preview = bgr_to_rgb(draw_detections(frame, dets, detector.cfg))
        holder.image(preview, use_container_width=True)
        bar.progress(min(frames / max(min(total, max_frames), 1), 1.0))
        info.caption(f"Frame {frames} | {frames / (time.perf_counter() - t0):.1f} FPS (end-to-end)")
    cap.release()
    holder.empty(); bar.empty(); info.empty()
    if frames == 0:
        raise EwasteError("The video contains no readable frames.")
    return {"preview": preview, "detections": all_dets, "summary": detector.sorter.summary(),
            "fps": frames / (time.perf_counter() - t0), "frames": frames,
            "mode": "video (unique tracked objects)" if use_tracking else "video (per-detection counts)"}


def render_results(res: dict, cfg) -> None:
    summary = res["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects counted", summary["total_objects"])
    c2.metric("Frames processed", res["frames"])
    c3.metric("FPS", f"{res['fps']:.1f}")
    c4.metric("Below threshold (rejected)", summary["rejected_low_confidence"])

    tabs = st.tabs(["Detection Preview", "Detected Objects", "Sorting Summary", "Class Distribution", "Model Performance"])
    with tabs[0]:
        if res["preview"] is not None:
            st.image(res["preview"], caption=f"Mode: {res['mode']}", use_container_width=True)
        else:
            st.info("No frame to display.")
    with tabs[1]:
        df = detections_table(res["detections"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        if len(df):
            st.pyplot(plot_detection_statistics(res["detections"]))
    with tabs[2]:
        rows = [{"category": c, "destination": cfg.categories[c]["destination"], "count": n}
                for c, n in summary["by_category"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.pyplot(plot_category_distribution(summary["by_category"]))
        if summary["total_objects"]:
            st.download_button("Download summary (JSON)", json.dumps(summary, indent=2), "sorting_summary.json")
    with tabs[3]:
        st.pyplot(plot_class_distribution(summary["by_class"], cfg))
    with tabs[4]:
        render_model_performance(cfg)
    st.caption(cfg.get("project.disclaimer", "").strip())


def render_model_performance(cfg) -> None:
    metrics_path = cfg.resolve(cfg.get("evaluation.output_dir")) / "metrics.json"
    if not metrics_path.exists():
        st.info("No evaluation results yet. Run `python src/evaluate.py` after training.")
    else:
        m = load_json(metrics_path)
        st.caption(f"Split: {m['meta']['split']} | weights: {m['meta']['weights']}")
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Precision", "precision"), ("Recall", "recall"),
                                            ("mAP@50", "mAP50"), ("mAP@50-95", "mAP50-95")]):
            col.metric(label, f"{m[key]:.3f}")
        pc = pd.DataFrame([{"class": k, **(v or {})} for k, v in m["per_class"].items()])
        st.dataframe(pc, use_container_width=True, hide_index=True)
        if m.get("confusion_matrix"):
            cm = m["confusion_matrix"]
            st.pyplot(plot_confusion_matrix(cm["matrix"], cm["labels"], normalize=True))
    csv = cfg.resolve(cfg.get("training.project_dir")) / cfg.get("training.run_name") / "results.csv"
    if csv.exists():
        st.pyplot(plot_training_curves(csv))


def main() -> None:
    try:
        cfg = get_config()
    except EwasteError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    st.title("♻️ YOLO-Based E-Waste Detection and Automatic Sorting")
    st.warning(cfg.get("project.disclaimer", "").strip())

    with st.sidebar:
        st.header("Settings")
        models = discover_models(cfg)
        choice = st.selectbox("Model", models + ["Custom path..."])
        weights = st.text_input("Custom weights path") if choice == "Custom path..." else choice
        device = st.selectbox("Device", ["auto", "cpu", "0"])
        conf = st.slider("Confidence threshold", 0.05, 0.95, float(cfg.get("inference.conf", 0.25)), 0.05,
                         help="Applied as YOLO floor AND as the sorting acceptance threshold for all classes.")
        mode = st.radio("Input", ["Image", "Video", "Webcam snapshot"])
        if mode == "Video":
            use_tracking = st.checkbox("Track objects (unique counting)", value=True)
            stride = st.number_input("Process every Nth frame", 1, 30, 1)
            max_frames = st.number_input("Max frames", 10, 5000, 300)

    try:
        detector = get_detector(weights, device)
    except EwasteError as exc:
        st.error(str(exc))
        st.stop()
    if not detector.is_finetuned and Path(detector.weights).name == Path(str(cfg.get("model.weights"))).name:
        st.warning("Using base (COCO-pretrained) weights - they cannot detect this project's e-waste classes. "
                   "Train a model first: `python src/train.py`.")
    detector.conf = conf
    detector.sorter.conf_override = conf

    try:
        res = None
        if mode == "Image":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if up:
                frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise EwasteError("The uploaded file is not a valid image.")
                res = process_image(detector, frame)
        elif mode == "Video":
            up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
            if up and st.button("Run detection", type="primary"):
                with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                    tmp.write(up.read())
                try:
                    res = process_video(detector, tmp.name, use_tracking, int(stride), int(max_frames))
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
        else:
            snap = st.camera_input("Take a picture")
            if snap:
                frame = cv2.imdecode(np.frombuffer(snap.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                res = process_image(detector, frame)
        if res:
            render_results(res, cfg)
        else:
            st.info("Provide an input to see detections.")
    except EwasteError as exc:
        st.error(str(exc))


main()
