"""Shared detection core used by predict.py, track.py, evaluate.py, benchmark and the Streamlit app.

One `Detector` wraps an Ultralytics model plus the `EwasteSorter`, so every entry point
produces identical, structured detections.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.config import Config
from src.errors import UnknownClassError
from src.sorting import EwasteSorter
from src.utils import load_yolo, resolve_weights, select_device

logger = logging.getLogger(__name__)


class Detector:
    """YOLO detector + sorting engine.

    Detections are dicts with: class_id, class_name, confidence, bbox [x1,y1,x2,y2],
    track_id, category, destination, decision, accepted, min_confidence, model_class_id.
    Only detections that pass the class confidence threshold are returned.
    """

    def __init__(
        self,
        cfg: Config,
        weights: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        device: str | None = None,
        imgsz: int | None = None,
        sort_conf_override: float | None = None,
    ):
        inf = cfg.section("inference")
        self.cfg = cfg
        self.conf = float(conf if conf is not None else inf.get("conf", 0.25))
        self.iou = float(iou if iou is not None else inf.get("iou", 0.45))
        self.imgsz = int(imgsz if imgsz is not None else inf.get("imgsz", 640))
        self.max_det = int(inf.get("max_det", 100))
        self.tracker = str(cfg.get("tracking.tracker", "bytetrack.yaml"))
        self.device = select_device(device or inf.get("device", "auto"))
        self.weights, self.is_finetuned = resolve_weights(cfg, weights)
        self.model = load_yolo(self.weights)
        self.sorter = EwasteSorter(cfg, conf_override=sort_conf_override)
        self.last_inference_ms: float = 0.0
        self._warned_classes: set[str] = set()
        logger.info("Loaded model: %s", self.weights)

    # ------------------------------------------------------------------ core
    def predict_frame(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Run detection on one BGR frame (numpy array)."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
            max_det=self.max_det, verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        return self._parse(results[0])

    def track_frame(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Run detection + multi-object tracking on one frame (keeps tracker state)."""
        t0 = time.perf_counter()
        results = self.model.track(
            frame_bgr, persist=True, tracker=self.tracker, conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.device, max_det=self.max_det, verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        return self._parse(results[0], require_id=True)

    def reset(self) -> None:
        """Clear sorting counters and tracker state (call between images/videos)."""
        self.sorter.reset()
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            if hasattr(tracker, "reset"):
                tracker.reset()  # restarts track IDs so a new video begins at #1

    # --------------------------------------------------------------- parsing
    def _parse(self, result: Any, require_id: bool = False) -> list[dict[str, Any]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        if require_id and getattr(boxes, "id", None) is None:
            return []  # tracker has not confirmed any track yet; never count un-tracked boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if getattr(boxes, "id", None) is not None else None
        names = result.names

        detections: list[dict[str, Any]] = []
        for i in range(len(xyxy)):
            model_cid = int(clss[i])
            name = str(names.get(model_cid, model_cid)) if isinstance(names, dict) else str(names[model_cid])
            track_id = int(ids[i]) if ids is not None else None
            try:
                det = self.sorter.process_detection(name, float(confs[i]), xyxy[i].tolist(), track_id)
            except UnknownClassError:
                if name not in self._warned_classes:
                    self._warned_classes.add(name)
                    logger.warning(
                        "Model predicts class '%s' which is not in configs/config.yaml - ignored. "
                        "Are you using COCO-pretrained weights instead of your trained model?", name)
                continue
            det["model_class_id"] = model_cid
            if det["accepted"]:
                detections.append(det)
        return detections
