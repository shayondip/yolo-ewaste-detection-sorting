# Methodology

## Task
Multi-class object detection of e-waste items in RGB images/video, followed by a rule-based
mapping from detected class to a simulated recycling destination.

## Model
A pretrained Ultralytics YOLO detection checkpoint (`model.weights` in the config) is
fine-tuned (transfer learning) on the e-waste dataset. The architecture is *not*
re-implemented. Optimizer, augmentation and loss follow Ultralytics defaults, except for the
values explicitly set in `configs/config.yaml` (image size, batch, epochs, `lr0`, `lrf`,
patience, workers, seed).

## Training protocol
1. Dataset audit (fails on structural errors).
2. Fine-tune on `train`, monitor `val` each epoch, keep `best.pt` (highest fitness =
   weighted mix of mAP50 and mAP50-95 as defined by Ultralytics) and `last.pt`.
3. Report final/best epoch metrics from `results.csv`.
4. **Final numbers come from the held-out `test` split** via `python src/evaluate.py`.
   Do not tune on `test`.

## Metrics
- **Precision / Recall** (at Ultralytics' optimal-F1 confidence).
- **mAP@50** and **mAP@50-95** (COCO-style, IoU 0.50 and 0.50:0.95).
- **Per-class** P/R/AP; **confusion matrix** (rows predicted, columns true, plus background).
- **Speed** via `scripts/benchmark.py` (CPU and GPU reported separately).

## Sorting logic
For every detection above the class-specific `min_confidence`, `EwasteSorter` looks up the
class's category and the category's destination and emits
`SEND_TO_<DESTINATION>`. Below-threshold detections are `NO_ACTION_LOW_CONFIDENCE`. All rules
live in the config. This is a *software simulation*: no actuator, safety, throughput or
material-purity considerations of a real sorting line are modelled.

## Counting
- Image: number of accepted detections.
- Video without tracking: **per-detection** counts (not unique objects).
- Video with tracking: an object ID is counted once when first accepted. ID switches, occlusion
  and re-entry can over- or under-count; counts are estimates.

## Threats to validity
Small or web-scraped datasets, class imbalance, near-duplicate frames across splits, single-
background photos, and label-definition mismatch between merged datasets all inflate
metrics relative to real deployment. Report these honestly.

## Future extensions (NOT implemented)
Class-imbalance handling (re-weighting/focal loss/oversampling), hard-negative mining,
lightweight YOLO variants for edge, domain adaptation and cross-dataset generalization
studies, uncertainty-aware detection, small-object detection (tiling/SAHI-style inference),
real conveyor-belt data with motion blur, synthetic-to-real transfer, INT8 quantization,
edge deployment (Jetson/Raspberry Pi/OpenVINO). The config-driven design (classes,
thresholds, dataset paths, model) and the shared `Detector` are meant to make such
experiments easy to add, but none of them exist yet.
