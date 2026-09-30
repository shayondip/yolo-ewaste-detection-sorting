# Dataset guide

**No dataset is included or downloaded automatically.** Download one yourself, check its
license, then import it with the provided scripts. Dataset folders are git-ignored.

## Required layout (YOLO format)

```
dataset/
├── data.yaml
├── images/{train,val,test}/  *.jpg|png ...
└── labels/{train,val,test}/  *.txt     # one line per object: class x_center y_center width height (normalized 0-1)
```

Class IDs follow the order in `configs/config.yaml` (`0 battery, 1 mobile_phone, 2 pcb, 3 cable, 4 plastic, 5 metal`).

## Candidate public datasets

Found by searching public dataset hubs; **verify the license, class list, image count and
annotation quality on the dataset page yourself before use** - the notes below come from the
search-result pages and were not independently checked by opening the data.

| Dataset | Source | License shown on page | Classes shown | Notes |
|---|---|---|---|---|
| E-Waste Classifications (student-esos5) | Roboflow Universe: `universe.roboflow.com/student-esos5/e-waste-classifications` | CC BY 4.0 | Battery, Bulb, Cable, PCB | Covers 3 of our classes (battery, cable, pcb) |
| E WASTE FINAL (battery-i4e70) | `universe.roboflow.com/battery-i4e70/e-waste-final` | CC BY 4.0 | Phone, battery, cable | Covers battery, mobile_phone, cable; page lists a large image count that may include augmentation - check |
| Object detection e waste (hela-abfom) | `universe.roboflow.com/hela-abfom/object-detection-e-waste-zmpnp` | CC BY 4.0 | Mouse, Keyboard, Battery, Mobile, PC | battery + mobile useful; others dropped |
| E-waste detection (jensen) | `universe.roboflow.com/jensen/e-waste-detection-g7vf3` | CC BY 4.0 | PCB, LCD, Battery (per preview images) | pcb + battery |
| Ewaste (object-detection-djqlx) | `universe.roboflow.com/object-detection-djqlx/ewaste-w4jyk` | CC BY 4.0 | usb, adapter, earbud, battery (per preview images) | battery only useful |
| PCB Object Detection Dataset | Hugging Face: `Arshia82sbn/pcb-object-detection-dataset` | Not visible in the excerpt - **check the dataset card** | 21 PCB *component* classes | Component-level; only useful if you want to redefine `pcb` |
| Electronic waste object detection system | `universe.roboflow.com/vision-experiments-fprmb/electronic-waste-object-detection-system` | Not visible in the excerpt - **check the page** | UNU-KEY based device classes (laptops, phones, ...) | Device-level e-waste; different taxonomy |

**Coverage gap:** none of the datasets above obviously provides `plastic` or `metal` scrap
classes. Options: (a) start with the four covered classes and keep `plastic`/`metal` in the
config as future classes (they will simply have zero instances - the audit flags this),
(b) find/annotate your own data for them, (c) remove them from `configs/config.yaml`.
Do not claim performance on classes with no training data.

Combining datasets from different sources can introduce **domain shift** and **label
inconsistency** (what counts as "cable"?). Inspect samples manually and keep attribution
(CC BY 4.0 requires it) in your own documentation.

## How to download (Roboflow Universe)

1. Open the dataset page, sign in (free account), click **Download Dataset**.
2. Choose format **YOLO** (e.g. "YOLOv8"/"YOLO11"), download the zip (no API key needed), unzip
   to `downloads/<name>/`. You should see `train/ valid/ test/` (each with `images/` `labels/`)
   and a `data.yaml`.

## Import with class remapping

```bash
python scripts/prepare_dataset.py init
# Edit configs/class_mapping.example.yaml (copy it) so source class names map to ours or null to drop.
python scripts/prepare_dataset.py convert-yolo --source downloads/e-waste-final \
    --mapping configs/class_mapping.example.yaml --prefix ewf
python scripts/prepare_dataset.py audit
```

Repeat for further datasets with a different `--prefix` (prevents filename collisions).
If a download only has a train split, add `--resplit-from-train 0.8 0.1 0.1`. Beware: random
splits of video-frame datasets leak near-duplicates between splits.

COCO-JSON sources: `python scripts/prepare_dataset.py coco-to-yolo --coco-json ann.json --images-dir imgs --mapping map.yaml --split train`.
Other formats (Pascal VOC XML, etc.): use Roboflow's export to YOLO, or convert to COCO first.

## Audit

`python src/dataset_audit.py` writes `outputs/metrics/dataset_audit.json` and
`dataset_audit.txt` and checks: missing labels, missing images (orphan labels), invalid YOLO
values / class IDs, corrupted images, empty annotation files, duplicate filenames (within
and across splits), per-class counts and imbalance, image size statistics.
`src/train.py` runs the audit automatically and stops on errors.
