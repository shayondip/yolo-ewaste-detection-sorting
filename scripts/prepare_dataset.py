"""Dataset preparation helpers.

Sub-commands:
  init          Create the dataset/ folder skeleton and dataset/data.yaml from configs/config.yaml
  write-yaml    (Re)generate dataset/data.yaml from configs/config.yaml
  convert-yolo  Import an external YOLO-format dataset (e.g. a Roboflow "YOLO" export), remapping
                its classes onto this project's classes via a mapping YAML, into dataset/
  coco-to-yolo  Convert a COCO-JSON dataset into YOLO labels with the same class mapping
  audit         Shortcut for `python src/dataset_audit.py`

Examples:
  python scripts/prepare_dataset.py init
  python scripts/prepare_dataset.py convert-yolo --source downloads/ewaste_v1 \
      --mapping configs/class_mapping.example.yaml
  python scripts/prepare_dataset.py coco-to-yolo --coco-json ann.json --images-dir imgs/ \
      --mapping configs/class_mapping.example.yaml --split train
  python scripts/prepare_dataset.py audit

Datasets themselves are never committed (see .gitignore) - see docs/dataset.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.config import Config, load_config, normalize_name
from src.dataset_audit import format_report, run_audit
from src.errors import ConfigError, DatasetError, EwasteError
from src.utils import IMAGE_EXTS, ensure_dir, list_images, setup_logging, write_data_yaml

logger = logging.getLogger("prepare_dataset")
SOURCE_SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}


# ------------------------------------------------------------------- mapping
def load_mapping(path: Path, cfg: Config) -> dict[str, str | None]:
    """Load source->target class mapping and validate targets against the config."""
    if not path.exists():
        raise ConfigError(f"Class mapping file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Class mapping must be a YAML mapping of source_name: target_name|null.")
    mapping: dict[str, str | None] = {}
    for src, dst in raw.items():
        if dst is None:
            mapping[normalize_name(src)] = None
            continue
        dst_n = normalize_name(dst)
        if dst_n not in cfg.class_by_name:
            raise ConfigError(f"Mapping target '{dst}' (from '{src}') is not a class in configs/config.yaml: "
                              f"{cfg.class_names}")
        mapping[normalize_name(src)] = dst_n
    return mapping


def build_id_remap(source_names: list[str], mapping: dict[str, str | None], cfg: Config) -> dict[int, int | None]:
    """source class id -> target class id (None = drop)."""
    remap: dict[int, int | None] = {}
    for sid, name in enumerate(source_names):
        target = mapping.get(normalize_name(name))
        if normalize_name(name) not in mapping:
            logger.warning("Source class '%s' is not in the mapping - its boxes will be dropped.", name)
        remap[sid] = cfg.class_by_name[target].id if target else None
    return remap


def remap_label_text(text: str, remap: dict[int, int | None]) -> tuple[str, int, int]:
    """Rewrite YOLO label lines. Returns (new_text, kept, dropped)."""
    out, kept, dropped = [], 0, 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            if parts:
                dropped += 1
            continue
        try:
            new_id = remap.get(int(float(parts[0])))
        except ValueError:
            dropped += 1
            continue
        if new_id is None:
            dropped += 1
        else:
            out.append(" ".join([str(new_id)] + parts[1:]))
            kept += 1
    return ("\n".join(out) + ("\n" if out else "")), kept, dropped


# ---------------------------------------------------------------- convert-yolo
def _find_split_dirs(source: Path) -> dict[str, tuple[Path, Path]]:
    """Detect Roboflow-style (<split>/images, <split>/labels) or (images/<split>, labels/<split>) layouts."""
    found: dict[str, tuple[Path, Path]] = {}
    for target, aliases in SOURCE_SPLIT_ALIASES.items():
        for a in aliases:
            for img, lbl in ((source / a / "images", source / a / "labels"),
                             (source / "images" / a, source / "labels" / a)):
                if img.is_dir():
                    found[target] = (img, lbl)
                    break
            if target in found:
                break
    return found


def _copy_pair(img: Path, lbl_dir: Path, remap: dict[int, int | None], dest_img: Path, dest_lbl: Path,
               prefix: str) -> tuple[int, int]:
    name = f"{prefix}{img.stem}"
    shutil.copy2(img, dest_img / f"{name}{img.suffix.lower()}")
    src_lbl = lbl_dir / f"{img.stem}.txt"
    text = src_lbl.read_text(encoding="utf-8") if src_lbl.exists() else ""
    new_text, kept, dropped = remap_label_text(text, remap)
    (dest_lbl / f"{name}.txt").write_text(new_text, encoding="utf-8")
    return kept, dropped


def cmd_convert_yolo(args: argparse.Namespace, cfg: Config) -> None:
    source = Path(args.source)
    if not source.is_dir():
        raise DatasetError(f"Source dataset folder not found: {source}")
    src_yaml = Path(args.source_yaml) if args.source_yaml else source / "data.yaml"
    if not src_yaml.exists():
        raise DatasetError(f"Source data.yaml not found at {src_yaml}. Pass --source-yaml.")
    names = yaml.safe_load(src_yaml.read_text(encoding="utf-8")).get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    if not names:
        raise DatasetError(f"No 'names' found in {src_yaml}.")
    remap = build_id_remap(names, load_mapping(Path(args.mapping), cfg), cfg)

    splits = _find_split_dirs(source)
    if not splits:
        raise DatasetError("Could not find <split>/images folders in the source. Expected a Roboflow-style "
                           "export (train/valid/test each with images/ and labels/).")
    prefix = f"{args.prefix}_" if args.prefix else ""
    total_kept = total_dropped = 0
    for split, (img_dir, lbl_dir) in splits.items():
        dest_img = ensure_dir(cfg.split_dir(split, "images"))
        dest_lbl = ensure_dir(cfg.split_dir(split, "labels"))
        images = list_images(img_dir)
        for img in images:
            k, d = _copy_pair(img, lbl_dir, remap, dest_img, dest_lbl, prefix)
            total_kept += k
            total_dropped += d
        logger.info("%s: copied %d images", split, len(images))
    logger.info("Boxes kept: %d, dropped (unmapped): %d", total_kept, total_dropped)
    logger.info("Done. Next: python scripts/prepare_dataset.py audit")
    if set(splits) != {"train", "val", "test"}:
        logger.warning("Missing splits in source: %s. Create them with --resplit-from-train or supply another export.",
                       sorted({"train", "val", "test"} - set(splits)))
    if args.resplit_from_train:
        _resplit(cfg, args.resplit_from_train, args.seed)


def _resplit(cfg: Config, ratios: list[float], seed: int) -> None:
    """Randomly redistribute the images currently in dataset/train into train/val/test."""
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError("--resplit-from-train needs three ratios summing to 1, e.g. 0.8 0.1 0.1")
    images = list_images(cfg.split_dir("train", "images"))
    if not images:
        raise DatasetError("No images in dataset/images/train to split.")
    rng = random.Random(seed)
    rng.shuffle(images)
    n = len(images)
    n_val, n_test = int(n * ratios[1]), int(n * ratios[2])
    buckets = {"val": images[:n_val], "test": images[n_val:n_val + n_test]}
    for split, items in buckets.items():
        di, dl = ensure_dir(cfg.split_dir(split, "images")), ensure_dir(cfg.split_dir(split, "labels"))
        for img in items:
            lbl = cfg.split_dir("train", "labels") / f"{img.stem}.txt"
            shutil.move(str(img), di / img.name)
            if lbl.exists():
                shutil.move(str(lbl), dl / lbl.name)
    logger.info("Resplit %d images -> train %d / val %d / test %d", n, n - n_val - n_test, n_val, n_test)
    logger.warning("Random resplitting can leak near-duplicate frames across splits; prefer source-provided splits.")


# ---------------------------------------------------------------- coco-to-yolo
def cmd_coco_to_yolo(args: argparse.Namespace, cfg: Config) -> None:
    coco_path, images_dir = Path(args.coco_json), Path(args.images_dir)
    if not coco_path.exists():
        raise DatasetError(f"COCO JSON not found: {coco_path}")
    if not images_dir.is_dir():
        raise DatasetError(f"Images folder not found: {images_dir}")
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid COCO JSON: {exc}") from exc

    mapping = load_mapping(Path(args.mapping), cfg)
    cats = {c["id"]: normalize_name(c["name"]) for c in coco.get("categories", [])}
    cat_to_target: dict[int, int | None] = {}
    for cid, name in cats.items():
        target = mapping.get(name)
        if name not in mapping:
            logger.warning("COCO category '%s' is not in the mapping - dropped.", name)
        cat_to_target[cid] = cfg.class_by_name[target].id if target else None

    imgs = {im["id"]: im for im in coco.get("images", [])}
    lines: dict[int, list[str]] = {i: [] for i in imgs}
    skipped = 0
    for ann in coco.get("annotations", []):
        im, tgt = imgs.get(ann["image_id"]), cat_to_target.get(ann["category_id"])
        if im is None or tgt is None or ann.get("iscrowd", 0):
            skipped += 1
            continue
        x, y, w, h = ann["bbox"]
        W, H = im["width"], im["height"]
        xc, yc, wn, hn = (x + w / 2) / W, (y + h / 2) / H, w / W, h / H
        if wn <= 0 or hn <= 0:
            skipped += 1
            continue
        clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
        lines[im["id"]].append(f"{tgt} {clip(xc):.6f} {clip(yc):.6f} {clip(wn):.6f} {clip(hn):.6f}")

    dest_img = ensure_dir(cfg.split_dir(args.split, "images"))
    dest_lbl = ensure_dir(cfg.split_dir(args.split, "labels"))
    copied = 0
    for iid, im in imgs.items():
        src_img = images_dir / Path(im["file_name"]).name
        if not src_img.exists():
            logger.warning("Image listed in COCO but missing on disk: %s", src_img)
            continue
        shutil.copy2(src_img, dest_img / src_img.name)
        body = "\n".join(lines[iid]) + ("\n" if lines[iid] else "")
        (dest_lbl / f"{src_img.stem}.txt").write_text(body, encoding="utf-8")
        copied += 1
    logger.info("Converted %d images to split '%s' (%d annotations skipped/dropped).", copied, args.split, skipped)


# ------------------------------------------------------------------- others
def cmd_init(cfg: Config) -> None:
    for kind in ("images", "labels"):
        for split in cfg.get("dataset.splits", {}):
            d = ensure_dir(cfg.split_dir(split, kind))
            (d / ".gitkeep").touch()
    logger.info("Created dataset skeleton under %s", cfg.dataset_root)
    logger.info("Wrote %s", write_data_yaml(cfg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="Create dataset skeleton + data.yaml")
    sub.add_parser("write-yaml", help="Regenerate dataset/data.yaml from the config")
    sub.add_parser("audit", help="Run the dataset audit")

    c = sub.add_parser("convert-yolo", help="Import an external YOLO-format dataset with class remapping")
    c.add_argument("--source", required=True, help="Folder of the downloaded dataset")
    c.add_argument("--source-yaml", default=None, help="Its data.yaml (default <source>/data.yaml)")
    c.add_argument("--mapping", required=True, help="Class mapping YAML (see configs/class_mapping.example.yaml)")
    c.add_argument("--prefix", default="", help="Filename prefix to avoid collisions when merging datasets")
    c.add_argument("--resplit-from-train", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=None,
                   help="Randomly split imported train images into train/val/test, e.g. 0.8 0.1 0.1")
    c.add_argument("--seed", type=int, default=42)

    k = sub.add_parser("coco-to-yolo", help="Convert COCO JSON annotations to YOLO labels")
    k.add_argument("--coco-json", required=True)
    k.add_argument("--images-dir", required=True)
    k.add_argument("--mapping", required=True)
    k.add_argument("--split", default="train", choices=["train", "val", "test"])

    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        if args.cmd == "init":
            cmd_init(cfg)
        elif args.cmd == "write-yaml":
            logger.info("Wrote %s", write_data_yaml(cfg))
        elif args.cmd == "audit":
            print(format_report(run_audit(cfg)))
        elif args.cmd == "convert-yolo":
            cmd_convert_yolo(args, cfg)
        elif args.cmd == "coco-to-yolo":
            cmd_coco_to_yolo(args, cfg)
    except EwasteError as exc:
        logger.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
