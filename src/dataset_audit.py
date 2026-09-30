"""Dataset audit for YOLO-format datasets.

Checks: missing labels/images, invalid YOLO coordinates, invalid class IDs, corrupted
images, empty annotation files, duplicate filenames, class distribution/imbalance and
image-dimension statistics. Produces ``dataset_audit.json`` and a readable text report.

Usage:
    python src/dataset_audit.py
    python src/dataset_audit.py --dataset-root path/to/dataset --output-dir outputs/metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python src/dataset_audit.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config
from src.errors import ConfigError, DatasetError, EwasteError
from src.utils import IMAGE_EXTS, ensure_dir, save_json, setup_logging

logger = logging.getLogger("dataset_audit")

DEFAULT_SPLITS = ("train", "val", "test")
MAX_EXAMPLES = 20  # cap listed offending files per issue type to keep reports readable


def _parse_label_file(path: Path, num_classes: int) -> tuple[list[int], list[str], bool]:
    """Return (class_ids, problems, is_empty) for one YOLO label file."""
    problems: list[str] = []
    class_ids: list[int] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], [f"unreadable file: {exc}"], False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], [], True
    for n, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {n}: expected 5 values (class x y w h), found {len(parts)}")
            continue
        try:
            cid_f = float(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"line {n}: non-numeric value")
            continue
        if cid_f != int(cid_f) or cid_f < 0:
            problems.append(f"line {n}: invalid class id '{parts[0]}'")
            continue
        cid = int(cid_f)
        if cid >= num_classes:
            problems.append(f"line {n}: class id {cid} out of range (0-{num_classes - 1})")
            continue
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            problems.append(f"line {n}: coordinates outside [0, 1]")
            continue
        if w <= 0 or h <= 0:
            problems.append(f"line {n}: non-positive width/height")
            continue
        class_ids.append(cid)
    return class_ids, problems, False


def _image_info(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    """Return ((width, height), None) or (None, error message) using Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise DatasetError("Pillow is required for dataset auditing (`pip install Pillow`).") from exc
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() invalidates the handle; reopen for size
            return img.size, None
    except Exception as exc:  # noqa: BLE001 - any decode failure means corrupted
        return None, str(exc)


def audit_dataset(
    dataset_root: str | Path,
    class_names: list[str],
    splits: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Audit a YOLO dataset and return a JSON-serializable report.

    Args:
        dataset_root: Folder containing ``images/<split>`` and ``labels/<split>``.
        class_names: Ordered class names (index = class id).
        splits: Mapping split -> images sub-path (default ``images/<split>``).
    """
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}. See docs/dataset.md for setup.")
    splits = splits or {s: f"images/{s}" for s in DEFAULT_SPLITS}
    nc = len(class_names)

    report: dict[str, Any] = {
        "dataset_root": str(root), "class_names": class_names, "splits": {}, "issues": {},
    }
    issues: dict[str, list[str]] = defaultdict(list)
    total_class_counts: Counter[int] = Counter()
    all_stems: dict[str, list[str]] = defaultdict(list)  # stem -> splits it appears in
    widths: list[int] = []
    heights: list[int] = []
    sizes: Counter[tuple[int, int]] = Counter()

    for split, img_rel in splits.items():
        img_dir = root / img_rel
        lbl_dir = root / img_rel.replace("images", "labels", 1)
        images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) \
            if img_dir.is_dir() else []
        labels = sorted(p for p in lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
        if not img_dir.is_dir():
            issues["missing_split_folders"].append(f"{split}: {img_dir}")

        split_counts: Counter[int] = Counter()
        stems_seen: dict[str, list[str]] = defaultdict(list)
        image_stems = set()
        empty = missing_labels = 0

        for img in images:
            stems_seen[img.stem].append(img.name)
            image_stems.add(img.stem)
            all_stems[img.stem].append(split)
            size, err = _image_info(img)
            if err:
                issues["corrupted_images"].append(f"{split}/{img.name}: {err}")
            elif size:
                widths.append(size[0]); heights.append(size[1]); sizes[size] += 1

            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                missing_labels += 1
                issues["missing_labels"].append(f"{split}/{img.name}")
                continue
            ids, problems, is_empty = _parse_label_file(lbl, nc)
            if is_empty:
                empty += 1
                issues["empty_annotations"].append(f"{split}/{lbl.name}")
            for pr in problems:
                issues["invalid_annotations"].append(f"{split}/{lbl.name}: {pr}")
            split_counts.update(ids)

        for stem, names in stems_seen.items():
            if len(names) > 1:
                issues["duplicate_filenames_within_split"].append(f"{split}: {names}")

        orphan = [lbl for lbl in labels if lbl.stem not in image_stems]
        for lbl in orphan:
            issues["missing_images"].append(f"{split}/{lbl.name} (label without image)")

        total_class_counts.update(split_counts)
        report["splits"][split] = {
            "images": len(images),
            "label_files": len(labels),
            "instances": int(sum(split_counts.values())),
            "missing_labels": missing_labels,
            "orphan_labels": len(orphan),
            "empty_annotations": empty,
            "class_counts": {class_names[i]: int(split_counts.get(i, 0)) for i in range(nc)},
        }

    for stem, where in all_stems.items():
        if len(set(where)) > 1:
            issues["duplicate_filenames_across_splits"].append(
                f"{stem} appears in {sorted(set(where))} (possible train/test leakage)")

    counts = {class_names[i]: int(total_class_counts.get(i, 0)) for i in range(nc)}
    nonzero = [v for v in counts.values() if v > 0]
    report["class_distribution"] = {
        "counts": counts,
        "classes_without_instances": [n for n, v in counts.items() if v == 0],
        "imbalance_ratio_max_over_min": round(max(nonzero) / min(nonzero), 2) if nonzero else None,
    }
    report["totals"] = {
        "images": sum(s["images"] for s in report["splits"].values()),
        "instances": sum(counts.values()),
    }
    report["image_dimensions"] = _dimension_stats(widths, heights, sizes)
    report["issues"] = {
        k: {"count": len(v), "examples": v[:MAX_EXAMPLES]} for k, v in sorted(issues.items())
    }

    critical = ("corrupted_images", "invalid_annotations", "missing_images", "missing_split_folders",
                "duplicate_filenames_within_split")
    report["summary"] = {
        "errors": sum(len(issues.get(k, [])) for k in critical),
        "warnings": sum(len(v) for k, v in issues.items() if k not in critical),
        "passed": all(not issues.get(k) for k in critical),
    }
    return report


def _dimension_stats(widths: list[int], heights: list[int], sizes: Counter) -> dict[str, Any]:
    if not widths:
        return {"count": 0}
    n = len(widths)
    return {
        "count": n,
        "width": {"min": min(widths), "max": max(widths), "mean": round(sum(widths) / n, 1)},
        "height": {"min": min(heights), "max": max(heights), "mean": round(sum(heights) / n, 1)},
        "most_common_sizes": [{"size": f"{w}x{h}", "count": c} for (w, h), c in sizes.most_common(5)],
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the audit dict as a human-readable text report."""
    L: list[str] = ["=" * 64, "DATASET AUDIT REPORT", "=" * 64,
                    f"Dataset root : {report['dataset_root']}",
                    f"Classes      : {', '.join(report['class_names'])}", ""]
    t = report["totals"]
    L.append(f"Total images: {t['images']}   Total instances: {t['instances']}")
    L.append("")
    L.append(f"{'split':<8}{'images':>8}{'labels':>8}{'objects':>9}{'no-label':>10}{'empty':>7}")
    for name, s in report["splits"].items():
        L.append(f"{name:<8}{s['images']:>8}{s['label_files']:>8}{s['instances']:>9}"
                 f"{s['missing_labels']:>10}{s['empty_annotations']:>7}")
    L += ["", "Class distribution (all splits):"]
    mx = max(report["class_distribution"]["counts"].values() or [1]) or 1
    for n, c in report["class_distribution"]["counts"].items():
        L.append(f"  {n:<16}{c:>7}  {'#' * int(30 * c / mx)}")
    ratio = report["class_distribution"]["imbalance_ratio_max_over_min"]
    L.append(f"  imbalance ratio (max/min, non-zero classes): {ratio}")
    if report["class_distribution"]["classes_without_instances"]:
        L.append(f"  WARNING no instances for: {report['class_distribution']['classes_without_instances']}")
    d = report["image_dimensions"]
    L += ["", "Image dimensions:"]
    if d.get("count"):
        L.append(f"  width  min/mean/max: {d['width']['min']}/{d['width']['mean']}/{d['width']['max']}")
        L.append(f"  height min/mean/max: {d['height']['min']}/{d['height']['mean']}/{d['height']['max']}")
        L.append("  most common: " + ", ".join(f"{x['size']} (x{x['count']})" for x in d["most_common_sizes"]))
    else:
        L.append("  no readable images")
    L += ["", "Issues:"]
    if not report["issues"]:
        L.append("  none found")
    for k, v in report["issues"].items():
        L.append(f"  - {k}: {v['count']}")
        for ex in v["examples"][:5]:
            L.append(f"      {ex}")
        if v["count"] > 5:
            L.append(f"      ... ({v['count'] - 5} more; see JSON)")
    s = report["summary"]
    L += ["", f"RESULT: {'PASSED' if s['passed'] else 'FAILED'} "
              f"({s['errors']} error(s), {s['warnings']} warning(s))",
          "Note: images without a label file are treated as background by YOLO but are reported "
          "here so accidental omissions are visible.", "=" * 64]
    return "\n".join(L)


def run_audit(cfg: Config, dataset_root: Path | None = None, output_dir: Path | None = None,
              save: bool = True) -> dict[str, Any]:
    """Audit using the project config; optionally write JSON + text reports."""
    root = dataset_root or cfg.dataset_root
    report = audit_dataset(root, cfg.class_names, dict(cfg.get("dataset.splits", {})) or None)
    if save:
        out = ensure_dir(output_dir or cfg.resolve(cfg.get("evaluation.output_dir", "outputs/metrics")))
        save_json(report, out / "dataset_audit.json")
        (out / "dataset_audit.txt").write_text(format_report(report), encoding="utf-8")
        logger.info("Saved %s and %s", out / "dataset_audit.json", out / "dataset_audit.txt")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit a YOLO-format e-waste dataset.")
    ap.add_argument("--config", default=None, help="Path to config YAML (default configs/config.yaml)")
    ap.add_argument("--dataset-root", default=None, help="Override dataset root folder")
    ap.add_argument("--output-dir", default=None, help="Where to write dataset_audit.json/.txt")
    ap.add_argument("--fail-on-error", action="store_true", help="Exit code 1 if the audit fails")
    args = ap.parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config)
        report = run_audit(cfg, Path(args.dataset_root) if args.dataset_root else None,
                           Path(args.output_dir) if args.output_dir else None)
    except (EwasteError, ConfigError) as exc:
        logger.error("%s", exc)
        return 2
    print(format_report(report))
    return 1 if (args.fail_on_error and not report["summary"]["passed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
