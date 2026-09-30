"""Tests for the dataset audit using a tiny synthetic dataset (no downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.dataset_audit import audit_dataset, format_report
from src.errors import DatasetError

NAMES = ["battery", "mobile_phone", "pcb"]


def _make_image(path: Path, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


def _make_label(root: Path, split: str, stem: str, content: str) -> None:
    p = root / "labels" / split / f"{stem}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _issue(report, key):
    return report["issues"].get(key, {"count": 0, "examples": []})


def test_clean_dataset_passes(tmp_path):
    for i, split in enumerate(["train", "val", "test"]):
        _make_image(tmp_path / "images" / split / f"img{i}.jpg")
        _make_label(tmp_path, split, f"img{i}", "0 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1\n")
    report = audit_dataset(tmp_path, NAMES)
    assert report["summary"]["passed"] is True
    assert report["totals"] == {"images": 3, "instances": 6}
    assert report["class_distribution"]["counts"] == {"battery": 3, "mobile_phone": 0, "pcb": 3}
    assert report["class_distribution"]["classes_without_instances"] == ["mobile_phone"]
    assert "PASSED" in format_report(report)


def test_missing_dataset_folder_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        audit_dataset(tmp_path / "nope", NAMES)


def test_missing_label_and_missing_image(tmp_path):
    _make_image(tmp_path / "images" / "train" / "a.jpg")           # no label
    _make_label(tmp_path, "train", "ghost", "0 0.5 0.5 0.2 0.2")    # no image
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train"})
    assert _issue(report, "missing_labels")["count"] == 1
    assert _issue(report, "missing_images")["count"] == 1
    assert report["summary"]["passed"] is False


def test_invalid_annotations_detected(tmp_path):
    bad_lines = "\n".join([
        "9 0.5 0.5 0.2 0.2",      # class id out of range
        "0 1.5 0.5 0.2 0.2",      # coordinate > 1
        "0 0.5 0.5 0.0 0.2",      # zero width
        "0 0.5 0.5 0.2",          # too few values
        "x 0.5 0.5 0.2 0.2",      # non-numeric class
        "0.5 0.5 0.5 0.2 0.2",    # fractional class id
        "0 0.5 0.5 0.2 0.2",      # valid
    ])
    _make_image(tmp_path / "images" / "train" / "a.png")
    _make_label(tmp_path, "train", "a", bad_lines)
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train"})
    assert _issue(report, "invalid_annotations")["count"] == 6
    assert report["class_distribution"]["counts"]["battery"] == 1


def test_empty_annotation_is_warning_not_error(tmp_path):
    _make_image(tmp_path / "images" / "train" / "bg.jpg")
    _make_label(tmp_path, "train", "bg", "\n")
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train"})
    assert _issue(report, "empty_annotations")["count"] == 1
    assert report["summary"]["passed"] is True and report["summary"]["warnings"] == 1


def test_corrupted_image_detected(tmp_path):
    bad = tmp_path / "images" / "train" / "broken.jpg"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not an image")
    _make_label(tmp_path, "train", "broken", "0 0.5 0.5 0.2 0.2")
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train"})
    assert _issue(report, "corrupted_images")["count"] == 1
    assert report["summary"]["passed"] is False


def test_duplicate_filenames(tmp_path):
    _make_image(tmp_path / "images" / "train" / "same.jpg")
    _make_image(tmp_path / "images" / "train" / "same.png")            # same stem, same split
    _make_image(tmp_path / "images" / "val" / "same.jpg")              # same stem across splits
    for split in ("train", "val"):
        _make_label(tmp_path, split, "same", "0 0.5 0.5 0.2 0.2")
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train", "val": "images/val"})
    assert _issue(report, "duplicate_filenames_within_split")["count"] == 1
    assert _issue(report, "duplicate_filenames_across_splits")["count"] == 1


def test_dimension_stats_and_imbalance(tmp_path):
    _make_image(tmp_path / "images" / "train" / "a.jpg", (100, 50))
    _make_image(tmp_path / "images" / "train" / "b.jpg", (200, 150))
    _make_label(tmp_path, "train", "a", "0 0.5 0.5 0.2 0.2\n0 0.4 0.4 0.1 0.1\n0 0.3 0.3 0.1 0.1\n")
    _make_label(tmp_path, "train", "b", "2 0.5 0.5 0.2 0.2")
    report = audit_dataset(tmp_path, NAMES, {"train": "images/train"})
    dims = report["image_dimensions"]
    assert dims["width"]["min"] == 100 and dims["width"]["max"] == 200
    assert report["class_distribution"]["imbalance_ratio_max_over_min"] == 3.0
