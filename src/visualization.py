"""Drawing and plotting utilities. Every plot is built from real data passed in - nothing is hard-coded."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit and CLI both render to figures/files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

logger = logging.getLogger(__name__)


def _finish(fig: plt.Figure, out: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if out:
        ensure_dir(Path(out).parent)
        fig.savefig(out, dpi=150)
        logger.info("Saved plot: %s", out)
    return fig


# ------------------------------------------------------------------ detections
def draw_detections(frame_bgr: np.ndarray, detections: list[dict[str, Any]], cfg: Config,
                    show_category: bool = True) -> np.ndarray:
    """Return a copy of the frame with boxes, class, confidence, [track id] and sorting category."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.45, min(h, w) / 1100)
    by_name = cfg.class_by_name
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        r, g, b = by_name[det["class_name"]].color
        color = (b, g, r)  # OpenCV is BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        prefix = f"#{det['track_id']} " if det.get("track_id") is not None else ""
        lines = [f"{prefix}{det['class_name']} {det['confidence']:.2f}"]
        if show_category:
            lines.append(det["category"])
        y_text = y1
        for line in lines:
            (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            top = max(0, y_text - th - base - 2)
            cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
            cv2.putText(out, line, (x1 + 2, top + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y_text = top
    return out


def draw_overlay_text(frame_bgr: np.ndarray, text: str, org: tuple[int, int] = (10, 25)) -> np.ndarray:
    """Draw a status line (FPS, counts) with a dark background for legibility."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame_bgr, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + base + 2), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


# ---------------------------------------------------------------------- charts
def plot_bar_counts(counts: dict[str, int], title: str, xlabel: str = "", out: str | Path | None = None,
                    colors: dict[str, tuple[int, int, int]] | None = None) -> plt.Figure:
    """Generic horizontal bar chart of {label: count}."""
    labels = list(counts)
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(labels) + 1)))
    bar_colors = [tuple(c / 255 for c in colors[k]) if colors and k in colors else None for k in labels]
    bars = ax.barh(labels, values, color=bar_colors if any(bar_colors) else None)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel or "count")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v}", va="center")
    ax.margins(x=0.12)
    return _finish(fig, out)


def plot_class_distribution(counts: dict[str, int], cfg: Config | None = None,
                            out: str | Path | None = None, title: str = "Class distribution") -> plt.Figure:
    colors = {c.name: c.color for c in cfg.classes} if cfg else None
    return plot_bar_counts(counts, title, "instances", out, colors)


def plot_category_distribution(counts: dict[str, int], out: str | Path | None = None,
                               title: str = "Sorting category distribution") -> plt.Figure:
    return plot_bar_counts(counts, title, "objects", out)


def plot_confusion_matrix(matrix: Iterable[Iterable[float]], labels: list[str],
                          out: str | Path | None = None, normalize: bool = False,
                          title: str = "Confusion matrix") -> plt.Figure:
    """Plot a confusion matrix (rows = predicted, columns = true, as produced by Ultralytics)."""
    m = np.asarray(matrix, dtype=float)
    if normalize:
        m = m / np.clip(m.sum(axis=0, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(labels), 1 + 0.8 * len(labels)))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title + (" (normalized)" if normalize else ""))
    thresh = m.max() / 2 if m.size else 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}" if normalize else f"{int(m[i, j])}", ha="center",
                    va="center", color="white" if m[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _finish(fig, out)


def read_results_csv(path: str | Path) -> pd.DataFrame:
    """Read an Ultralytics ``results.csv`` (column names are whitespace-padded)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Training results not found: {p}. Run `python src/train.py` first.")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_training_curves(results_csv: str | Path, out: str | Path | None = None) -> plt.Figure:
    """Loss and metric curves from Ultralytics' ``results.csv``."""
    df = read_results_csv(results_csv)
    loss_cols = [c for c in df.columns if c.startswith(("train/", "val/")) and c.endswith("loss")]
    metric_cols = [c for c in df.columns if c.startswith("metrics/")]
    epoch = df["epoch"] if "epoch" in df else np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in loss_cols:
        axes[0].plot(epoch, df[c], label=c)
    axes[0].set(title="Training / validation loss", xlabel="epoch", ylabel="loss")
    axes[0].legend(fontsize=7)
    for c in metric_cols:
        axes[1].plot(epoch, df[c], label=c.replace("metrics/", ""))
    axes[1].set(title="Validation metrics", xlabel="epoch", ylim=(0, 1.02))
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    return _finish(fig, out)


def plot_detection_statistics(detections: list[dict[str, Any]], out: str | Path | None = None) -> plt.Figure:
    """Confidence histogram and per-class confidence spread for a list of detections."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if not detections:
        for ax in axes:
            ax.text(0.5, 0.5, "no detections", ha="center", va="center")
            ax.axis("off")
        return _finish(fig, out)
    df = pd.DataFrame(detections)
    axes[0].hist(df["confidence"], bins=np.linspace(0, 1, 21), color="#457b9d")
    axes[0].set(title="Confidence distribution", xlabel="confidence", ylabel="detections")
    groups = [(n, g["confidence"].to_numpy()) for n, g in df.groupby("class_name")]
    axes[1].boxplot([g for _, g in groups], tick_labels=[n for n, _ in groups])
    axes[1].set(title="Confidence by class", ylabel="confidence", ylim=(0, 1.02))
    axes[1].tick_params(axis="x", rotation=30)
    return _finish(fig, out)
