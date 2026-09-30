"""Sorting engine: maps detected classes to recycling categories and sorting decisions.

Everything is driven by ``configs/config.yaml`` - no class names are hard-coded here.
This is a software *simulation* of sorting decisions; it controls no hardware.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.config import Config, ClassSpec, normalize_name
from src.errors import UnknownClassError


class EwasteSorter:
    """Turns raw detections into structured sorting decisions and running statistics.

    Example:
        >>> sorter = EwasteSorter(load_config())
        >>> sorter.get_category("battery")
        'hazardous_electronic'
        >>> sorter.process_detection("battery", 0.91)["decision"]
        'SEND_TO_HAZARDOUS_ELECTRONIC_BIN'
    """

    def __init__(self, cfg: Config, conf_override: float | None = None):
        """
        Args:
            cfg: Loaded project configuration.
            conf_override: If given, replaces every per-class ``min_confidence``.
        """
        if conf_override is not None and not 0.0 <= conf_override <= 1.0:
            raise ValueError(f"conf_override must be within [0, 1], got {conf_override}")
        self.cfg = cfg
        self.conf_override = conf_override
        self._prefix = str(cfg.get("sorting.decision_prefix", "SEND_TO_"))
        self._rejected = str(cfg.get("sorting.rejected_decision", "NO_ACTION_LOW_CONFIDENCE"))
        self._by_name = cfg.class_by_name
        self._by_id = cfg.class_by_id
        self.reset()

    # ---------------------------------------------------------------- lookups
    def get_class(self, name_or_id: str | int) -> ClassSpec:
        """Return the ClassSpec for a class name or ID; raise UnknownClassError otherwise."""
        if isinstance(name_or_id, int) and not isinstance(name_or_id, bool):
            if name_or_id in self._by_id:
                return self._by_id[name_or_id]
            raise UnknownClassError(
                f"Invalid class ID {name_or_id}. Valid IDs: 0-{self.cfg.num_classes - 1}."
            )
        key = normalize_name(str(name_or_id))
        if key not in self._by_name:
            raise UnknownClassError(
                f"Unknown class '{name_or_id}'. Configured classes: {self.cfg.class_names}"
            )
        return self._by_name[key]

    def get_category(self, name_or_id: str | int) -> str:
        """Recycling category of a class, e.g. ``'hazardous_electronic'``."""
        return self.get_class(name_or_id).category

    def get_destination(self, name_or_id: str | int) -> str:
        """Simulated destination bin of a class, e.g. ``'ELECTRONIC_COMPONENT_BIN'``."""
        return str(self.cfg.categories[self.get_category(name_or_id)]["destination"])

    def get_decision(self, name_or_id: str | int) -> str:
        """Decision string, e.g. ``'SEND_TO_HAZARDOUS_ELECTRONIC_BIN'``."""
        return f"{self._prefix}{self.get_destination(name_or_id)}"

    def min_confidence(self, name_or_id: str | int) -> float:
        """Effective acceptance threshold for a class."""
        if self.conf_override is not None:
            return self.conf_override
        return self.get_class(name_or_id).min_confidence

    # ------------------------------------------------------------- processing
    def process_detection(
        self,
        class_name: str | int,
        confidence: float,
        bbox: list[float] | tuple[float, ...] | None = None,
        track_id: int | None = None,
    ) -> dict[str, Any]:
        """Convert one detection into a structured sorting decision and update counters.

        Detections below the class threshold are returned with ``accepted=False`` and are
        counted only in ``rejected_low_confidence``. When ``track_id`` is given, an object
        is counted once no matter how many frames it appears in.
        """
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence must be within [0, 1], got {confidence}")
        spec = self.get_class(class_name)
        threshold = self.min_confidence(spec.name)
        accepted = float(confidence) >= threshold

        result: dict[str, Any] = {
            "class_id": spec.id,
            "class_name": spec.name,
            "confidence": round(float(confidence), 4),
            "bbox": [round(float(v), 1) for v in bbox] if bbox is not None else None,
            "track_id": track_id,
            "category": spec.category,
            "destination": self.get_destination(spec.name) if accepted else None,
            "decision": self.get_decision(spec.name) if accepted else self._rejected,
            "accepted": accepted,
            "min_confidence": threshold,
            "already_counted": False,
        }

        if not accepted:
            self._rejected_count += 1
            return result

        if track_id is not None:
            if track_id in self._counted_tracks:
                result["already_counted"] = True
                return result
            self._counted_tracks.add(track_id)

        self._by_class[spec.name] += 1
        self._by_category[spec.category] += 1
        self._by_destination[result["destination"]] += 1
        return result

    def process_batch(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process dicts with keys ``class_name``, ``confidence`` and optional ``bbox``/``track_id``."""
        return [
            self.process_detection(d["class_name"], d["confidence"], d.get("bbox"), d.get("track_id"))
            for d in detections
        ]

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        """Structured counts. Zero-count classes/categories are included for stable charts."""
        by_class = {n: self._by_class.get(n, 0) for n in self.cfg.class_names}
        cats = {c: self._by_category.get(c, 0) for c in self.cfg.categories}
        dests = {str(v["destination"]): self._by_destination.get(str(v["destination"]), 0)
                 for v in self.cfg.categories.values()}
        return {
            "total_objects": int(sum(by_class.values())),
            "unique_tracks": len(self._counted_tracks),
            "by_class": by_class,
            "by_category": cats,
            "by_destination": dests,
            "rejected_low_confidence": self._rejected_count,
            "disclaimer": str(self.cfg.get("project.disclaimer", "")).strip(),
        }

    def reset(self) -> None:
        """Clear all counters (call between images/videos)."""
        self._by_class: Counter[str] = Counter()
        self._by_category: Counter[str] = Counter()
        self._by_destination: Counter[str] = Counter()
        self._counted_tracks: set[int] = set()
        self._rejected_count = 0
