"""Tests for the sorting engine: class mapping, thresholds, invalid classes, counting."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.errors import UnknownClassError
from src.sorting import EwasteSorter


@pytest.fixture()
def sorter() -> EwasteSorter:
    return EwasteSorter(load_config())


def test_category_mapping_covers_all_classes(sorter: EwasteSorter):
    expected = {
        "battery": "hazardous_electronic", "mobile_phone": "electronic_device",
        "pcb": "electronic_component", "cable": "recyclable_electronic",
        "plastic": "plastic_recycling", "metal": "metal_recycling",
    }
    for name, cat in expected.items():
        assert sorter.get_category(name) == cat


def test_get_by_id_and_normalized_name(sorter: EwasteSorter):
    assert sorter.get_category(0) == "hazardous_electronic"
    assert sorter.get_category("Mobile Phone") == "electronic_device"


def test_destination_and_decision(sorter: EwasteSorter):
    assert sorter.get_destination("pcb") == "ELECTRONIC_COMPONENT_BIN"
    assert sorter.get_decision("battery") == "SEND_TO_HAZARDOUS_ELECTRONIC_BIN"


def test_unknown_class_name_raises(sorter: EwasteSorter):
    with pytest.raises(UnknownClassError, match="Unknown class"):
        sorter.get_category("toaster")


def test_invalid_class_id_raises(sorter: EwasteSorter):
    with pytest.raises(UnknownClassError, match="Invalid class ID"):
        sorter.get_category(99)
    with pytest.raises(UnknownClassError):
        sorter.get_category(-1)


def test_process_detection_structure(sorter: EwasteSorter):
    d = sorter.process_detection("battery", 0.91, [1, 2, 30, 40])
    assert d["accepted"] is True
    assert d["class_name"] == "battery" and d["class_id"] == 0
    assert d["category"] == "hazardous_electronic"
    assert d["decision"] == "SEND_TO_HAZARDOUS_ELECTRONIC_BIN"
    assert d["bbox"] == [1.0, 2.0, 30.0, 40.0]


def test_per_class_threshold_rejects(sorter: EwasteSorter):
    # battery has min_confidence 0.40 in the shipped config, cable uses the default 0.35
    assert sorter.process_detection("battery", 0.39)["accepted"] is False
    assert sorter.process_detection("cable", 0.36)["accepted"] is True


def test_rejected_detection_not_counted(sorter: EwasteSorter):
    d = sorter.process_detection("battery", 0.10)
    assert d["decision"] == "NO_ACTION_LOW_CONFIDENCE" and d["destination"] is None
    s = sorter.summary()
    assert s["total_objects"] == 0 and s["rejected_low_confidence"] == 1


def test_confidence_override_replaces_class_thresholds():
    s = EwasteSorter(load_config(), conf_override=0.9)
    assert s.process_detection("cable", 0.8)["accepted"] is False
    assert s.process_detection("cable", 0.95)["accepted"] is True


def test_invalid_confidence_raises(sorter: EwasteSorter):
    with pytest.raises(ValueError):
        sorter.process_detection("battery", 1.7)
    with pytest.raises(ValueError):
        EwasteSorter(load_config(), conf_override=-0.1)


def test_summary_counts(sorter: EwasteSorter):
    sorter.process_batch([
        {"class_name": "battery", "confidence": 0.9},
        {"class_name": "battery", "confidence": 0.8},
        {"class_name": "pcb", "confidence": 0.7},
    ])
    s = sorter.summary()
    assert s["total_objects"] == 3
    assert s["by_class"]["battery"] == 2 and s["by_class"]["cable"] == 0
    assert s["by_category"]["hazardous_electronic"] == 2
    assert s["by_destination"]["ELECTRONIC_COMPONENT_BIN"] == 1
    assert set(s["by_class"]) == set(load_config().class_names)


def test_track_ids_are_counted_once(sorter: EwasteSorter):
    first = sorter.process_detection("cable", 0.8, track_id=7)
    again = sorter.process_detection("cable", 0.82, track_id=7)
    other = sorter.process_detection("cable", 0.8, track_id=8)
    assert not first["already_counted"] and again["already_counted"] and not other["already_counted"]
    s = sorter.summary()
    assert s["by_class"]["cable"] == 2 and s["unique_tracks"] == 2


def test_low_confidence_track_does_not_block_later_counting(sorter: EwasteSorter):
    sorter.process_detection("battery", 0.1, track_id=3)
    sorter.process_detection("battery", 0.9, track_id=3)
    assert sorter.summary()["by_class"]["battery"] == 1


def test_reset(sorter: EwasteSorter):
    sorter.process_detection("pcb", 0.9, track_id=1)
    sorter.reset()
    assert sorter.summary()["total_objects"] == 0
    assert sorter.process_detection("pcb", 0.9, track_id=1)["already_counted"] is False
