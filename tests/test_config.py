"""Tests for configuration loading and validation (no dataset or GPU needed)."""

from __future__ import annotations

import copy

import pytest
import yaml

from src.config import Config, load_config
from src.errors import ConfigError, DatasetError
from src.utils import (build_data_yaml, check_data_yaml_matches_config, read_data_yaml,
                       write_data_yaml)


@pytest.fixture()
def cfg() -> Config:
    return load_config()


def test_default_config_loads(cfg: Config):
    assert cfg.num_classes == 6
    assert cfg.class_names == ["battery", "mobile_phone", "pcb", "cable", "plastic", "metal"]
    assert [c.id for c in cfg.classes] == list(range(6))


def test_every_class_category_is_defined(cfg: Config):
    for spec in cfg.classes:
        assert spec.category in cfg.categories


def test_get_and_resolve(cfg: Config):
    assert cfg.get("training.epochs") == 100
    assert cfg.get("does.not.exist", 7) == 7
    assert cfg.resolve("dataset").is_absolute()


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)


def test_empty_yaml_raises(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(empty)


def test_missing_section_raises(cfg: Config):
    raw = copy.deepcopy(cfg.raw)
    del raw["classes"]
    with pytest.raises(ConfigError, match="missing required section"):
        Config(raw)


def test_unknown_category_raises(cfg: Config):
    raw = copy.deepcopy(cfg.raw)
    raw["classes"][0]["category"] = "does_not_exist"
    with pytest.raises(ConfigError, match="unknown category"):
        Config(raw)


def test_duplicate_class_raises(cfg: Config):
    raw = copy.deepcopy(cfg.raw)
    raw["classes"].append(dict(raw["classes"][0]))
    with pytest.raises(ConfigError, match="Duplicate"):
        Config(raw)


def test_invalid_threshold_raises(cfg: Config):
    with pytest.raises(ConfigError):
        cfg.with_overrides(**{"inference.conf": 1.5})
    with pytest.raises(ConfigError):
        cfg.with_overrides(**{"training.batch": 0})


def test_overrides_ignore_none(cfg: Config):
    new = cfg.with_overrides(**{"training.epochs": 3, "training.batch": None})
    assert new.get("training.epochs") == 3
    assert new.get("training.batch") == cfg.get("training.batch")
    assert cfg.get("training.epochs") == 100  # original untouched


def test_adding_a_class_needs_only_config(cfg: Config):
    raw = copy.deepcopy(cfg.raw)
    raw["classes"].append({"name": "Circuit Breaker", "category": "electronic_component"})
    new = Config(raw)
    assert new.num_classes == 7
    assert new.class_names[-1] == "circuit_breaker"


def test_shipped_data_yaml_matches_config(cfg: Config):
    check_data_yaml_matches_config(cfg, read_data_yaml(cfg.data_yaml_path))


def test_data_yaml_mismatch_detected(cfg: Config):
    data = build_data_yaml(cfg)
    data["names"][0] = "wrong"
    with pytest.raises(DatasetError, match="do not match"):
        check_data_yaml_matches_config(cfg, data)


def test_write_data_yaml_roundtrip(cfg: Config, tmp_path):
    out = write_data_yaml(cfg, tmp_path / "data.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["nc"] == cfg.num_classes
    check_data_yaml_matches_config(cfg, data)
