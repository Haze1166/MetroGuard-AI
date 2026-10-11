"""
Basic sanity tests for project skeleton and configuration.
"""
from pathlib import Path
import yaml
import pytest


def test_project_structure():
    root = Path(__file__).resolve().parent.parent
    assert (root / "config" / "config.yaml").exists()
    assert (root / "requirements.txt").exists()
    assert (root / ".gitignore").exists()
    assert (root / "README.md").exists()
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "models").is_dir()
    assert (root / "reports" / "figures").is_dir()
    assert (root / "data" / "raw").is_dir()


def test_config_loading():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["project"]["name"] == "MetroGuard AI"
    assert config["project"]["random_seed"] == 42
    assert "analog" in config["sensors"]
    assert "digital" in config["sensors"]
    assert len(config["sensors"]["analog"]) == 7
    assert len(config["sensors"]["digital"]) == 8
    assert config["labeling"]["horizon_hours"] == 2.0


def test_raw_data_exists():
    raw_csv = Path(__file__).resolve().parent.parent / "data" / "raw" / "MetroPT3(AirCompressor).csv"
    assert raw_csv.exists(), "Raw dataset MetroPT3(AirCompressor).csv must exist in data/raw/"
    assert raw_csv.stat().st_size > 100_000_000, "Raw CSV must be the full MetroPT-3 dataset (>100MB)"
