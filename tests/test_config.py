from pathlib import Path

import pytest

from slam_torch.config import AppConfig, load_config


def test_defaults_are_valid() -> None:
    config = AppConfig()
    config.validate()
    assert config.dataset.type == "tartanair"
    assert config.device.requested == "auto"
    assert config.assets.root is None


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("dataset:\n  type: tartanair\n  typo: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(path)


def test_nested_config_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "valid.yaml"
    path.write_text(
        "assets:\n  root: /tmp/slam-assets\n"
        "dataset:\n  type: euroc\n  path: data/euroc/MH_01_easy-300\n  max_frames: 20\n"
        "device:\n  requested: cpu\n  precision: deterministic\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dataset.type == "euroc"
    assert config.dataset.max_frames == 20
    assert config.dataset.path == "data/euroc/MH_01_easy-300"
    assert config.assets.root == "/tmp/slam-assets"
    assert config.device.precision == "deterministic"


def test_non_positive_depth_scale_is_rejected() -> None:
    config = AppConfig()
    config.depth.scale_factor = 0.0
    with pytest.raises(ValueError, match="scale_factor"):
        config.validate()
