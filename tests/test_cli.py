import json
from pathlib import Path

import pytest
import yaml

import slam_torch.cli as cli
from slam_torch.config import AppConfig
from slam_torch.devices import DeviceUnavailableError, RuntimeDevice


def test_auto_cuda_validation_failure_uses_next_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested_devices: list[str] = []

    def resolve(requested: str, *, precision: str) -> RuntimeDevice:
        requested_devices.append(requested)
        if requested == "auto":
            return RuntimeDevice(requested, "cuda", precision)
        if requested == "mps":
            raise DeviceUnavailableError("MPS unavailable")
        return RuntimeDevice(requested, "cpu", precision)

    monkeypatch.setattr(cli, "resolve_device", resolve)
    monkeypatch.setattr(
        cli,
        "doctor_report",
        lambda *_args, **_kwargs: {
            "requirement": {"satisfied": False, "errors": ["driver too old"]}
        },
    )
    with pytest.warns(UserWarning, match="driver too old"):
        runtime = cli._resolve_runtime_device(
            "auto", precision="balanced", asset_root=tmp_path
        )
    assert runtime.resolved == "cpu"
    assert requested_devices == ["auto", "mps", "cpu"]


def test_explicit_cuda_failure_writes_failure_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig()
    config.dataset.path = str(tmp_path / "dataset")
    config.output.root = str(tmp_path / "runs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise DeviceUnavailableError("CUDA unavailable")

    monkeypatch.setattr(cli, "resolve_device", unavailable)
    with pytest.raises(DeviceUnavailableError, match="CUDA unavailable"):
        cli.main(["run", "--device", "cuda", "--config", str(config_path)])

    failures = list((tmp_path / "runs").glob("*/failure.json"))
    assert len(failures) == 1
    payload = json.loads(failures[0].read_text(encoding="utf-8"))
    assert payload["error_type"] == "DeviceUnavailableError"
    assert payload["runtime"]["requested"] == "cuda"
    assert payload["runtime"]["resolved"] is None


def test_cuda_oom_is_recorded_without_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AppConfig()
    config.assets.root = str(tmp_path)
    config.dataset.path = str(tmp_path / "dataset")
    config.output.root = str(tmp_path / "runs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    resolutions: list[str] = []

    def resolve(
        requested: str, *, precision: str, asset_root: Path
    ) -> RuntimeDevice:
        assert precision == "balanced"
        assert asset_root == tmp_path.resolve()
        resolutions.append(requested)
        return RuntimeDevice(requested, "cuda", precision)

    class Source:
        def validate(self) -> dict[str, object]:
            return {"valid": True}

    class OomSlam:
        def __init__(self, *_args: object) -> None:
            pass

        def run(self, _source: object) -> object:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cli, "_resolve_runtime_device", resolve)
    monkeypatch.setattr(
        cli,
        "device_metadata",
        lambda _device: {"requested": "cuda", "resolved": "cuda"},
    )
    monkeypatch.setattr(cli, "asset_runtime_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(cli, "create_source", lambda *_args, **_kwargs: Source())
    monkeypatch.setattr(cli, "build_models", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(cli, "SemanticSlam", OomSlam)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        cli.main(["run", "--device", "cuda", "--config", str(config_path)])

    assert resolutions == ["cuda"]
    failure = next((tmp_path / "runs").glob("*/failure.json"))
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["runtime"]["resolved"] == "cuda"
    assert payload["runtime"]["cuda_oom"] is True
