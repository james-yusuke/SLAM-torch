from types import SimpleNamespace

import pytest
import yaml

import slam_torch.devices as devices
from slam_torch.devices import (
    DeviceUnavailableError,
    RuntimeDevice,
    device_metadata,
    is_cuda_oom,
    resolve_device,
)


class FakeCuda:
    def __init__(self, available: bool, count: int = 1) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def fake_torch(cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(cuda=FakeCuda(cuda), backends=SimpleNamespace(mps=FakeMps(mps)))


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_probe", lambda _torch, _name: None)
    monkeypatch.setattr(devices, "configure_reproducibility", lambda _torch, _precision: None)
    result = resolve_device("auto", torch_module=fake_torch(cuda=True, mps=True))
    assert result.resolved == "cuda"


def test_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_probe", lambda _torch, _name: None)
    monkeypatch.setattr(devices, "configure_reproducibility", lambda _torch, _precision: None)
    with pytest.warns(UserWarning):
        result = resolve_device("auto", torch_module=fake_torch(cuda=False, mps=True))
    assert result.resolved == "mps"


def test_explicit_cuda_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "_probe", lambda _torch, _name: None)
    with pytest.raises(DeviceUnavailableError):
        resolve_device("cuda", torch_module=fake_torch(cuda=False, mps=True))


def test_cuda_oom_string_is_detected() -> None:
    assert is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not is_cuda_oom(RuntimeError("different error"))


def test_device_metadata_is_yaml_serializable() -> None:
    metadata = device_metadata(RuntimeDevice("cpu", "cpu", "balanced"))
    assert type(metadata["torch"]) is str
    yaml.safe_dump(metadata)
