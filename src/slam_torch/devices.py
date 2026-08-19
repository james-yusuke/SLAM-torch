from __future__ import annotations

import contextlib
import importlib.metadata
import os
import platform
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from slam_torch.types import DeviceSpec


class DeviceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeDevice:
    requested: str
    resolved: str
    precision: str

    @property
    def is_cuda(self) -> bool:
        return self.resolved.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        return self.resolved == "mps"


def _probe(torch: Any, name: str) -> None:
    value = torch.ones((2, 2), device=name)
    result = (value @ value).sum()
    if float(result.cpu()) != 8.0:
        raise RuntimeError(f"Device probe returned an invalid result on {name}")


def resolve_device(
    spec: DeviceSpec | str,
    *,
    precision: str = "balanced",
    torch_module: Any | None = None,
) -> RuntimeDevice:
    if precision not in {"balanced", "deterministic"}:
        raise ValueError(f"Unsupported precision profile: {precision}")
    requested = DeviceSpec(spec if isinstance(spec, str) else spec.requested).requested
    if torch_module is None:
        import torch as imported_torch

        torch: Any = imported_torch
    else:
        torch = torch_module
    candidates = [requested]
    if requested == "auto":
        candidates = ["cuda", "mps", "cpu"]

    errors: list[str] = []
    for candidate in candidates:
        try:
            if candidate.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise DeviceUnavailableError("torch.cuda.is_available() is false")
                if ":" in candidate:
                    index = int(candidate.split(":", 1)[1])
                    if index >= torch.cuda.device_count():
                        raise DeviceUnavailableError(
                            f"CUDA device {index} does not exist; count={torch.cuda.device_count()}"
                        )
                _probe(torch, candidate)
            elif candidate == "mps":
                if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                    raise DeviceUnavailableError("MPS is not available")
                _probe(torch, candidate)
            else:
                _probe(torch, "cpu")
            configure_reproducibility(torch, precision)
            return RuntimeDevice(requested=requested, resolved=candidate, precision=precision)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            if requested != "auto":
                raise DeviceUnavailableError("; ".join(errors)) from exc
            warnings.warn(
                f"Device {candidate} is unavailable; trying the next backend: {exc}",
                stacklevel=2,
            )
    raise DeviceUnavailableError("No usable PyTorch device: " + "; ".join(errors))


def configure_reproducibility(torch: Any, precision: str) -> None:
    torch.manual_seed(0)
    deterministic = precision == "deterministic"
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest" if deterministic else "high")
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = not deterministic
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = not deterministic


@contextlib.contextmanager
def inference_context(device: RuntimeDevice, *, allow_fp16: bool) -> Iterator[None]:
    import torch

    use_amp = device.is_cuda and device.precision == "balanced" and allow_fp16
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield


def is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
    except ImportError:
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def enable_mps_fallback() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device_metadata(device: RuntimeDevice) -> dict[str, object]:
    import torch

    result: dict[str, object] = {
        "requested": device.requested,
        "resolved": device.resolved,
        "precision": device.precision,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_build": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "torchvision",
                "transformers",
                "opencv-python-headless",
                "numpy",
                "scipy",
            )
        },
    }
    if device.is_cuda:
        index = torch.cuda.current_device() if device.resolved == "cuda" else int(
            device.resolved.split(":", 1)[1]
        )
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "gpu_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_vram_bytes": properties.total_memory,
            }
        )
    return result
