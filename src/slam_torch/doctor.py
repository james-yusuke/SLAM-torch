from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from slam_torch.assets import resolve_asset_root
from slam_torch.models import model_status


def _nvidia_driver() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _driver_diagnostics(driver: str | None, system: str) -> dict[str, object]:
    minimum = "528.33" if system == "Windows" else "525.60.13"
    recommended = "570.0"
    result: dict[str, object] = {
        "minimum_driver_for_cuda_12_x": minimum,
        "recommended_driver_for_cuda_12_8": recommended,
        "driver_compatible": None,
        "recommended_driver_met": None,
        "warnings": [],
    }
    if driver is None:
        return result
    compatible = _version_tuple(driver) >= _version_tuple(minimum)
    recommended_met = _version_tuple(driver) >= _version_tuple(recommended)
    result["driver_compatible"] = compatible
    result["recommended_driver_met"] = recommended_met
    if compatible and not recommended_met:
        warnings = result["warnings"]
        assert isinstance(warnings, list)
        warnings.append(
            f"Driver {driver} uses CUDA 12.x minor compatibility; a 570-series or newer "
            "driver is recommended for CUDA 12.8"
        )
    return result


def _required_device_status(
    require_device: str,
    report: dict[str, Any],
) -> dict[str, object]:
    errors: list[str] = []
    if require_device == "cpu":
        if not report.get("cpu_probe"):
            errors.append("CPU tensor probe failed")
    elif require_device == "mps":
        mps = report.get("mps", {})
        if not isinstance(mps, dict) or not mps.get("available"):
            errors.append("MPS is not available")
    elif require_device == "cuda":
        cuda = report.get("cuda", {})
        if not isinstance(cuda, dict):
            errors.append("CUDA diagnostics are unavailable")
        else:
            if cuda.get("torch_build") != "12.8":
                errors.append(
                    f"PyTorch CUDA build must be 12.8, got {cuda.get('torch_build')!r}"
                )
            if not cuda.get("available"):
                errors.append("torch.cuda.is_available() is false")
            if cuda.get("driver") is None:
                errors.append("NVIDIA driver version could not be determined with nvidia-smi")
            if cuda.get("driver_compatible") is False:
                errors.append("NVIDIA driver is below the CUDA 12.x compatibility minimum")
            devices = cuda.get("devices", [])
            if not isinstance(devices, list) or not devices:
                errors.append("No CUDA device completed the tensor probe")
            elif not all(isinstance(item, dict) and item.get("probe_ok") for item in devices):
                errors.append("One or more CUDA tensor probes failed")
    else:
        errors.append(f"Unsupported required device: {require_device}")
    return {"device": require_device, "satisfied": not errors, "errors": errors}


def _run_model_smoke(asset_root: Path, require_device: str | None) -> dict[str, object]:
    import numpy as np
    import torch

    from slam_torch.config import AppConfig
    from slam_torch.devices import resolve_device
    from slam_torch.models import build_models

    requested = require_device or "auto"
    runtime = resolve_device(requested, precision="balanced")
    config = AppConfig()
    config.assets.root = str(asset_root)
    config.device.requested = requested
    config.device.precision = "balanced"
    if runtime.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    depth, detector = build_models(config, runtime)
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    prediction = depth.predict(image)
    detections = detector.detect(image)
    depth_parameter_device = str(next(depth._model.parameters()).device)
    detector_parameter_device = str(next(detector._model.parameters()).device)
    expected_prefix = "cuda" if runtime.is_cuda else runtime.resolved
    detector_expected_prefix = "cpu" if runtime.is_mps else expected_prefix
    device_ok = (
        depth_parameter_device.startswith(expected_prefix)
        and detector_parameter_device.startswith(detector_expected_prefix)
        and str(depth.last_input_device).startswith(expected_prefix)
        and str(detector.last_input_device).startswith(detector_expected_prefix)
        and str(depth.last_output_device).startswith(expected_prefix)
        and str(detector.last_output_device).startswith(detector_expected_prefix)
    )
    result: dict[str, object] = {
        "ok": bool(np.all(np.isfinite(prediction))) and device_ok,
        "device": runtime.resolved,
        "depth_parameter_device": depth_parameter_device,
        "depth_input_device": depth.last_input_device,
        "depth_output_device": depth.last_output_device,
        "detector_parameter_device": detector_parameter_device,
        "detector_input_device": detector.last_input_device,
        "detector_output_device": detector.last_output_device,
        "depth_shape": list(prediction.shape),
        "detection_count": len(detections),
        "depth_inference_seconds": depth.last_inference_seconds,
        "detector_inference_seconds": detector.last_inference_seconds,
    }
    if runtime.is_cuda:
        result["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    return result


def doctor_report(
    asset_root: str | Path | None = None,
    *,
    require_device: str | None = None,
    model_smoke: bool = False,
) -> dict[str, Any]:
    resolved_root = resolve_asset_root(asset_root)
    report: dict[str, Any] = {
        "ok": True,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "asset_root": str(resolved_root),
        "models": model_status(resolved_root),
        "packages": {},
        "cuda": {"available": False},
        "mps": {"available": False},
    }
    for package in ("torch", "torchvision", "transformers", "opencv-python-headless", "open3d"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
            report["ok"] = False
    try:
        import torch

        cpu_probe = float((torch.ones((2, 2)) @ torch.ones((2, 2))).sum())
        report["cpu_probe"] = cpu_probe == 8.0
        report["mps"] = {
            "built": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built()),
            "available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        }
        driver = _nvidia_driver()
        cuda: dict[str, Any] = {
            "available": torch.cuda.is_available(),
            "torch_build": torch.version.cuda,
            "driver": driver,
            "devices": [],
        }
        cuda.update(_driver_diagnostics(driver, platform.system()))
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                probe = torch.ones((2, 2), device=f"cuda:{index}")
                cuda["devices"].append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                        "vram_bytes": properties.total_memory,
                        "probe_ok": float((probe @ probe).sum().cpu()) == 8.0,
                    }
                )
        report["cuda"] = cuda
    except Exception as exc:
        report["ok"] = False
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
    if require_device is not None:
        requirement = _required_device_status(require_device, report)
        report["requirement"] = requirement
        report["ok"] = bool(report["ok"] and requirement["satisfied"])
    if model_smoke:
        try:
            smoke = _run_model_smoke(resolved_root, require_device)
        except Exception as exc:
            smoke = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["model_smoke"] = smoke
        report["ok"] = bool(report["ok"] and smoke["ok"])
    return report
