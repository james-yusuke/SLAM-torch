from typing import Any

from slam_torch.doctor import _driver_diagnostics, _required_device_status


def test_cuda_driver_compatibility_and_recommendation_are_distinct() -> None:
    result = _driver_diagnostics("535.104.05", "Linux")
    assert result["driver_compatible"] is True
    assert result["recommended_driver_met"] is False
    assert result["warnings"]


def test_windows_driver_below_minimum_is_rejected() -> None:
    result = _driver_diagnostics("527.0", "Windows")
    assert result["driver_compatible"] is False


def test_required_cuda_reports_all_missing_capabilities() -> None:
    report: dict[str, Any] = {
        "cuda": {
            "available": False,
            "torch_build": None,
            "driver_compatible": None,
            "devices": [],
        }
    }
    result = _required_device_status("cuda", report)
    assert result["satisfied"] is False
    assert len(result["errors"]) >= 3  # type: ignore[arg-type]
