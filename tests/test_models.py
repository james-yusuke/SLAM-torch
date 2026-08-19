import numpy as np
import pytest

from slam_torch.config import AppConfig
from slam_torch.devices import resolve_device
from slam_torch.models import build_models, model_status


def _weights_available() -> bool:
    status = model_status()
    depth = status["depth"]
    detector = status["detector"]
    return bool(
        isinstance(depth, dict)
        and isinstance(detector, dict)
        and depth.get("sha256_ok")
        and detector.get("sha256_ok")
    )


@pytest.mark.model
def test_real_models_run_offline_on_cpu() -> None:
    if not _weights_available():
        pytest.skip("Run `slam-torch models fetch` before the model smoke test")
    config = AppConfig()
    config.device.precision = "deterministic"
    runtime = resolve_device("cpu", precision="deterministic")
    depth, detector = build_models(config, runtime)
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    prediction = depth.predict(image)
    detections = detector.detect(image)
    assert prediction.shape == image.shape[:2]
    assert np.all(np.isfinite(prediction))
    assert isinstance(detections, list)
    assert depth.device == "cpu"
    assert detector.device == "cpu"


@pytest.mark.cuda
def test_real_models_stay_on_explicit_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is unavailable")
    if not _weights_available():
        pytest.skip("Run `slam-torch models fetch` before the CUDA smoke test")
    config = AppConfig()
    runtime = resolve_device("cuda", precision="deterministic")
    depth, detector = build_models(config, runtime)
    assert next(depth._model.parameters()).is_cuda
    assert next(detector._model.parameters()).is_cuda
    image = np.full((120, 160, 3), 127, dtype=np.uint8)
    prediction = depth.predict(image)
    detector.detect(image)
    assert np.all(np.isfinite(prediction))
    assert str(depth.last_input_device).startswith("cuda")
    assert str(depth.last_output_device).startswith("cuda")
    assert str(detector.last_input_device).startswith("cuda")
    assert str(detector.last_output_device).startswith("cuda")
