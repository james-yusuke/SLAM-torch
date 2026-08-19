from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from slam_torch.assets import (
    _ensure_free_space,
    _install_stage,
    download_verified,
    resolve_asset_root,
    sha256_file,
)
from slam_torch.config import AppConfig
from slam_torch.devices import RuntimeDevice, inference_context, resolve_device
from slam_torch.types import Detection2D

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEPTH_MODEL_REVISION = "fd2c22027eaf20374204f14099b8341e1925ad39"
DEPTH_MODEL_SHA256 = "ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124"
DETECTOR_URL = (
    "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
)
DETECTOR_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"


def model_cache_root(asset_root: str | Path | None = None) -> Path:
    return resolve_asset_root(asset_root) / "models"


def depth_model_path(asset_root: str | Path | None = None) -> Path:
    return model_cache_root(asset_root) / "depth-anything-v2-metric-outdoor-small"


def detector_model_path(asset_root: str | Path | None = None) -> Path:
    return model_cache_root(asset_root) / "ssdlite320_mobilenet_v3_large_coco.pth"


def _depth_model_complete(asset_root: str | Path | None = None) -> bool:
    path = depth_model_path(asset_root)
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    return all((path / name).is_file() for name in required)


def model_status(asset_root: str | Path | None = None) -> dict[str, object]:
    depth_weights = depth_model_path(asset_root) / "model.safetensors"
    detector_weights = detector_model_path(asset_root)
    return {
        "cache": str(model_cache_root(asset_root)),
        "depth": {
            "available": _depth_model_complete(asset_root),
            "path": str(depth_model_path(asset_root)),
            "sha256": sha256_file(depth_weights) if depth_weights.is_file() else None,
            "sha256_ok": _depth_model_complete(asset_root)
            and sha256_file(depth_weights) == DEPTH_MODEL_SHA256,
        },
        "detector": {
            "available": detector_weights.is_file(),
            "path": str(detector_weights),
            "sha256": sha256_file(detector_weights) if detector_weights.is_file() else None,
            "sha256_ok": detector_weights.is_file()
            and sha256_file(detector_weights) == DETECTOR_SHA256,
        },
    }


def fetch_models(asset_root: str | Path | None = None) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    root = model_cache_root(asset_root)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_free_space(root, 256 * 1024**2)
    depth_dir = depth_model_path(asset_root)
    if not _depth_model_complete(asset_root) or sha256_file(
        depth_dir / "model.safetensors"
    ) != DEPTH_MODEL_SHA256:
        stage = Path(tempfile.mkdtemp(prefix=".depth-model-", dir=root))
        try:
            snapshot_download(
                repo_id=DEPTH_MODEL_ID,
                revision=DEPTH_MODEL_REVISION,
                local_dir=stage,
                cache_dir=resolve_asset_root(asset_root) / ".downloads" / "huggingface",
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
            )
            shutil.rmtree(stage / ".cache", ignore_errors=True)
            staged_weights = stage / "model.safetensors"
            actual_depth_hash = sha256_file(staged_weights)
            if actual_depth_hash != DEPTH_MODEL_SHA256:
                raise RuntimeError(
                    "Depth model checksum mismatch: "
                    f"expected {DEPTH_MODEL_SHA256}, got {actual_depth_hash}"
                )
            _install_stage(stage, depth_dir)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    depth_weights = depth_dir / "model.safetensors"
    actual_depth_hash = sha256_file(depth_weights)
    if actual_depth_hash != DEPTH_MODEL_SHA256:
        raise RuntimeError(
            f"Depth model checksum mismatch: expected {DEPTH_MODEL_SHA256}, got {actual_depth_hash}"
        )

    detector_path = detector_model_path(asset_root)
    if not detector_path.is_file() or sha256_file(detector_path) != DETECTOR_SHA256:
        download_verified(
            DETECTOR_URL,
            detector_path,
            size_bytes=14_069_355,
            sha256=DETECTOR_SHA256,
        )

    manifest: dict[str, object] = {
        "depth": {
            "model_id": DEPTH_MODEL_ID,
            "revision": DEPTH_MODEL_REVISION,
            "sha256": sha256_file(depth_weights),
            "license": "Apache-2.0",
        },
        "detector": {
            "url": DETECTOR_URL,
            "revision": "COCO_V1",
            "sha256": sha256_file(detector_path),
            "license": "BSD-3-Clause; COCO dataset terms apply",
        },
    }
    manifest_path = root / "manifest.json"
    manifest_part = manifest_path.with_name(manifest_path.name + ".part")
    manifest_part.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest_part.replace(manifest_path)
    return manifest


class DepthAnythingEstimator:
    def __init__(self, device: RuntimeDevice, asset_root: str | Path | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        path = depth_model_path(asset_root)
        weights = path / "model.safetensors"
        if not _depth_model_complete(asset_root) or sha256_file(weights) != DEPTH_MODEL_SHA256:
            raise FileNotFoundError(
                "Depth weights are missing or invalid; run `slam-torch models fetch`"
            )
        self._torch = torch
        self._runtime = device
        self.device = device.resolved
        self._processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            path, local_files_only=True
        )
        self._model = AutoModelForDepthEstimation.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).to(self.device)
        self._model.eval()
        self.last_inference_seconds = 0.0
        self.last_input_device: str | None = None
        self.last_output_device: str | None = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        self.last_input_device = str(next(iter(inputs.values())).device)
        with inference_context(self._runtime, allow_fp16=True):
            output = self._model(**inputs).predicted_depth
            self.last_output_device = str(output.device)
            resized = self._torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
        depth = resized[0].float().cpu().numpy().astype(np.float32)
        self.last_inference_seconds = time.perf_counter() - start
        return depth


class SSDLiteDetector:
    def __init__(
        self,
        device: RuntimeDevice,
        score_threshold: float,
        allowed: list[str],
        asset_root: str | Path | None = None,
    ) -> None:
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        path = detector_model_path(asset_root)
        if not path.is_file() or sha256_file(path) != DETECTOR_SHA256:
            raise FileNotFoundError(
                "Detector weights are missing or invalid; run `slam-torch models fetch`"
            )
        self._torch = torch
        self._runtime = device
        self.device = device.resolved
        self._threshold = score_threshold
        self._allowed = set(allowed)
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self._labels = list(weights.meta["categories"])
        self._model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=len(self._labels),
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.to(self.device).eval()
        self.last_inference_seconds = 0.0
        self.last_input_device: str | None = None
        self.last_output_device: str | None = None

    def detect(self, image: np.ndarray) -> list[Detection2D]:
        start = time.perf_counter()
        tensor = (
            self._torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(self.device)
        )
        self.last_input_device = str(tensor.device)
        with inference_context(self._runtime, allow_fp16=False):
            output = self._model([tensor])[0]
        self.last_output_device = str(output["boxes"].device)
        detections: list[Detection2D] = []
        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        for box, class_id, score in zip(boxes, labels, scores, strict=True):
            if float(score) < self._threshold:
                continue
            label = self._labels[int(class_id)]
            if self._allowed and label not in self._allowed:
                continue
            detections.append(
                Detection2D(
                    class_id=int(class_id),
                    label=label,
                    confidence=float(score),
                    box_xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
                )
            )
        self.last_inference_seconds = time.perf_counter() - start
        return detections


def build_models(config: AppConfig, device: RuntimeDevice) -> tuple[Any, Any]:
    if config.depth.model_id != DEPTH_MODEL_ID or config.depth.revision != DEPTH_MODEL_REVISION:
        raise ValueError(
            "This MVP supports only the pinned Depth Anything V2 outdoor-small checkpoint"
        )
    asset_root = resolve_asset_root(config_root=config.assets.root)
    depth = DepthAnythingEstimator(device, asset_root)
    detector_device = (
        resolve_device("cpu", precision=config.device.precision) if device.is_mps else device
    )
    detector = SSDLiteDetector(
        detector_device,
        config.detector.score_threshold,
        config.detector.allowed_classes,
        asset_root,
    )
    return depth, detector
