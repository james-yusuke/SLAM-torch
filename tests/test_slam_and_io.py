from collections.abc import Iterator
from pathlib import Path

import numpy as np

from slam_torch.config import AppConfig
from slam_torch.io import save_run
from slam_torch.slam import SemanticSlam
from slam_torch.types import CameraModel, Detection2D, Frame, PoseSE3


class FakeDepth:
    device = "cpu"
    last_inference_seconds = 0.001

    def predict(self, image: np.ndarray) -> np.ndarray:
        return np.full(image.shape[:2], 3.0, dtype=np.float32)


class FakeDetector:
    device = "cpu"
    last_inference_seconds = 0.001

    def detect(self, image: np.ndarray) -> list[Detection2D]:
        return [Detection2D(1, "chair", 0.9, (8.0, 8.0, 24.0, 24.0))]


class OneFrameSource:
    name = "synthetic"

    def __iter__(self) -> Iterator[Frame]:
        image = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
        yield Frame(
            0,
            0.0,
            image,
            CameraModel(64, 48, 50.0, 50.0, 32.0, 24.0),
            ground_truth_pose=PoseSE3.identity(),
            ground_truth_depth=np.full((48, 64), 3.0, dtype=np.float32),
        )

    def validate(self) -> dict[str, object]:
        return {"valid": True}


class LostAndRecoveredSource:
    name = "synthetic-relocalization"

    def __iter__(self) -> Iterator[Frame]:
        texture = np.random.default_rng(7).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        camera = CameraModel(160, 120, 120.0, 120.0, 80.0, 60.0)
        yield Frame(0, 0.0, texture, camera)
        yield Frame(1, 0.1, np.zeros_like(texture), camera)
        yield Frame(2, 0.2, texture.copy(), camera)

    def validate(self) -> dict[str, object]:
        return {"valid": True}


def test_single_frame_pipeline_writes_all_artifacts(tmp_path: Path) -> None:
    config = AppConfig()
    config.mapping.stride = 4
    result = SemanticSlam(config, FakeDepth(), FakeDetector()).run(OneFrameSource())
    assert result.metrics.tracking_rate == 1.0
    assert result.metrics.depth_abs_rel == 0.0
    assert len(result.state.points_world) > 0
    save_run(tmp_path, result, config, {"resolved": "cpu"})
    artifacts = (
        "trajectory.tum",
        "groundtruth.tum",
        "map.ply",
        "objects.json",
        "metrics.json",
        "run.yaml",
    )
    for name in artifacts:
        assert (tmp_path / name).is_file()


def test_depth_scale_factor_is_applied_before_mapping() -> None:
    config = AppConfig()
    config.depth.scale_factor = 0.5
    result = SemanticSlam(config, FakeDepth(), FakeDetector()).run(OneFrameSource())
    assert np.allclose(result.state.keyframes[0].depth, 1.5)


def test_lost_frame_can_relocalize_against_an_older_keyframe() -> None:
    config = AppConfig()
    config.tracking.min_matches = 12
    config.tracking.min_inliers = 8
    result = SemanticSlam(config, FakeDepth(), FakeDetector()).run(LostAndRecoveredSource())
    statuses = [entry.status.value for entry in result.state.trajectory]
    assert statuses == ["tracking", "lost", "tracking"]
    assert all(entry.pose is not None for entry in result.state.trajectory)
