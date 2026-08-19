from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from slam_torch.config import AppConfig
from slam_torch.slam import SlamResult
from slam_torch.types import ObjectLandmark3D, PoseSE3


def create_run_directory(root: str | Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def pose_tum_line(timestamp: float, pose: PoseSE3) -> str:
    qx, qy, qz, qw = Rotation.from_matrix(pose.rotation).as_quat()
    tx, ty, tz = pose.translation
    return (
        f"{timestamp:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
        f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
    )


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and colors must both have shape (N, 3)")
    with Path(path).open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _object_dict(item: ObjectLandmark3D) -> dict[str, Any]:
    return {
        "id": item.id,
        "label": item.label,
        "confidence": item.confidence,
        "center_world": item.center_world.tolist(),
        "aabb_min_world": item.aabb_min_world.tolist(),
        "aabb_max_world": item.aabb_max_world.tolist(),
        "observation_count": item.observation_count,
        "first_seen": item.first_seen,
        "last_seen": item.last_seen,
        "dynamic": item.dynamic,
    }


def save_run(
    run_dir: str | Path,
    result: SlamResult,
    config: AppConfig,
    runtime: dict[str, object],
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    estimated = [
        pose_tum_line(entry.timestamp, entry.pose)
        for entry in result.state.trajectory
        if entry.pose is not None
    ]
    (path / "trajectory.tum").write_text("\n".join(estimated) + "\n", encoding="utf-8")
    ground_truth = [
        pose_tum_line(entry.timestamp, entry.ground_truth_pose)
        for entry in result.state.trajectory
        if entry.ground_truth_pose is not None
    ]
    if ground_truth:
        (path / "groundtruth.tum").write_text(
            "\n".join(ground_truth) + "\n", encoding="utf-8"
        )
    write_ply(path / "map.ply", result.state.points_world, result.state.colors_rgb)
    (path / "objects.json").write_text(
        json.dumps(
            [_object_dict(item) for item in result.state.objects],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = result.metrics.to_dict()
    metrics["runtime"] = runtime
    (path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    resolved = config.to_dict()
    resolved["runtime"] = runtime
    (path / "run.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_failure(run_dir: str | Path, exc: BaseException, runtime: dict[str, object]) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "runtime": runtime,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    (path / "failure.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
