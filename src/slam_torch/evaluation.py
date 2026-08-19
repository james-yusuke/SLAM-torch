from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from slam_torch.types import PoseSE3


@dataclass(frozen=True, slots=True)
class TimedPose:
    timestamp: float
    pose: PoseSE3


def read_tum(path: str | Path) -> list[TimedPose]:
    result: list[TimedPose] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 8:
            raise ValueError(f"Invalid TUM trajectory row: {line}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = values
        result.append(
            TimedPose(
                timestamp,
                PoseSE3(
                    Rotation.from_quat([qx, qy, qz, qw]).as_matrix(),
                    np.asarray([tx, ty, tz]),
                ),
            )
        )
    return result


def associate_trajectories(
    estimated: list[TimedPose], ground_truth: list[TimedPose], max_difference: float = 0.02
) -> tuple[list[PoseSE3], list[PoseSE3]]:
    gt_times = np.asarray([item.timestamp for item in ground_truth])
    est_poses: list[PoseSE3] = []
    gt_poses: list[PoseSE3] = []
    if not len(gt_times):
        return est_poses, gt_poses
    for item in estimated:
        index = int(np.argmin(np.abs(gt_times - item.timestamp)))
        if abs(float(gt_times[index]) - item.timestamp) <= max_difference:
            est_poses.append(item.pose)
            gt_poses.append(ground_truth[index].pose)
    return est_poses, gt_poses


def align_similarity(
    source: NDArray[np.float64], target: NDArray[np.float64], with_scale: bool
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.sum(source_centered**2) / len(source))
    scale = float(np.sum(singular * sign) / variance) if with_scale and variance > 0 else 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def evaluate_trajectories(
    estimated: list[TimedPose], ground_truth: list[TimedPose]
) -> dict[str, float | int]:
    est, gt = associate_trajectories(estimated, ground_truth)
    if len(est) < 3:
        raise ValueError("At least three associated poses are required")
    est_xyz = np.asarray([pose.translation for pose in est])
    gt_xyz = np.asarray([pose.translation for pose in gt])
    scale, rotation, translation = align_similarity(est_xyz, gt_xyz, with_scale=True)
    aligned = (scale * (rotation @ est_xyz.T)).T + translation
    errors = np.linalg.norm(aligned - gt_xyz, axis=1)
    path_length = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))

    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    alignment_pose = PoseSE3(rotation, translation)
    aligned_poses = [
        PoseSE3(
            alignment_pose.rotation @ pose.rotation,
            scale * (rotation @ pose.translation) + translation,
        )
        for pose in est
    ]
    for index in range(len(gt) - 1):
        estimated_delta = aligned_poses[index].inverse() @ aligned_poses[index + 1]
        truth_delta = gt[index].inverse() @ gt[index + 1]
        error = truth_delta.inverse() @ estimated_delta
        rpe_translation.append(float(np.linalg.norm(error.translation)))
        rpe_rotation.append(float(np.degrees(Rotation.from_matrix(error.rotation).magnitude())))
    ate = float(np.sqrt(np.mean(errors**2)))
    return {
        "associated_poses": len(est),
        "sim3_scale": scale,
        "ate_sim3_rmse_m": ate,
        "ate_normalized_by_path": ate / max(path_length, 1e-9),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
        "ground_truth_path_length_m": path_length,
    }


def evaluate_run(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    estimated_path = root / "trajectory.tum"
    ground_truth_path = root / "groundtruth.tum"
    if not estimated_path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {estimated_path}")
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {ground_truth_path}")
    evaluation = evaluate_trajectories(read_tum(estimated_path), read_tum(ground_truth_path))
    metrics_path = root / "metrics.json"
    metrics: dict[str, object] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["trajectory_evaluation"] = evaluation
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics
