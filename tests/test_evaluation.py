import numpy as np

from slam_torch.evaluation import TimedPose, evaluate_trajectories
from slam_torch.types import PoseSE3


def test_sim3_evaluation_removes_global_scale_and_translation() -> None:
    ground_truth = [
        TimedPose(float(index), PoseSE3(np.eye(3), np.asarray([float(index), 0.0, 0.0])))
        for index in range(5)
    ]
    estimated = [
        TimedPose(float(index), PoseSE3(np.eye(3), np.asarray([2.0 * index + 5.0, 3.0, 0.0])))
        for index in range(5)
    ]
    metrics = evaluate_trajectories(estimated, ground_truth)
    assert metrics["ate_sim3_rmse_m"] < 1e-8
    assert abs(float(metrics["sim3_scale"]) - 0.5) < 1e-8
