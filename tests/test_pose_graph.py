import numpy as np
from scipy.spatial.transform import Rotation

from slam_torch.geometry import relative_pose
from slam_torch.pose_graph import PoseGraph, PoseGraphEdge
from slam_torch.types import PoseSE3


def test_loop_constraint_reduces_pose_graph_cost() -> None:
    true_poses = [
        PoseSE3(Rotation.from_euler("z", angle).as_matrix(), np.asarray(position, dtype=float))
        for position, angle in [
            ((0.0, 0.0, 0.0), 0.0),
            ((1.0, 0.0, 0.0), 0.0),
            ((1.0, 1.0, 0.0), 90.0),
            ((0.0, 1.0, 0.0), 180.0),
            ((0.0, 0.0, 0.0), -90.0),
        ]
    ]
    initial = [
        PoseSE3(pose.rotation, pose.translation + np.asarray([0.08 * index, 0.02 * index, 0.0]))
        for index, pose in enumerate(true_poses)
    ]
    graph = PoseGraph()
    for index in range(len(true_poses) - 1):
        graph.add_edge(
            PoseGraphEdge(index, index + 1, relative_pose(true_poses[index], true_poses[index + 1]))
        )
    graph.add_edge(PoseGraphEdge(0, 4, relative_pose(true_poses[0], true_poses[4]), 4.0, "loop"))
    before = graph.cost(initial)
    optimized = graph.optimize(initial)
    after = graph.cost(optimized)
    assert after < before
    assert np.linalg.norm(optimized[-1].translation) < np.linalg.norm(initial[-1].translation)
