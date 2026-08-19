from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from slam_torch.geometry import pose_to_vector, relative_pose, vector_to_pose
from slam_torch.types import PoseSE3


@dataclass(frozen=True, slots=True)
class PoseGraphEdge:
    source: int
    target: int
    measurement_source_target: PoseSE3
    weight: float = 1.0
    kind: str = "odometry"


class PoseGraph:
    def __init__(self) -> None:
        self.edges: list[PoseGraphEdge] = []

    def add_edge(self, edge: PoseGraphEdge) -> None:
        if edge.source == edge.target:
            raise ValueError("A pose graph edge cannot connect a node to itself")
        self.edges.append(edge)

    def cost(self, poses: list[PoseSE3]) -> float:
        residual = self._residuals(poses)
        return float(residual @ residual)

    def optimize(self, poses: list[PoseSE3], max_iterations: int = 100) -> list[PoseSE3]:
        if len(poses) < 2 or not self.edges:
            return list(poses)
        fixed = poses[0]
        initial = np.concatenate([pose_to_vector(pose) for pose in poses[1:]])

        def unpack(vector: NDArray[np.float64]) -> list[PoseSE3]:
            result = [fixed]
            result.extend(vector_to_pose(chunk) for chunk in vector.reshape(-1, 6))
            return result

        valid_edges = [
            edge
            for edge in self.edges
            if edge.source < len(poses) and edge.target < len(poses)
        ]
        if not valid_edges:
            return list(poses)
        sparsity = lil_matrix((6 * len(valid_edges), 6 * (len(poses) - 1)), dtype=np.int8)
        for edge_index, edge in enumerate(valid_edges):
            rows = slice(6 * edge_index, 6 * (edge_index + 1))
            if edge.source > 0:
                columns = slice(6 * (edge.source - 1), 6 * edge.source)
                sparsity[rows, columns] = 1
            if edge.target > 0:
                columns = slice(6 * (edge.target - 1), 6 * edge.target)
                sparsity[rows, columns] = 1

        solution = least_squares(
            lambda vector: self._residuals(unpack(vector)),
            initial,
            loss="huber",
            f_scale=1.0,
            jac_sparsity=sparsity.tocsr(),
            max_nfev=max_iterations,
        )
        return unpack(solution.x)

    def _residuals(self, poses: list[PoseSE3]) -> NDArray[np.float64]:
        values: list[NDArray[np.float64]] = []
        for edge in self.edges:
            if edge.source >= len(poses) or edge.target >= len(poses):
                continue
            predicted = relative_pose(poses[edge.source], poses[edge.target])
            error = edge.measurement_source_target.inverse() @ predicted
            scale = np.sqrt(max(edge.weight, 1e-8))
            translation = error.translation * scale
            rotation = Rotation.from_matrix(error.rotation).as_rotvec() * scale
            values.append(np.concatenate((translation, rotation)))
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)
