from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from slam_torch.types import CameraModel, PoseSE3


@dataclass(frozen=True, slots=True)
class DescriptorMatches:
    query_indices: NDArray[np.int32]
    train_indices: NDArray[np.int32]
    distances: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.query_indices)


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    pose: PoseSE3
    inlier_count: int
    match_count: int

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_count / max(self.match_count, 1)


def extract_orb(
    image_rgb: NDArray[np.uint8], max_features: int
) -> tuple[NDArray[np.float32], NDArray[np.uint8] | None]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=max_features, fastThreshold=12
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32), None
    points = np.asarray([point.pt for point in keypoints], dtype=np.float32)
    return points, descriptors


def match_descriptors(
    query: NDArray[np.uint8] | None,
    train: NDArray[np.uint8] | None,
    ratio: float,
) -> DescriptorMatches:
    if query is None or train is None or len(query) < 2 or len(train) < 2:
        empty_i = np.empty(0, dtype=np.int32)
        return DescriptorMatches(empty_i, empty_i.copy(), np.empty(0, dtype=np.float32))
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(query, train, k=2)
    accepted = [first for first, second in pairs if first.distance < ratio * second.distance]
    accepted.sort(key=lambda item: item.distance)
    return DescriptorMatches(
        np.asarray([item.queryIdx for item in accepted], dtype=np.int32),
        np.asarray([item.trainIdx for item in accepted], dtype=np.int32),
        np.asarray([item.distance for item in accepted], dtype=np.float32),
    )


def backproject_pixels(
    pixels: NDArray[np.floating], depths: NDArray[np.floating], camera: CameraModel
) -> NDArray[np.float64]:
    pixels64 = np.asarray(pixels, dtype=np.float64)
    depth64 = np.asarray(depths, dtype=np.float64).reshape(-1)
    x = (pixels64[:, 0] - camera.cx) * depth64 / camera.fx
    y = (pixels64[:, 1] - camera.cy) * depth64 / camera.fy
    return np.column_stack((x, y, depth64))


def project_points(
    points_camera: NDArray[np.floating], camera: CameraModel
) -> NDArray[np.float64]:
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    return np.column_stack(
        (camera.fx * points[:, 0] / z + camera.cx, camera.fy * points[:, 1] / z + camera.cy)
    )


def sample_depth(
    depth: NDArray[np.float32], pixels: NDArray[np.floating]
) -> NDArray[np.float32]:
    xy = np.rint(pixels).astype(np.int32)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < depth.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < depth.shape[0])
    )
    values = np.full(len(xy), np.nan, dtype=np.float32)
    values[valid] = depth[xy[valid, 1], xy[valid, 0]]
    return values


def estimate_world_camera_pose(
    reference_pose: PoseSE3,
    reference_pixels: NDArray[np.float32],
    reference_depth: NDArray[np.float32],
    current_pixels: NDArray[np.float32],
    camera: CameraModel,
    *,
    reprojection_error_px: float,
    min_inliers: int,
) -> PoseEstimate | None:
    depths = sample_depth(reference_depth, reference_pixels)
    valid = np.isfinite(depths) & (depths > 0.0)
    if np.count_nonzero(valid) < max(min_inliers, 6):
        return None
    points_reference = backproject_pixels(reference_pixels[valid], depths[valid], camera)
    points_world = reference_pose.transform_points(points_reference)
    pixels_current = np.asarray(current_pixels[valid], dtype=np.float64)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_world,
        pixels_current,
        camera.matrix,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=reprojection_error_px,
        confidence=0.999,
        iterationsCount=150,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None
    indices = inliers.reshape(-1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            points_world[indices], pixels_current[indices], camera.matrix, None, rvec, tvec
        )
    except cv2.error:
        pass
    rotation_camera_world, _ = cv2.Rodrigues(rvec)
    pose_camera_world = PoseSE3(
        np.asarray(rotation_camera_world, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )
    return PoseEstimate(
        pose=pose_camera_world.inverse(),
        inlier_count=len(indices),
        match_count=int(np.count_nonzero(valid)),
    )


def rotation_angle_degrees(pose: PoseSE3) -> float:
    return float(np.degrees(Rotation.from_matrix(pose.rotation).magnitude()))


def pose_to_vector(pose: PoseSE3) -> NDArray[np.float64]:
    return np.concatenate((pose.translation, Rotation.from_matrix(pose.rotation).as_rotvec()))


def vector_to_pose(vector: NDArray[np.floating]) -> PoseSE3:
    value = np.asarray(vector, dtype=np.float64)
    return PoseSE3(Rotation.from_rotvec(value[3:]).as_matrix(), value[:3])


def relative_pose(first: PoseSE3, second: PoseSE3) -> PoseSE3:
    return first.inverse() @ second
