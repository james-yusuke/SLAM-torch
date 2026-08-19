import numpy as np
from scipy.spatial.transform import Rotation

from slam_torch.geometry import (
    backproject_pixels,
    estimate_world_camera_pose,
    project_points,
)
from slam_torch.types import CameraModel, PoseSE3


def test_backproject_and_project_round_trip() -> None:
    camera = CameraModel(640, 480, 400.0, 400.0, 320.0, 240.0)
    pixels = np.asarray([[320.0, 240.0], [400.0, 200.0]], dtype=np.float32)
    depths = np.asarray([2.0, 3.0], dtype=np.float32)
    points = backproject_pixels(pixels, depths, camera)
    np.testing.assert_allclose(project_points(points, camera), pixels, atol=1e-6)


def test_pnp_recovers_world_camera_pose() -> None:
    camera = CameraModel(640, 480, 450.0, 450.0, 320.0, 240.0)
    xx, yy, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-0.7, 0.7, 4),
        np.asarray([4.0, 5.0]),
    )
    world_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    reference_pixels = project_points(world_points, camera).astype(np.float32)
    reference_depth = np.full((480, 640), np.nan, dtype=np.float32)
    rounded = np.rint(reference_pixels).astype(int)
    reference_depth[rounded[:, 1], rounded[:, 0]] = world_points[:, 2]
    expected = PoseSE3(
        Rotation.from_euler("y", 3.0, degrees=True).as_matrix(),
        np.asarray([0.2, -0.05, 0.1]),
    )
    current_points = expected.inverse().transform_points(world_points)
    current_pixels = project_points(current_points, camera).astype(np.float32)
    estimate = estimate_world_camera_pose(
        PoseSE3.identity(),
        reference_pixels,
        reference_depth,
        current_pixels,
        camera,
        reprojection_error_px=2.0,
        min_inliers=10,
    )
    assert estimate is not None
    np.testing.assert_allclose(estimate.pose.translation, expected.translation, atol=2e-2)
    np.testing.assert_allclose(estimate.pose.rotation, expected.rotation, atol=2e-2)
