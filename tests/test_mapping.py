import numpy as np

from slam_torch.config import MappingConfig
from slam_torch.mapping import ObjectMapBuilder, build_semantic_map, voxel_downsample
from slam_torch.types import CameraModel, Detection2D, Keyframe, PoseSE3


def keyframe(timestamp: float = 0.0, x_translation: float = 0.0) -> Keyframe:
    image = np.full((24, 32, 3), [100, 150, 200], dtype=np.uint8)
    depth = np.full((24, 32), 4.0, dtype=np.float32)
    return Keyframe(
        id=0,
        frame_index=0,
        timestamp=timestamp,
        image=image,
        camera=CameraModel(32, 24, 20.0, 20.0, 16.0, 12.0),
        pose=PoseSE3(np.eye(3), np.asarray([x_translation, 0.0, 0.0])),
        keypoints=np.empty((0, 2), dtype=np.float32),
        descriptors=None,
        depth=depth,
        detections=[Detection2D(1, "chair", 0.9, (8.0, 6.0, 24.0, 20.0))],
    )


def test_voxel_downsample_averages_points_and_colors() -> None:
    points = np.asarray([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.asarray([[0, 0, 0], [100, 100, 100], [255, 0, 0]], dtype=np.uint8)
    reduced_points, reduced_colors = voxel_downsample(points, colors, 0.1)
    assert len(reduced_points) == 2
    assert any(np.array_equal(color, [50, 50, 50]) for color in reduced_colors)


def test_semantic_map_contains_point_cloud_and_object() -> None:
    config = MappingConfig(stride=2, voxel_size_m=0.1)
    points, colors, objects = build_semantic_map([keyframe()], config, {"person"})
    assert len(points) > 0
    assert points.shape == colors.shape
    assert len(objects) == 1
    assert objects[0].label == "chair"
    assert not objects[0].dynamic


def test_dynamic_objects_only_merge_inside_ttl() -> None:
    config = MappingConfig(dynamic_ttl_seconds=1.0)
    builder = ObjectMapBuilder(config, {"chair"})
    builder.add_keyframe(keyframe(0.0))
    builder.add_keyframe(keyframe(3.0))
    assert len(builder.objects()) == 2


def test_dynamic_object_keeps_latest_position() -> None:
    config = MappingConfig(dynamic_ttl_seconds=5.0, object_merge_distance_m=5.0)
    builder = ObjectMapBuilder(config, {"chair"})
    builder.add_keyframe(keyframe(0.0, 0.0))
    first_x = float(builder.objects()[0].center_world[0])
    builder.add_keyframe(keyframe(1.0, 1.0))
    assert len(builder.objects()) == 1
    assert float(builder.objects()[0].center_world[0]) > first_x + 0.9
