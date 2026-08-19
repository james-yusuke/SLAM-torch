from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from slam_torch.config import MappingConfig
from slam_torch.geometry import backproject_pixels
from slam_torch.types import Detection2D, Keyframe, ObjectLandmark3D


def detection_mask(
    shape: tuple[int, int], detections: list[Detection2D], labels: set[str]
) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for detection in detections:
        if detection.label not in labels:
            continue
        x1, y1, x2, y2 = detection.box_xyxy
        left = max(0, min(width, int(np.floor(x1))))
        top = max(0, min(height, int(np.floor(y1))))
        right = max(0, min(width, int(np.ceil(x2))))
        bottom = max(0, min(height, int(np.ceil(y2))))
        mask[top:bottom, left:right] = True
    return mask


def pixels_outside_detections(
    pixels: NDArray[np.float32], detections: list[Detection2D], labels: set[str]
) -> NDArray[np.bool_]:
    keep = np.ones(len(pixels), dtype=bool)
    for detection in detections:
        if detection.label not in labels:
            continue
        x1, y1, x2, y2 = detection.box_xyxy
        inside = (
            (pixels[:, 0] >= x1)
            & (pixels[:, 0] <= x2)
            & (pixels[:, 1] >= y1)
            & (pixels[:, 1] <= y2)
        )
        keep &= ~inside
    return keep


def voxel_downsample(
    points: NDArray[np.float32], colors: NDArray[np.uint8], voxel_size: float
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    if len(points) == 0:
        return points, colors
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxels, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    point_sum = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) for axis in range(3)]
    )
    color_sum = np.column_stack(
        [np.bincount(inverse, weights=colors[:, axis]) for axis in range(3)]
    )
    result_points = (point_sum / count[:, None]).astype(np.float32)
    result_colors = np.clip(color_sum / count[:, None], 0, 255).astype(np.uint8)
    return result_points, result_colors


def keyframe_point_cloud(
    keyframe: Keyframe,
    config: MappingConfig,
    dynamic_labels: set[str],
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    depth = keyframe.depth
    rows = np.arange(0, depth.shape[0], config.stride)
    columns = np.arange(0, depth.shape[1], config.stride)
    xx, yy = np.meshgrid(columns, rows)
    pixels = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)
    values = depth[yy, xx].reshape(-1)
    valid = (
        np.isfinite(values)
        & (values >= config.min_depth_m)
        & (values <= config.max_depth_m)
    )
    dynamic = detection_mask(depth.shape, keyframe.detections, dynamic_labels)
    valid &= ~dynamic[yy.reshape(-1), xx.reshape(-1)]
    pixels = pixels[valid]
    values = values[valid]
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    points_camera = backproject_pixels(pixels, values, keyframe.camera)
    points_world = keyframe.pose.transform_points(points_camera).astype(np.float32)
    colors = keyframe.image[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
    return points_world, colors


@dataclass(slots=True)
class ObjectMapBuilder:
    config: MappingConfig
    dynamic_labels: set[str]
    _objects: list[ObjectLandmark3D] = field(default_factory=list, init=False)
    _next_id: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._objects.clear()
        self._next_id = 0

    def add_keyframe(self, keyframe: Keyframe) -> None:
        for detection in keyframe.detections:
            observation = self._observation(keyframe, detection)
            if observation is not None:
                self._merge_or_append(observation)

    def objects(self) -> list[ObjectLandmark3D]:
        return self._objects

    def _observation(
        self, keyframe: Keyframe, detection: Detection2D
    ) -> ObjectLandmark3D | None:
        height, width = keyframe.depth.shape
        x1, y1, x2, y2 = detection.box_xyxy
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        patch = keyframe.depth[top:bottom:2, left:right:2]
        yy, xx = np.mgrid[top:bottom:2, left:right:2]
        depths = patch.reshape(-1)
        pixels = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float32)
        valid = (
            np.isfinite(depths)
            & (depths >= self.config.min_depth_m)
            & (depths <= self.config.max_depth_m)
        )
        depths = depths[valid]
        pixels = pixels[valid]
        if len(depths) < 12:
            return None
        q1, q3 = np.percentile(depths, [25.0, 75.0])
        iqr = max(float(q3 - q1), 0.1)
        robust = (depths >= q1 - 1.5 * iqr) & (depths <= q3 + 1.5 * iqr)
        depths = depths[robust]
        pixels = pixels[robust]
        if len(depths) < 8:
            return None
        points_camera = backproject_pixels(pixels, depths, keyframe.camera)
        points_world = keyframe.pose.transform_points(points_camera)
        low = np.percentile(points_world, 10.0, axis=0)
        high = np.percentile(points_world, 90.0, axis=0)
        center = np.median(points_world, axis=0)
        return ObjectLandmark3D(
            id=-1,
            label=detection.label,
            confidence=detection.confidence,
            center_world=center,
            aabb_min_world=low,
            aabb_max_world=high,
            observation_count=1,
            first_seen=keyframe.timestamp,
            last_seen=keyframe.timestamp,
            dynamic=detection.label in self.dynamic_labels,
        )

    def _merge_or_append(self, observation: ObjectLandmark3D) -> None:
        candidates = [item for item in self._objects if item.label == observation.label]
        if observation.dynamic:
            candidates = [
                item
                for item in candidates
                if observation.last_seen - item.last_seen <= self.config.dynamic_ttl_seconds
            ]
        nearest = min(
            candidates,
            key=lambda item: float(np.linalg.norm(item.center_world - observation.center_world)),
            default=None,
        )
        if nearest is None or np.linalg.norm(nearest.center_world - observation.center_world) > (
            self.config.object_merge_distance_m
        ):
            observation.id = self._next_id
            self._next_id += 1
            self._objects.append(observation)
            return
        total = nearest.observation_count + 1
        if nearest.dynamic:
            nearest.center_world = observation.center_world
            nearest.aabb_min_world = observation.aabb_min_world
            nearest.aabb_max_world = observation.aabb_max_world
        else:
            nearest.center_world = (
                nearest.center_world * nearest.observation_count + observation.center_world
            ) / total
            nearest.aabb_min_world = np.minimum(
                nearest.aabb_min_world, observation.aabb_min_world
            )
            nearest.aabb_max_world = np.maximum(
                nearest.aabb_max_world, observation.aabb_max_world
            )
        nearest.confidence = max(nearest.confidence, observation.confidence)
        nearest.observation_count = total
        nearest.last_seen = observation.last_seen


def build_semantic_map(
    keyframes: list[Keyframe], config: MappingConfig, dynamic_labels: set[str]
) -> tuple[NDArray[np.float32], NDArray[np.uint8], list[ObjectLandmark3D]]:
    clouds: list[NDArray[np.float32]] = []
    colors: list[NDArray[np.uint8]] = []
    objects = ObjectMapBuilder(config, dynamic_labels)
    for keyframe in keyframes:
        points, rgb = keyframe_point_cloud(keyframe, config, dynamic_labels)
        if len(points):
            clouds.append(points)
            colors.append(rgb)
        objects.add_keyframe(keyframe)
    if not clouds:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            objects.objects(),
        )
    points = np.concatenate(clouds)
    rgb = np.concatenate(colors)
    points, rgb = voxel_downsample(points, rgb, config.voxel_size_m)
    return points, rgb, objects.objects()
