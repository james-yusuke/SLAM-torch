from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from slam_torch.config import AppConfig
from slam_torch.geometry import (
    DescriptorMatches,
    PoseEstimate,
    estimate_world_camera_pose,
    extract_orb,
    match_descriptors,
    relative_pose,
    rotation_angle_degrees,
)
from slam_torch.mapping import build_semantic_map, pixels_outside_detections
from slam_torch.pose_graph import PoseGraph, PoseGraphEdge
from slam_torch.types import (
    DepthEstimator,
    Frame,
    FrameSource,
    Keyframe,
    MapState,
    ObjectDetector,
    PoseSE3,
    TrackingStatus,
    TrajectoryEntry,
)


@dataclass(slots=True)
class SlamMetrics:
    processed_frames: int = 0
    tracked_frames: int = 0
    lost_frames: int = 0
    keyframes: int = 0
    loop_closures: int = 0
    elapsed_seconds: float = 0.0
    depth_inference_seconds: float = 0.0
    detection_inference_seconds: float = 0.0
    depth_abs_rel: float | None = None
    depth_evaluated_pixels: int = 0

    @property
    def tracking_rate(self) -> float:
        return self.tracked_frames / max(self.processed_frames, 1)

    @property
    def fps(self) -> float:
        return self.processed_frames / max(self.elapsed_seconds, 1e-9)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tracking_rate"] = self.tracking_rate
        result["fps"] = self.fps
        return result


@dataclass(slots=True)
class SlamResult:
    state: MapState
    metrics: SlamMetrics


class SemanticSlam:
    def __init__(
        self,
        config: AppConfig,
        depth_estimator: DepthEstimator,
        object_detector: ObjectDetector,
    ) -> None:
        self.config = config
        self.depth_estimator = depth_estimator
        self.object_detector = object_detector
        self.state = MapState()
        self.pose_graph = PoseGraph()
        self.metrics = SlamMetrics()
        self.dynamic_labels = set(config.detector.dynamic_classes)
        self._depth_relative_error_sum = 0.0
        self._depth_valid_count = 0

    def run(self, source: FrameSource) -> SlamResult:
        started = time.perf_counter()
        for frame in source:
            self._process_frame(frame)
        if not self.state.keyframes:
            raise RuntimeError("No frames were processed")
        if self.state.loop_closures:
            optimized = self.pose_graph.optimize(
                [item.pose for item in self.state.keyframes], max_iterations=20
            )
            for item, pose in zip(self.state.keyframes, optimized, strict=True):
                item.pose = pose
        self._refresh_trajectory()
        points, colors, objects = build_semantic_map(
            self.state.keyframes, self.config.mapping, self.dynamic_labels
        )
        self.state.points_world = points
        self.state.colors_rgb = colors
        self.state.objects = objects
        self.metrics.elapsed_seconds = time.perf_counter() - started
        self.metrics.keyframes = len(self.state.keyframes)
        self.metrics.loop_closures = self.state.loop_closures
        if self._depth_valid_count:
            self.metrics.depth_abs_rel = self._depth_relative_error_sum / self._depth_valid_count
            self.metrics.depth_evaluated_pixels = self._depth_valid_count
        return SlamResult(self.state, self.metrics)

    def _process_frame(self, frame: Frame) -> None:
        self.metrics.processed_frames += 1
        keypoints, descriptors = extract_orb(frame.image, self.config.tracking.max_features)
        if not self.state.keyframes:
            keyframe = self._make_keyframe(frame, PoseSE3.identity(), keypoints, descriptors)
            self.state.keyframes.append(keyframe)
            self.state.trajectory.append(
                TrajectoryEntry(
                    frame.index,
                    frame.timestamp,
                    keyframe.pose,
                    TrackingStatus.TRACKING,
                    reference_keyframe_id=keyframe.id,
                    transform_reference_frame=PoseSE3.identity(),
                    ground_truth_pose=frame.ground_truth_pose,
                )
            )
            self.metrics.tracked_frames += 1
            return

        tracked = self._track(keypoints, descriptors, frame)
        if tracked is None:
            previous = self.state.trajectory[-1]
            self.state.trajectory.append(
                TrajectoryEntry(
                    frame.index,
                    frame.timestamp,
                    previous.pose,
                    TrackingStatus.LOST,
                    reference_keyframe_id=previous.reference_keyframe_id,
                    transform_reference_frame=previous.transform_reference_frame,
                    ground_truth_pose=frame.ground_truth_pose,
                )
            )
            self.metrics.lost_frames += 1
            return
        reference, estimate, _matches = tracked
        self.metrics.tracked_frames += 1
        transform_reference_frame = relative_pose(reference.pose, estimate.pose)
        should_add = self._should_add_keyframe(frame, reference, estimate)
        if not should_add:
            self.state.trajectory.append(
                TrajectoryEntry(
                    frame.index,
                    frame.timestamp,
                    estimate.pose,
                    TrackingStatus.TRACKING,
                    reference_keyframe_id=reference.id,
                    transform_reference_frame=transform_reference_frame,
                    ground_truth_pose=frame.ground_truth_pose,
                )
            )
            return

        keyframe = self._make_keyframe(frame, estimate.pose, keypoints, descriptors)
        self.state.keyframes.append(keyframe)
        self.pose_graph.add_edge(
            PoseGraphEdge(
                reference.id,
                keyframe.id,
                transform_reference_frame,
                weight=max(1.0, estimate.inlier_count / 20.0),
            )
        )
        self.state.trajectory.append(
            TrajectoryEntry(
                frame.index,
                frame.timestamp,
                keyframe.pose,
                TrackingStatus.TRACKING,
                reference_keyframe_id=keyframe.id,
                transform_reference_frame=PoseSE3.identity(),
                ground_truth_pose=frame.ground_truth_pose,
            )
        )
        if self.config.loop_closure.enabled:
            loop = self._find_loop_closure(keyframe)
            if loop is not None:
                candidate, loop_estimate = loop
                self.pose_graph.add_edge(
                    PoseGraphEdge(
                        candidate.id,
                        keyframe.id,
                        relative_pose(candidate.pose, loop_estimate.pose),
                        weight=max(2.0, loop_estimate.inlier_count / 10.0),
                        kind="loop",
                    )
                )
                self.state.loop_closures += 1

    def _track(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray | None,
        frame: Frame,
    ) -> tuple[Keyframe, PoseEstimate, DescriptorMatches] | None:
        candidates = list(
            reversed(
                self.state.keyframes[-self.config.tracking.relocalization_keyframes :]
            )
        )
        for reference in candidates:
            matches = match_descriptors(
                reference.descriptors, descriptors, self.config.tracking.match_ratio
            )
            if len(matches) < self.config.tracking.min_matches:
                continue
            reference_pixels = reference.keypoints[matches.query_indices]
            current_pixels = keypoints[matches.train_indices]
            static = pixels_outside_detections(
                reference_pixels, reference.detections, self.dynamic_labels
            )
            reference_pixels = reference_pixels[static]
            current_pixels = current_pixels[static]
            if len(reference_pixels) < self.config.tracking.min_matches:
                continue
            estimate = estimate_world_camera_pose(
                reference.pose,
                reference_pixels,
                reference.depth,
                current_pixels,
                frame.camera,
                reprojection_error_px=self.config.tracking.pnp_reprojection_error_px,
                min_inliers=self.config.tracking.min_inliers,
            )
            if estimate is not None:
                return reference, estimate, matches
        return None

    def _should_add_keyframe(
        self, frame: Frame, reference: Keyframe, estimate: PoseEstimate
    ) -> bool:
        gap = frame.index - self.state.keyframes[-1].frame_index
        if gap < self.config.keyframes.min_frame_gap:
            return False
        if gap >= self.config.keyframes.max_frame_gap:
            return True
        motion = relative_pose(reference.pose, estimate.pose)
        return bool(
            np.linalg.norm(motion.translation) >= self.config.keyframes.translation_m
            or rotation_angle_degrees(motion) >= self.config.keyframes.rotation_deg
            or estimate.inlier_ratio <= self.config.keyframes.inlier_ratio
        )

    def _make_keyframe(
        self,
        frame: Frame,
        pose: PoseSE3,
        keypoints: np.ndarray,
        descriptors: np.ndarray | None,
    ) -> Keyframe:
        depth = self.depth_estimator.predict(frame.image)
        detections = self.object_detector.detect(frame.image)
        self.metrics.depth_inference_seconds += float(
            getattr(self.depth_estimator, "last_inference_seconds", 0.0)
        )
        self.metrics.detection_inference_seconds += float(
            getattr(self.object_detector, "last_inference_seconds", 0.0)
        )
        depth = np.asarray(depth, dtype=np.float32)
        depth *= self.config.depth.scale_factor
        depth[~np.isfinite(depth)] = np.nan
        depth[(depth <= 0.0) | (depth > self.config.depth.max_depth_m)] = np.nan
        self._accumulate_depth_metrics(depth, frame.ground_truth_depth)
        return Keyframe(
            id=len(self.state.keyframes),
            frame_index=frame.index,
            timestamp=frame.timestamp,
            image=frame.image,
            camera=frame.camera,
            pose=pose,
            keypoints=keypoints,
            descriptors=descriptors,
            depth=depth,
            detections=detections,
        )

    def _accumulate_depth_metrics(
        self, predicted: np.ndarray, ground_truth: np.ndarray | None
    ) -> None:
        if ground_truth is None:
            return
        truth = np.asarray(ground_truth, dtype=np.float32)
        if truth.shape != predicted.shape:
            truth = np.asarray(
                cv2.resize(
                    truth,
                    (predicted.shape[1], predicted.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ),
                dtype=np.float32,
            )
        valid = (
            np.isfinite(predicted)
            & np.isfinite(truth)
            & (truth > 0.0)
            & (truth <= self.config.depth.max_depth_m)
        )
        if np.any(valid):
            self._depth_relative_error_sum += float(
                np.sum(np.abs(predicted[valid] - truth[valid]) / truth[valid])
            )
            self._depth_valid_count += int(np.count_nonzero(valid))

    def _find_loop_closure(self, current: Keyframe) -> tuple[Keyframe, PoseEstimate] | None:
        candidates: list[tuple[int, Keyframe, DescriptorMatches]] = []
        for candidate in self.state.keyframes:
            if current.id - candidate.id < self.config.loop_closure.min_keyframe_gap:
                continue
            matches = match_descriptors(
                candidate.descriptors, current.descriptors, self.config.tracking.match_ratio
            )
            if len(matches) >= self.config.loop_closure.min_matches:
                candidates.append((len(matches), candidate, matches))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, candidate, matches in candidates[: self.config.loop_closure.max_candidates]:
            reference_pixels = candidate.keypoints[matches.query_indices]
            current_pixels = current.keypoints[matches.train_indices]
            static = pixels_outside_detections(
                reference_pixels, candidate.detections, self.dynamic_labels
            )
            estimate = estimate_world_camera_pose(
                candidate.pose,
                reference_pixels[static],
                candidate.depth,
                current_pixels[static],
                current.camera,
                reprojection_error_px=self.config.tracking.pnp_reprojection_error_px,
                min_inliers=self.config.loop_closure.min_inliers,
            )
            if estimate is not None:
                return candidate, estimate
        return None

    def _refresh_trajectory(self) -> None:
        for entry in self.state.trajectory:
            if (
                entry.reference_keyframe_id is None
                or entry.transform_reference_frame is None
                or entry.reference_keyframe_id >= len(self.state.keyframes)
            ):
                continue
            reference = self.state.keyframes[entry.reference_keyframe_id]
            entry.pose = reference.pose @ entry.transform_reference_frame
