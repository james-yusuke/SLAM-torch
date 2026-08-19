from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


class TrackingStatus(StrEnum):
    TRACKING = "tracking"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    requested: str = "auto"

    def __post_init__(self) -> None:
        value = self.requested.lower()
        valid = value in {"auto", "cpu", "mps", "cuda"} or (
            value.startswith("cuda:") and value[5:].isdigit()
        )
        if not valid:
            raise ValueError(f"Unsupported device: {self.requested}")
        object.__setattr__(self, "requested", value)


@dataclass(frozen=True, slots=True)
class PoseSE3:
    """Rigid transform T_world_camera mapping camera points into world coordinates."""

    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("pose contains non-finite values")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> PoseSE3:
        return cls(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, matrix: FloatArray) -> PoseSE3:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4):
            raise ValueError("matrix must have shape (4, 4)")
        return cls(value[:3, :3], value[:3, 3])

    def as_matrix(self) -> FloatArray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> PoseSE3:
        rotation = self.rotation.T
        return PoseSE3(rotation, -(rotation @ self.translation))

    def compose(self, other: PoseSE3) -> PoseSE3:
        return PoseSE3(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )

    def __matmul__(self, other: PoseSE3) -> PoseSE3:
        return self.compose(other)

    def transform_points(self, points: FloatArray) -> FloatArray:
        value = np.asarray(points, dtype=np.float64)
        return (self.rotation @ value.T).T + self.translation


@dataclass(frozen=True, slots=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @property
    def matrix(self) -> FloatArray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def scaled(self, width: int, height: int) -> CameraModel:
        sx = width / self.width
        sy = height / self.height
        return CameraModel(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            distortion=self.distortion,
        )


@dataclass(slots=True)
class Frame:
    index: int
    timestamp: float
    image: UInt8Array
    camera: CameraModel
    ground_truth_pose: PoseSE3 | None = None
    ground_truth_depth: Float32Array | None = None


@dataclass(frozen=True, slots=True)
class Detection2D:
    class_id: int
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(slots=True)
class Keyframe:
    id: int
    frame_index: int
    timestamp: float
    image: UInt8Array
    camera: CameraModel
    pose: PoseSE3
    keypoints: Float32Array
    descriptors: UInt8Array | None
    depth: Float32Array
    detections: list[Detection2D] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MapPoint:
    id: int
    position_world: tuple[float, float, float]
    color_rgb: tuple[int, int, int]
    observations: int = 1


@dataclass(slots=True)
class ObjectLandmark3D:
    id: int
    label: str
    confidence: float
    center_world: FloatArray
    aabb_min_world: FloatArray
    aabb_max_world: FloatArray
    observation_count: int
    first_seen: float
    last_seen: float
    dynamic: bool


@dataclass(slots=True)
class TrajectoryEntry:
    frame_index: int
    timestamp: float
    pose: PoseSE3 | None
    status: TrackingStatus
    reference_keyframe_id: int | None = None
    transform_reference_frame: PoseSE3 | None = None
    ground_truth_pose: PoseSE3 | None = None


@dataclass(slots=True)
class MapState:
    keyframes: list[Keyframe] = field(default_factory=list)
    trajectory: list[TrajectoryEntry] = field(default_factory=list)
    points_world: Float32Array = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    colors_rgb: UInt8Array = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))
    objects: list[ObjectLandmark3D] = field(default_factory=list)
    loop_closures: int = 0


@runtime_checkable
class FrameSource(Protocol):
    name: str

    def __iter__(self) -> Iterator[Frame]: ...

    def validate(self) -> dict[str, object]: ...


@runtime_checkable
class DepthEstimator(Protocol):
    device: str

    def predict(self, image: UInt8Array) -> Float32Array: ...


@runtime_checkable
class ObjectDetector(Protocol):
    device: str

    def detect(self, image: UInt8Array) -> list[Detection2D]: ...
