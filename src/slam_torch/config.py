from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass(slots=True)
class DatasetConfig:
    type: str = "tartanair"
    path: str | None = None
    max_frames: int | None = None


@dataclass(slots=True)
class DeviceConfig:
    requested: str = "auto"
    precision: str = "balanced"
    allow_mps_fallback: bool = True


@dataclass(slots=True)
class DepthConfig:
    model_id: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    revision: str = "fd2c22027eaf20374204f14099b8341e1925ad39"
    max_depth_m: float = 80.0
    scale_factor: float = 1.0


@dataclass(slots=True)
class DetectorConfig:
    score_threshold: float = 0.45
    allowed_classes: list[str] = field(default_factory=list)
    dynamic_classes: list[str] = field(
        default_factory=lambda: [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "train",
            "truck",
            "bird",
            "cat",
            "dog",
            "horse",
        ]
    )


@dataclass(slots=True)
class TrackingConfig:
    max_features: int = 2500
    match_ratio: float = 0.75
    min_matches: int = 20
    min_inliers: int = 12
    pnp_reprojection_error_px: float = 3.0
    relocalization_keyframes: int = 30


@dataclass(slots=True)
class KeyframeConfig:
    min_frame_gap: int = 5
    max_frame_gap: int = 20
    translation_m: float = 0.35
    rotation_deg: float = 12.0
    inlier_ratio: float = 0.55


@dataclass(slots=True)
class LoopClosureConfig:
    enabled: bool = True
    min_keyframe_gap: int = 20
    min_matches: int = 50
    min_inliers: int = 20
    max_candidates: int = 5


@dataclass(slots=True)
class MappingConfig:
    stride: int = 6
    min_depth_m: float = 0.25
    max_depth_m: float = 80.0
    voxel_size_m: float = 0.08
    object_merge_distance_m: float = 1.5
    dynamic_ttl_seconds: float = 2.0


@dataclass(slots=True)
class OutputConfig:
    root: str = "runs"


@dataclass(slots=True)
class AssetsConfig:
    root: str | None = None


@dataclass(slots=True)
class AppConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    loop_closure: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.dataset.type not in {"tartanair", "euroc"}:
            raise ValueError("dataset.type must be 'tartanair' or 'euroc'")
        if self.dataset.max_frames is not None and self.dataset.max_frames <= 0:
            raise ValueError("dataset.max_frames must be positive")
        if self.device.precision not in {"balanced", "deterministic"}:
            raise ValueError("device.precision must be 'balanced' or 'deterministic'")
        if self.depth.scale_factor <= 0.0:
            raise ValueError("depth.scale_factor must be positive")
        if not 0.0 < self.detector.score_threshold <= 1.0:
            raise ValueError("detector.score_threshold must be in (0, 1]")
        if not 0.0 < self.tracking.match_ratio < 1.0:
            raise ValueError("tracking.match_ratio must be in (0, 1)")
        if self.tracking.min_matches < 6 or self.tracking.min_inliers < 4:
            raise ValueError("tracking match/inlier thresholds are too small")
        if self.keyframes.min_frame_gap <= 0:
            raise ValueError("keyframes.min_frame_gap must be positive")
        if self.keyframes.max_frame_gap < self.keyframes.min_frame_gap:
            raise ValueError("keyframes.max_frame_gap must be >= min_frame_gap")
        if self.mapping.stride <= 0 or self.mapping.voxel_size_m <= 0:
            raise ValueError("mapping stride and voxel size must be positive")
        if self.mapping.min_depth_m >= self.mapping.max_depth_m:
            raise ValueError("mapping depth range is invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {UnionType, Union}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _build_dataclass(cls: type[T], raw: dict[str, Any], path: str = "config") -> T:
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must be a mapping")
    definitions = {item.name: item for item in fields(cast(Any, cls))}
    unknown = sorted(set(raw) - set(definitions))
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, value in raw.items():
        annotation = _strip_optional(hints[name])
        if isinstance(annotation, type) and is_dataclass(annotation):
            values[name] = _build_dataclass(annotation, value, f"{path}.{name}")
        else:
            values[name] = value
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = _build_dataclass(AppConfig, raw)
    config.validate()
    return config


def save_config(config: AppConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, sort_keys=False, allow_unicode=True)
