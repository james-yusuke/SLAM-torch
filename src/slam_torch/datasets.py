from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from slam_torch.types import CameraModel, Frame, FrameSource, PoseSE3


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _find_directory(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_dir():
            return direct
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in names:
            return candidate
    return None


def _find_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.name in names:
            return candidate
    return None


def _pose_from_tartan(values: list[float]) -> PoseSE3:
    if len(values) < 7:
        raise ValueError("TartanAir pose row must contain tx ty tz qx qy qz qw")
    return PoseSE3(Rotation.from_quat(values[3:7]).as_matrix(), np.asarray(values[:3]))


class TartanAirSource:
    name = "tartanair"

    def __init__(self, root: str | Path, max_frames: int | None = None) -> None:
        self.root = Path(root)
        self.max_frames = max_frames
        self.image_dir = _find_directory(
            self.root, ("image_lcam_front", "image_left", "lcam_front")
        )
        self.depth_dir = _find_directory(
            self.root, ("depth_lcam_front", "depth_left", "lcam_front_depth")
        )
        self.pose_file = _find_file(
            self.root, ("pose_lcam_front.txt", "pose_left.txt", "pose.txt")
        )
        self._depth_index: dict[int, Path] = {}
        if self.depth_dir is not None:
            for path in self.depth_dir.iterdir():
                match = re.search(r"\d+", path.stem)
                if match and path.suffix.lower() in {".npy", ".png"}:
                    self._depth_index[int(match.group())] = path

    def validate(self) -> dict[str, object]:
        errors: list[str] = []
        if not self.root.is_dir():
            errors.append(f"Input directory does not exist: {self.root}")
        if self.image_dir is None:
            errors.append("RGB directory was not found")
        images = self._images() if self.image_dir else []
        if not images:
            errors.append("No RGB images were found")
        if self.depth_dir is not None and len(self._depth_index) < len(images):
            depth_count = len(self._depth_index)
            errors.append(
                f"Depth entries do not cover every RGB frame: {depth_count}/{len(images)}"
            )
        if self.pose_file is not None:
            try:
                pose_count = len(self._poses())
            except (OSError, ValueError) as exc:
                errors.append(f"Invalid TartanAir pose file: {exc}")
            else:
                if pose_count < len(images):
                    errors.append(
                        f"Pose rows do not cover every RGB frame: {pose_count}/{len(images)}"
                    )
        return {
            "valid": not errors,
            "dataset": self.name,
            "root": str(self.root),
            "image_count": len(images),
            "has_depth": self.depth_dir is not None,
            "has_ground_truth_pose": self.pose_file is not None,
            "errors": errors,
        }

    def _images(self) -> list[Path]:
        if self.image_dir is None:
            return []
        paths = [
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return sorted(paths, key=_natural_key)

    def _poses(self) -> list[PoseSE3]:
        if self.pose_file is None:
            return []
        poses: list[PoseSE3] = []
        for line in self.pose_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                poses.append(_pose_from_tartan([float(value) for value in line.split()]))
        return poses

    def _depth_for(self, image_path: Path) -> np.ndarray | None:
        if self.depth_dir is None:
            return None
        match = re.search(r"\d+", image_path.stem)
        if match is None:
            return None
        path = self._depth_index.get(int(match.group()))
        if path is None:
            return None
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path), dtype=np.float32)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 4:
            return np.squeeze(np.ascontiguousarray(raw).view("<f4"), axis=-1).copy()
        return raw.astype(np.float32)

    def __iter__(self) -> Iterator[Frame]:
        report = self.validate()
        if not report["valid"]:
            raise ValueError("; ".join(report["errors"]))  # type: ignore[arg-type]
        images = self._images()
        poses = self._poses()
        limit = len(images) if self.max_frames is None else min(len(images), self.max_frames)
        for index, path in enumerate(images[:limit]):
            image = _read_rgb(path)
            height, width = image.shape[:2]
            camera = CameraModel(
                width, height, width / 2.0, height / 2.0, width / 2.0, height / 2.0
            )
            yield Frame(
                index=index,
                timestamp=index / 10.0,
                image=image,
                camera=camera,
                ground_truth_pose=poses[index] if index < len(poses) else None,
                ground_truth_depth=self._depth_for(path),
            )


def _opencv_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^%YAML:\s*1\.0\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("!!opencv-matrix", "")
    value = yaml.safe_load(text)
    return value if isinstance(value, dict) else {}


class EurocSource:
    name = "euroc"

    def __init__(self, root: str | Path, max_frames: int | None = None) -> None:
        self.root = Path(root)
        self.max_frames = max_frames
        self.cam_root = self.root / "mav0" / "cam0"
        self.data_csv = self.cam_root / "data.csv"
        self.sensor_yaml = self.cam_root / "sensor.yaml"
        self.gt_csv = self.root / "mav0" / "state_groundtruth_estimate0" / "data.csv"

    def validate(self) -> dict[str, object]:
        errors = [
            f"Missing required file: {path}"
            for path in (self.data_csv, self.sensor_yaml)
            if not path.is_file()
        ]
        count = len(self._rows()) if not errors else 0
        if not errors:
            missing_images = sum(
                not (self.cam_root / "data" / filename).is_file()
                for _, filename in self._rows()
            )
            if missing_images:
                errors.append(f"Missing {missing_images} EuRoC camera images")
            try:
                camera, extrinsic = self._camera_and_extrinsic()
                camera_values = np.asarray(
                    [camera.fx, camera.fy, camera.cx, camera.cy], dtype=np.float64
                )
                if (
                    camera.width <= 0
                    or camera.height <= 0
                    or camera.fx <= 0.0
                    or camera.fy <= 0.0
                    or not np.all(np.isfinite(camera_values))
                    or not np.all(np.isfinite(extrinsic.as_matrix()))
                ):
                    raise ValueError("camera calibration contains invalid values")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"Invalid EuRoC camera calibration: {exc}")
        return {
            "valid": not errors and count > 0,
            "dataset": self.name,
            "root": str(self.root),
            "image_count": count,
            "has_depth": False,
            "has_ground_truth_pose": self.gt_csv.is_file(),
            "errors": errors + (["No camera rows were found"] if not errors and count == 0 else []),
        }

    def _rows(self) -> list[tuple[int, str]]:
        if not self.data_csv.is_file():
            return []
        rows: list[tuple[int, str]] = []
        with self.data_csv.open("r", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if not row or row[0].startswith("#"):
                    continue
                rows.append((int(row[0]), row[1]))
        return rows

    def _camera_and_extrinsic(self) -> tuple[CameraModel, PoseSE3]:
        raw = _opencv_yaml(self.sensor_yaml)
        width, height = (int(value) for value in raw["resolution"])
        fx, fy, cx, cy = (float(value) for value in raw["intrinsics"])
        distortion = tuple(float(value) for value in raw.get("distortion_coefficients", []))
        matrix_raw = raw.get("T_BS", {})
        matrix = np.asarray(
            matrix_raw.get("data", np.eye(4).reshape(-1)), dtype=np.float64
        ).reshape(4, 4)
        return CameraModel(width, height, fx, fy, cx, cy, distortion), PoseSE3.from_matrix(matrix)

    def _ground_truth(self, transform_body_sensor: PoseSE3) -> list[tuple[int, PoseSE3]]:
        result: list[tuple[int, PoseSE3]] = []
        if not self.gt_csv.is_file():
            return result
        with self.gt_csv.open("r", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if not row or row[0].startswith("#"):
                    continue
                timestamp = int(row[0])
                translation = np.asarray([float(value) for value in row[1:4]])
                # EuRoC stores q_w, q_x, q_y, q_z.
                qw, qx, qy, qz = (float(value) for value in row[4:8])
                body = PoseSE3(Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), translation)
                result.append((timestamp, body @ transform_body_sensor))
        return result

    def __iter__(self) -> Iterator[Frame]:
        report = self.validate()
        if not report["valid"]:
            raise ValueError("; ".join(report["errors"]))  # type: ignore[arg-type]
        camera, body_sensor = self._camera_and_extrinsic()
        rows = self._rows()
        ground_truth_rows = self._ground_truth(body_sensor)
        ground_truth = dict(ground_truth_rows)
        gt_times = np.asarray(sorted(ground_truth), dtype=np.int64)
        frame_aligned_ground_truth = (
            (self.root / ".complete.json").is_file()
            and len(ground_truth_rows) == len(rows)
        )
        limit = len(rows) if self.max_frames is None else min(len(rows), self.max_frames)
        for index, (timestamp_ns, filename) in enumerate(rows[:limit]):
            image_raw = cv2.imread(str(self.cam_root / "data" / filename), cv2.IMREAD_GRAYSCALE)
            if image_raw is None:
                raise ValueError(f"Could not read EuRoC image: {filename}")
            if camera.distortion:
                image_raw = cv2.undistort(
                    image_raw,
                    camera.matrix,
                    np.asarray(camera.distortion, dtype=np.float64),
                )
            image = np.asarray(
                cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB), dtype=np.uint8
            )
            gt_pose = None
            if frame_aligned_ground_truth:
                gt_pose = ground_truth_rows[index][1]
            elif len(gt_times):
                nearest_index = int(np.argmin(np.abs(gt_times - timestamp_ns)))
                if abs(int(gt_times[nearest_index]) - timestamp_ns) <= 10_000_000:
                    gt_pose = ground_truth[int(gt_times[nearest_index])]
            yield Frame(
                index=index,
                timestamp=timestamp_ns / 1e9,
                image=image,
                camera=CameraModel(
                    camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy
                ),
                ground_truth_pose=gt_pose,
            )


def create_source(
    dataset_type: str, root: str | Path, max_frames: int | None = None
) -> FrameSource:
    if dataset_type == "tartanair":
        return TartanAirSource(root, max_frames)
    if dataset_type == "euroc":
        return EurocSource(root, max_frames)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
