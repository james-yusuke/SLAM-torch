import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import slam_torch.assets as assets
from slam_torch.datasets import EurocSource, TartanAirSource


def _png_bytes(value: int, *, grayscale: bool = False) -> bytes:
    shape = (12, 16) if grayscale else (12, 16, 3)
    image = np.full(shape, value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return bytes(encoded)


def _npy_bytes(value: float) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.full((12, 16), value, dtype=np.float32))
    return stream.getvalue()


def _tartanair_archives(tmp_path: Path, frames: int = 3) -> tuple[Path, Path]:
    rgb = tmp_path / "rgb.zip"
    depth = tmp_path / "depth.zip"
    prefix = "OldTownFall/Data_easy/P000"
    with zipfile.ZipFile(rgb, "w") as archive:
        for index in range(frames):
            archive.writestr(
                f"{prefix}/image_lcam_front/{index:06d}.png", _png_bytes(index)
            )
        poses = "".join(f"{index} 0 0 0 0 0 1\n" for index in range(frames))
        archive.writestr(f"{prefix}/pose_lcam_front.txt", poses)
    with zipfile.ZipFile(depth, "w") as archive:
        for index in range(frames):
            archive.writestr(
                f"{prefix}/depth_lcam_front/{index:06d}.npy",
                _npy_bytes(float(index + 1)),
            )
    return rgb, depth


def _euroc_sequence_zip(tmp_path: Path, frames: int = 3) -> Path:
    sequence = tmp_path / "MH_01_easy.zip"
    prefix = "MH_01_easy/mav0"
    timestamps = [1_000_000_000 + index * 50_000_000 for index in range(frames)]
    camera_csv = "#timestamp [ns],filename\n" + "".join(
        f"{timestamp},{timestamp}.png\n" for timestamp in timestamps
    )
    ground_truth = "#timestamp,p,q\n" + "".join(
        f"{timestamp + 20_000_000},{index},0,0,1,0,0,0\n"
        for index, timestamp in enumerate(timestamps)
    )
    sensor = """sensor_type: camera
resolution: [16, 12]
intrinsics: [10.0, 10.0, 8.0, 6.0]
distortion_coefficients: [0.0, 0.0, 0.0, 0.0]
T_BS:
  rows: 4
  cols: 4
  data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
"""
    with zipfile.ZipFile(sequence, "w") as archive:
        archive.writestr(f"{prefix}/cam0/data.csv", camera_csv)
        archive.writestr(f"{prefix}/cam0/sensor.yaml", sensor)
        archive.writestr(
            f"{prefix}/state_groundtruth_estimate0/data.csv", ground_truth
        )
        for index, timestamp in enumerate(timestamps):
            archive.writestr(
                f"{prefix}/cam0/data/{timestamp}.png",
                _png_bytes(index, grayscale=True),
            )
    return sequence


def test_asset_root_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    legacy = tmp_path / "legacy"
    config = tmp_path / "config"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("SLAM_TORCH_HOME", str(home))
    monkeypatch.setenv("SLAM_TORCH_CACHE", str(legacy))

    assert assets.resolve_asset_root(explicit, config) == explicit.resolve()
    assert assets.resolve_asset_root(None, config) == config.resolve()
    assert assets.resolve_asset_root() == home.resolve()
    monkeypatch.delenv("SLAM_TORCH_HOME")
    assert assets.resolve_asset_root() == legacy.resolve()


def test_tartanair_extracts_aligned_subset_and_manifest(tmp_path: Path) -> None:
    rgb, depth = _tartanair_archives(tmp_path)
    target = tmp_path / "data" / "tartanair"
    result = assets.extract_tartanair_demo(rgb, depth, target, frames=3)
    assert result["frames"] == 3
    assert result["tree_sha256"] == assets.tree_sha256(target)
    source = TartanAirSource(target)
    frames = list(source)
    assert len(frames) == 3
    assert all(frame.ground_truth_pose is not None for frame in frames)
    assert all(frame.ground_truth_depth is not None for frame in frames)


def test_euroc_extracts_nested_sequence_and_aligned_ground_truth(tmp_path: Path) -> None:
    sequence = _euroc_sequence_zip(tmp_path)
    outer = tmp_path / "machine-hall.zip"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(sequence, "Machine Hall Datasets/MH_01_easy.zip")
    target = tmp_path / "data" / "euroc"
    result = assets.extract_euroc_demo(outer, target, frames=3)
    assert result["frames"] == 3
    camera_rows = assets._parse_csv_text(
        (target / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8")
    )[1]
    ground_truth_rows = assets._parse_csv_text(
        (target / "mav0" / "state_groundtruth_estimate0" / "data.csv").read_text(
            encoding="utf-8"
        )
    )[1]
    assert [row[0] for row in ground_truth_rows] == [row[0] for row in camera_rows]
    source = EurocSource(target)
    frames = list(source)
    assert len(frames) == 3
    assert all(frame.ground_truth_pose is not None for frame in frames)


def test_zip_traversal_is_rejected_without_installing_target(tmp_path: Path) -> None:
    rgb, depth = _tartanair_archives(tmp_path)
    with zipfile.ZipFile(rgb, "a") as archive:
        archive.writestr("../outside.txt", b"unsafe")
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        assets.extract_tartanair_demo(rgb, depth, target, frames=3)
    assert not target.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_failed_extract_preserves_existing_dataset(tmp_path: Path) -> None:
    rgb, depth = _tartanair_archives(tmp_path)
    with zipfile.ZipFile(rgb, "a") as archive:
        archive.writestr("/absolute.txt", b"unsafe")
    target = tmp_path / "dataset"
    target.mkdir()
    marker = target / "existing.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        assets.extract_tartanair_demo(rgb, depth, target, frames=3)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".dataset-invalid-*"))


def test_atomic_install_replaces_invalid_target_without_leaving_backup(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    stage = tmp_path / ".dataset-stage"
    stage.mkdir()
    (stage / "new.txt").write_text("new", encoding="utf-8")

    assets._install_stage(stage, target)

    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob(".dataset.invalid-*"))


class _Response(io.BytesIO):
    def __init__(self, value: bytes, status: int) -> None:
        super().__init__(value)
        self.status = status


def test_download_resumes_a_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"
    target.with_name("asset.bin.part").write_bytes(b"abc")
    expected = hashlib.sha256(b"abcdef").hexdigest()

    def opener(request: Any, timeout: int) -> _Response:
        assert timeout == 120
        assert request.headers["Range"] == "bytes=3-"
        return _Response(b"def", 206)

    resolved = assets._ResolvedDownload("https://example.invalid/a", 6, expected)
    assets.download_resumable(resolved, target, opener=opener)
    assert target.read_bytes() == b"abcdef"
    assert not target.with_name("asset.bin.part").exists()


def test_download_checksum_failure_removes_invalid_partial(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"

    def opener(_request: Any, timeout: int) -> _Response:
        assert timeout == 120
        return _Response(b"bad", 200)

    resolved = assets._ResolvedDownload("https://example.invalid/a", 3, "0" * 64)
    with pytest.raises(RuntimeError, match="verification failed"):
        assets.download_resumable(resolved, target, opener=opener)
    assert not target.exists()
    assert not target.with_name("asset.bin.part").exists()


def test_incomplete_download_is_retained_for_resume(tmp_path: Path) -> None:
    target = tmp_path / "asset.bin"

    def opener(_request: Any, timeout: int) -> _Response:
        assert timeout == 120
        return _Response(b"partial", 200)

    resolved = assets._ResolvedDownload("https://example.invalid/a", 20, None)
    with pytest.raises(RuntimeError, match="retained for resume"):
        assets.download_resumable(resolved, target, opener=opener)
    assert target.with_name("asset.bin.part").read_bytes() == b"partial"


def test_completion_manifest_detects_tree_changes(tmp_path: Path) -> None:
    rgb, depth = _tartanair_archives(tmp_path)
    target = tmp_path / "dataset"
    assets.extract_tartanair_demo(rgb, depth, target, frames=3)
    completion = json.loads((target / assets.COMPLETE_NAME).read_text(encoding="utf-8"))
    assert completion["tree_sha256"] == assets.tree_sha256(target)
    image = next((target / "image_lcam_front").iterdir())
    image.write_bytes(b"changed")
    assert completion["tree_sha256"] != assets.tree_sha256(target)


def test_dataset_status_rejects_a_stale_locked_revision(tmp_path: Path) -> None:
    rgb, depth = _tartanair_archives(tmp_path)
    target = tmp_path / "dataset"
    assets.extract_tartanair_demo(
        rgb,
        depth,
        target,
        frames=3,
        source_metadata={"revision": "old", "rgb_sha256": "a", "depth_sha256": "b"},
    )
    specs = [
        assets.AssetSpec("tartanair_rgb", "url", "new", None, "a", "license", "rgb"),
        assets.AssetSpec("tartanair_depth", "url", "new", None, "b", "license", "depth"),
    ]
    status = assets._dataset_status(
        "tartanair", target, expected_frames=3, verify_tree=True, specs=specs
    )
    assert not status.verified
    assert "revision" in " ".join(status.errors)
