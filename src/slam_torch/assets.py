from __future__ import annotations

import bisect
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import yaml

ASSET_LOCK_NAME = "assets.lock.yaml"
COMPLETE_NAME = ".complete.json"
EUROC_API_ROOT = "https://www.research-collection.ethz.ch/server/api/core"


@dataclass(frozen=True, slots=True)
class AssetSpec:
    id: str
    source: str
    revision: str | None
    size_bytes: int | None
    sha256: str | None
    license: str
    target: str


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    name: str
    frames: int
    datasets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssetStatus:
    id: str
    path: str
    available: bool
    verified: bool
    file_count: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ResolvedDownload:
    url: str
    size_bytes: int | None
    checksum: str | None
    checksum_algorithm: str = "sha256"


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ASSET_LOCK_NAME).is_file():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    return source_root if (source_root / ASSET_LOCK_NAME).is_file() else current


def resolve_asset_root(
    cli_root: str | Path | None = None,
    config_root: str | Path | None = None,
) -> Path:
    configured = cli_root or config_root or os.environ.get("SLAM_TORCH_HOME")
    if configured is None:
        configured = os.environ.get("SLAM_TORCH_CACHE")
    return Path(configured).expanduser().resolve() if configured else project_root()


def load_asset_lock(path: str | Path | None = None) -> dict[str, Any]:
    lock_path = Path(path) if path is not None else project_root() / ASSET_LOCK_NAME
    with lock_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError(f"Unsupported or invalid asset lock: {lock_path}")
    return value


def dataset_profile(lock: dict[str, Any], name: str) -> DatasetProfile:
    profiles = lock.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise ValueError(f"Unknown asset profile: {name}")
    raw = profiles[name]
    if not isinstance(raw, dict):
        raise TypeError(f"Asset profile {name} must be a mapping")
    frames = int(raw.get("frames", 0))
    datasets = tuple(str(item) for item in raw.get("datasets", []))
    if frames <= 0 or not datasets:
        raise ValueError(f"Asset profile {name} is incomplete")
    return DatasetProfile(name, frames, datasets)


def _asset_spec(raw: dict[str, Any], *, default_id: str | None = None) -> AssetSpec:
    return AssetSpec(
        id=str(raw.get("id", default_id or "")),
        source=str(raw["source"]),
        revision=str(raw["revision"]) if raw.get("revision") is not None else None,
        size_bytes=int(raw["size_bytes"]) if raw.get("size_bytes") is not None else None,
        sha256=str(raw["sha256"]) if raw.get("sha256") is not None else None,
        license=str(raw["license"]),
        target=str(raw["target"]),
    )


def _dataset_entry(lock: dict[str, Any], name: str) -> dict[str, Any]:
    datasets = lock.get("datasets")
    if not isinstance(datasets, dict) or name not in datasets:
        raise ValueError(f"Dataset is missing from asset lock: {name}")
    raw = datasets[name]
    if not isinstance(raw, dict):
        raise TypeError(f"Dataset {name} must be a mapping")
    return raw


def _dataset_specs(lock: dict[str, Any], name: str) -> list[AssetSpec]:
    assets = _dataset_entry(lock, name).get("assets")
    if not isinstance(assets, list):
        raise TypeError(f"Dataset {name}.assets must be a list")
    return [_asset_spec(item) for item in assets if isinstance(item, dict)]


def checksum_file(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm.lower().replace("-", ""))
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return checksum_file(path, "sha256")


def tree_sha256(root: str | Path, *, excluded: Iterable[str] = (COMPLETE_NAME,)) -> str:
    base = Path(root)
    excluded_names = set(excluded)
    digest = hashlib.sha256()
    files = sorted(
        (path for path in base.rglob("*") if path.is_file() and path.name not in excluded_names),
        key=lambda item: item.relative_to(base).as_posix(),
    )
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _ensure_free_space(path: Path, required_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    safety = min(max(required_bytes // 10, 64 * 1024 * 1024), 1024 * 1024 * 1024)
    free = shutil.disk_usage(path).free
    if free < required_bytes + safety:
        raise OSError(
            f"Insufficient disk space at {path}: required at least "
            f"{required_bytes + safety} bytes, available {free} bytes"
        )


def _remaining_download_bytes(target: Path, expected_bytes: int | None) -> int:
    if expected_bytes is None:
        return 0
    partial = target.with_name(target.name + ".part")
    completed = max(
        target.stat().st_size if target.is_file() else 0,
        partial.stat().st_size if partial.is_file() else 0,
    )
    if completed > expected_bytes:
        completed = 0
    return expected_bytes - completed


def _verify_download(path: Path, resolved: _ResolvedDownload) -> bool:
    if not path.is_file():
        return False
    if resolved.size_bytes is not None and path.stat().st_size != resolved.size_bytes:
        return False
    if resolved.checksum is not None:
        actual = checksum_file(path, resolved.checksum_algorithm)
        return actual.lower() == resolved.checksum.lower()
    return True


def download_resumable(
    resolved: _ResolvedDownload,
    target: str | Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> Path:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verify_download(destination, resolved):
        return destination
    if destination.exists():
        destination.unlink()
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if resolved.size_bytes is not None and offset > resolved.size_bytes:
        partial.unlink()
        offset = 0
    remaining = max(0, (resolved.size_bytes or 0) - offset)
    _ensure_free_space(destination.parent, remaining)

    headers = {"User-Agent": "slam-torch/0.1 asset-fetcher"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(resolved.url, headers=headers)
    try:
        try:
            response = opener(request, timeout=120)
        except TypeError:
            response = opener(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and _verify_download(partial, resolved):
            partial.replace(destination)
            return destination
        raise

    status = getattr(response, "status", None)
    if offset and status != 206:
        offset = 0
    mode = "ab" if offset else "wb"
    with response, partial.open(mode) as output:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            output.write(block)

    if not _verify_download(partial, resolved):
        actual_size = partial.stat().st_size
        if resolved.size_bytes is not None and actual_size < resolved.size_bytes:
            raise RuntimeError(
                f"Incomplete download retained for resume: {resolved.url}: "
                f"received {actual_size} of {resolved.size_bytes} bytes"
            )
        actual_checksum = (
            checksum_file(partial, resolved.checksum_algorithm)
            if resolved.checksum is not None
            else None
        )
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded asset verification failed for {resolved.url}: "
            f"size={actual_size}, checksum={actual_checksum}"
        )
    partial.replace(destination)
    return destination


def download_verified(
    url: str,
    target: str | Path,
    *,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> Path:
    return download_resumable(
        _ResolvedDownload(url, size_bytes, sha256, "sha256"), target
    )


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/hal+json, application/json",
            "User-Agent": "slam-torch/0.1 asset-fetcher",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    if "json" not in content_type.lower():
        raise RuntimeError(
            "ETH Research Collection did not return JSON metadata. "
            "Open the dataset page in a browser and retry after access is restored."
        )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("ETH Research Collection returned invalid metadata")
    return value


def _hal_entries(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    embedded = payload.get("_embedded", {})
    values = embedded.get(name, []) if isinstance(embedded, dict) else []
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _resolve_dspace_download(spec: AssetSpec) -> _ResolvedDownload:
    match = re.fullmatch(r"dspace://([^/]+)/(.+)", spec.source)
    if match is None:
        return _ResolvedDownload(spec.source, spec.size_bytes, spec.sha256)
    item_uuid, requested_name = match.groups()
    bundles = _request_json(f"{EUROC_API_ROOT}/items/{item_uuid}/bundles?size=100")
    originals = [
        item
        for item in _hal_entries(bundles, "bundles")
        if item.get("name") == "ORIGINAL"
    ]
    if not originals:
        raise RuntimeError("EuRoC ORIGINAL bundle was not found in ETH Research Collection")
    bundle_uuid = str(originals[0].get("uuid", ""))
    bitstreams = _request_json(f"{EUROC_API_ROOT}/bundles/{bundle_uuid}/bitstreams?size=100")
    candidates = [
        item
        for item in _hal_entries(bitstreams, "bitstreams")
        if requested_name.lower() in str(item.get("name", "")).lower()
    ]
    if len(candidates) != 1:
        names = [str(item.get("name")) for item in _hal_entries(bitstreams, "bitstreams")]
        raise RuntimeError(
            f"Expected one EuRoC bitstream matching {requested_name!r}; available={names}"
        )
    bitstream = candidates[0]
    bitstream_uuid = str(bitstream.get("uuid", ""))
    if spec.revision is not None and spec.revision != bitstream_uuid:
        raise RuntimeError(
            "EuRoC bitstream revision differs from assets.lock.yaml: "
            f"expected {spec.revision}, official metadata reports {bitstream_uuid}"
        )
    links = bitstream.get("_links", {})
    content = links.get("content", {}) if isinstance(links, dict) else {}
    url = content.get("href") if isinstance(content, dict) else None
    if not isinstance(url, str):
        url = f"{EUROC_API_ROOT}/bitstreams/{bitstream_uuid}/content"
    checksum = bitstream.get("checkSum", {})
    checksum_value = checksum.get("value") if isinstance(checksum, dict) else None
    checksum_algorithm = (
        str(checksum.get("checkSumAlgorithm", "sha256"))
        if isinstance(checksum, dict)
        else "sha256"
    )
    size = bitstream.get("sizeBytes", spec.size_bytes)
    resolved_size = int(size) if size is not None else None
    if (
        spec.size_bytes is not None
        and resolved_size is not None
        and spec.size_bytes != resolved_size
    ):
        raise RuntimeError(
            "EuRoC archive size differs from assets.lock.yaml: "
            f"expected {spec.size_bytes}, official metadata reports {resolved_size}"
        )
    return _ResolvedDownload(
        url=url,
        size_bytes=resolved_size,
        checksum=str(checksum_value) if checksum_value else spec.sha256,
        checksum_algorithm=checksum_algorithm if checksum_value else "sha256",
    )


def _normalized_member(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or any(part == ".." for part in path.parts)
        or (path.parts and re.match(r"^[A-Za-z]:", path.parts[0]))
    ):
        raise ValueError(f"Unsafe ZIP member: {name}")
    return path


def _validate_zip(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        _normalized_member(info.filename)


def _frame_number(name: str) -> int | None:
    matches = re.findall(r"\d+", PurePosixPath(name).stem)
    return int(matches[0]) if matches else None


def _member_map(
    archive: zipfile.ZipFile,
    *,
    trajectory: str,
    directory: str,
    suffixes: tuple[str, ...],
) -> dict[int, zipfile.ZipInfo]:
    result: dict[int, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        path = _normalized_member(info.filename)
        if info.is_dir() or trajectory not in path.parts or directory not in path.parts:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        index = _frame_number(path.name)
        if index is not None:
            result.setdefault(index, info)
    return result


def _copy_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def _write_completion(stage: Path, payload: dict[str, object]) -> dict[str, object]:
    payload = dict(payload)
    payload["tree_sha256"] = tree_sha256(stage)
    (stage / COMPLETE_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _install_stage(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.invalid-{uuid.uuid4().hex[:8]}")
        target.rename(backup)
    try:
        stage.rename(target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup is not None and backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def extract_tartanair_demo(
    rgb_archive: str | Path,
    depth_archive: str | Path,
    target: str | Path,
    *,
    frames: int = 300,
    trajectory: str = "P000",
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        with zipfile.ZipFile(rgb_archive) as rgb_zip, zipfile.ZipFile(depth_archive) as depth_zip:
            _validate_zip(rgb_zip)
            _validate_zip(depth_zip)
            images = _member_map(
                rgb_zip,
                trajectory=trajectory,
                directory="image_lcam_front",
                suffixes=(".png", ".jpg", ".jpeg"),
            )
            depths = _member_map(
                depth_zip,
                trajectory=trajectory,
                directory="depth_lcam_front",
                suffixes=(".npy", ".png"),
            )
            selected = sorted(set(images) & set(depths))[:frames]
            if len(selected) != frames:
                raise ValueError(
                    f"TartanAir {trajectory} contains {len(selected)} aligned RGB/depth frames; "
                    f"expected {frames}"
                )
            pose_info: zipfile.ZipInfo | None = None
            pose_archive = rgb_zip
            for archive in (rgb_zip, depth_zip):
                for info in archive.infolist():
                    path = _normalized_member(info.filename)
                    if (
                        trajectory in path.parts
                        and path.suffix.lower() == ".txt"
                        and path.name.lower().startswith("pose_lcam")
                    ):
                        pose_info = info
                        pose_archive = archive
                        break
                if pose_info is not None:
                    break
            if pose_info is None:
                raise ValueError(f"TartanAir pose file for {trajectory} was not found")
            pose_lines = [
                line
                for line in pose_archive.read(pose_info).decode("utf-8-sig").splitlines()
                if line.strip()
            ]
            if not selected or max(selected) >= len(pose_lines):
                raise ValueError("TartanAir pose rows do not cover the selected frames")
            extraction_bytes = sum(
                images[index].file_size + depths[index].file_size for index in selected
            )
            _ensure_free_space(stage.parent, extraction_bytes + 64 * 1024**2)
            for index in selected:
                _copy_zip_member(
                    rgb_zip,
                    images[index],
                    stage / "image_lcam_front" / PurePosixPath(images[index].filename).name,
                )
                _copy_zip_member(
                    depth_zip,
                    depths[index],
                    stage / "depth_lcam_front" / PurePosixPath(depths[index].filename).name,
                )
            (stage / "pose_lcam_front.txt").write_text(
                "\n".join(pose_lines[index] for index in selected) + "\n", encoding="utf-8"
            )
        completion = _write_completion(
            stage,
            {
                "id": "tartanair-oldtownfall-easy-p000-300",
                "dataset": "tartanair",
                "frames": frames,
                "trajectory": trajectory,
                "camera": "lcam_front",
                "license": "CC-BY-4.0",
                "source": source_metadata or {},
            },
        )
        _install_stage(stage, destination)
        return completion
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _zip_text(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    return archive.read(info).decode("utf-8-sig")


def _find_member(
    archive: zipfile.ZipFile,
    suffix: str,
    *,
    must_contain: str | None = None,
) -> zipfile.ZipInfo | None:
    normalized_suffix = suffix.lower().replace("\\", "/")
    for info in archive.infolist():
        name = _normalized_member(info.filename).as_posix().lower()
        if name.endswith(normalized_suffix) and (
            must_contain is None or must_contain.lower() in name
        ):
            return info
    return None


def _parse_csv_text(text: str) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            headers.append(line)
            continue
        parsed = next(csv.reader([line]))
        if parsed:
            rows.append(parsed)
    return headers, rows


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for header in headers:
            stream.write(header + "\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def _extract_euroc_from_zip(
    archive: zipfile.ZipFile,
    stage: Path,
    *,
    frames: int,
) -> None:
    _validate_zip(archive)
    camera_csv = _find_member(
        archive, "mav0/cam0/data.csv", must_contain="mh_01_easy"
    ) or _find_member(archive, "mav0/cam0/data.csv")
    sensor_yaml = _find_member(
        archive, "mav0/cam0/sensor.yaml", must_contain="mh_01_easy"
    ) or _find_member(archive, "mav0/cam0/sensor.yaml")
    ground_truth_csv = _find_member(
        archive,
        "mav0/state_groundtruth_estimate0/data.csv",
        must_contain="mh_01_easy",
    ) or _find_member(archive, "mav0/state_groundtruth_estimate0/data.csv")
    if camera_csv is None or sensor_yaml is None or ground_truth_csv is None:
        raise ValueError("EuRoC MH_01_easy camera, calibration, or ground-truth files are missing")

    camera_headers, camera_rows = _parse_csv_text(_zip_text(archive, camera_csv))
    selected_rows = camera_rows[:frames]
    if len(selected_rows) != frames:
        raise ValueError(f"EuRoC contains {len(selected_rows)} camera rows; expected {frames}")
    base = PurePosixPath(camera_csv.filename).parent
    info_by_name = {
        _normalized_member(info.filename).as_posix(): info for info in archive.infolist()
    }
    for row in selected_rows:
        if len(row) < 2:
            raise ValueError("EuRoC camera CSV row is incomplete")
        member_name = (base / "data" / row[1]).as_posix()
        info = info_by_name.get(member_name)
        if info is None:
            raise ValueError(f"EuRoC image is missing from archive: {member_name}")
        _copy_zip_member(archive, info, stage / "mav0" / "cam0" / "data" / row[1])
    _write_csv(stage / "mav0" / "cam0" / "data.csv", camera_headers, selected_rows)
    _copy_zip_member(archive, sensor_yaml, stage / "mav0" / "cam0" / "sensor.yaml")

    gt_headers, gt_rows = _parse_csv_text(_zip_text(archive, ground_truth_csv))
    if not gt_rows:
        raise ValueError("EuRoC ground-truth CSV has no rows")
    gt_times = [int(row[0]) for row in gt_rows]
    aligned_gt: list[list[str]] = []
    for row in selected_rows:
        timestamp = int(row[0])
        position = bisect.bisect_left(gt_times, timestamp)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(gt_rows)]
        nearest = min(candidates, key=lambda index: abs(gt_times[index] - timestamp))
        # The extracted file is frame-aligned rather than a raw IMU-rate stream.
        # Preserve the nearest official pose, but key it by the camera timestamp.
        aligned_gt.append([row[0], *gt_rows[nearest][1:]])
    _write_csv(
        stage / "mav0" / "state_groundtruth_estimate0" / "data.csv",
        gt_headers,
        aligned_gt,
    )


def extract_euroc_demo(
    machine_hall_archive: str | Path,
    target: str | Path,
    *,
    frames: int = 300,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    nested_path: Path | None = None
    try:
        with zipfile.ZipFile(machine_hall_archive) as outer:
            _validate_zip(outer)
            direct = _find_member(outer, "mav0/cam0/data.csv", must_contain="mh_01_easy")
            if direct is not None:
                _extract_euroc_from_zip(outer, stage, frames=frames)
            else:
                nested = next(
                    (
                        info
                        for info in outer.infolist()
                        if PurePosixPath(info.filename).name.lower() == "mh_01_easy.zip"
                    ),
                    None,
                )
                if nested is None:
                    raise ValueError("EuRoC Machine Hall archive does not contain MH_01_easy")
                _ensure_free_space(stage.parent, nested.file_size + 512 * 1024**2)
                nested_path = stage.parent / f".{destination.name}-{uuid.uuid4().hex}.zip"
                _copy_zip_member(outer, nested, nested_path)
                with zipfile.ZipFile(nested_path) as sequence:
                    _extract_euroc_from_zip(sequence, stage, frames=frames)
        completion = _write_completion(
            stage,
            {
                "id": "euroc-MH_01_easy-300",
                "dataset": "euroc",
                "frames": frames,
                "sequence": "MH_01_easy",
                "camera": "cam0",
                "license": "In Copyright - Non-Commercial Use Permitted",
                "source": source_metadata or {},
            },
        )
        _install_stage(stage, destination)
        return completion
    finally:
        if nested_path is not None:
            nested_path.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


def _read_completion(target: Path) -> dict[str, Any] | None:
    path = target / COMPLETE_NAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _dataset_status(
    name: str,
    target: Path,
    *,
    expected_frames: int,
    verify_tree: bool,
    specs: list[AssetSpec] | None = None,
) -> AssetStatus:
    errors: list[str] = []
    completion = _read_completion(target)
    if completion is None:
        errors.append("completion manifest is missing or invalid")
    elif int(completion.get("frames", -1)) != expected_frames:
        errors.append("completion manifest frame count does not match the profile")
    if completion is not None and specs:
        source = completion.get("source", {})
        source = source if isinstance(source, dict) else {}
        expected_revisions = {spec.revision for spec in specs if spec.revision is not None}
        if expected_revisions and source.get("revision") not in expected_revisions:
            errors.append("dataset source revision does not match assets.lock.yaml")
        if name == "tartanair":
            checksum_keys = {
                "tartanair_rgb": "rgb_sha256",
                "tartanair_depth": "depth_sha256",
            }
            for spec in specs:
                key = checksum_keys.get(spec.id)
                if key is not None and spec.sha256 is not None and source.get(key) != spec.sha256:
                    errors.append(f"dataset source checksum does not match for {spec.id}")
        else:
            spec = specs[0]
            if spec.sha256 is not None and source.get("archive_sha256") != spec.sha256:
                errors.append("EuRoC archive SHA-256 does not match assets.lock.yaml")
    if name == "tartanair":
        image_count = len(list((target / "image_lcam_front").glob("*")))
        depth_count = len(list((target / "depth_lcam_front").glob("*")))
        pose_path = target / "pose_lcam_front.txt"
        pose_count = (
            len([line for line in pose_path.read_text(encoding="utf-8").splitlines() if line])
            if pose_path.is_file()
            else 0
        )
        file_count = image_count
        if (image_count, depth_count, pose_count) != (
            expected_frames,
            expected_frames,
            expected_frames,
        ):
            errors.append(
                f"expected {expected_frames} RGB/depth/pose entries; "
                f"got {image_count}/{depth_count}/{pose_count}"
            )
    else:
        image_count = len(list((target / "mav0" / "cam0" / "data").glob("*")))
        camera_csv = target / "mav0" / "cam0" / "data.csv"
        gt_csv = target / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        camera_count = (
            len(_parse_csv_text(camera_csv.read_text(encoding="utf-8"))[1])
            if camera_csv.is_file()
            else 0
        )
        gt_count = (
            len(_parse_csv_text(gt_csv.read_text(encoding="utf-8"))[1])
            if gt_csv.is_file()
            else 0
        )
        file_count = image_count
        if (image_count, camera_count, gt_count) != (
            expected_frames,
            expected_frames,
            expected_frames,
        ):
            errors.append(
                f"expected {expected_frames} EuRoC image/camera/pose entries; "
                f"got {image_count}/{camera_count}/{gt_count}"
            )
    if completion is not None and verify_tree and target.is_dir():
        expected_tree = completion.get("tree_sha256")
        if not isinstance(expected_tree, str) or tree_sha256(target) != expected_tree:
            errors.append("dataset tree SHA-256 mismatch")
    available = target.is_dir() and completion is not None
    return AssetStatus(
        name,
        str(target),
        available,
        available and not errors,
        file_count,
        tuple(errors),
    )


def assets_status(
    asset_root: str | Path | None = None,
    *,
    profile_name: str = "demo",
    lock_path: str | Path | None = None,
    verify_tree: bool = True,
) -> dict[str, object]:
    from slam_torch.models import model_status

    root = resolve_asset_root(asset_root)
    resolved_lock_path = (
        Path(lock_path) if lock_path is not None else project_root() / ASSET_LOCK_NAME
    )
    lock = load_asset_lock(resolved_lock_path)
    profile = dataset_profile(lock, profile_name)
    raw_models = model_status(root)
    depth = raw_models["depth"]
    detector = raw_models["detector"]
    models_ok = bool(
        isinstance(depth, dict)
        and isinstance(detector, dict)
        and depth.get("sha256_ok")
        and detector.get("sha256_ok")
    )
    model_errors = () if models_ok else ("one or more model weights are missing or invalid",)
    statuses: dict[str, AssetStatus] = {
        "models": AssetStatus(
            "models",
            str(root / "models"),
            (root / "models").is_dir(),
            models_ok,
            2 if models_ok else 0,
            model_errors,
        )
    }
    for name in profile.datasets:
        entry = _dataset_entry(lock, name)
        target = root / str(entry["target"])
        statuses[name] = _dataset_status(
            name,
            target,
            expected_frames=profile.frames,
            verify_tree=verify_tree,
            specs=_dataset_specs(lock, name),
        )
    components: dict[str, dict[str, object]] = {}
    raw_model_specs = lock.get("models", {})
    model_specs = raw_model_specs if isinstance(raw_model_specs, dict) else {}
    for name, status in statuses.items():
        payload = status.to_dict()
        if name == "models":
            payload["completion"] = models_ok
            payload["checksums"] = {
                model_id: spec.get("sha256")
                for model_id, spec in model_specs.items()
                if isinstance(spec, dict)
            }
            payload["licenses"] = {
                model_id: spec.get("license")
                for model_id, spec in model_specs.items()
                if isinstance(spec, dict)
            }
        else:
            entry = _dataset_entry(lock, name)
            completion = _read_completion(root / str(entry["target"]))
            expected = {
                spec.id: spec.sha256 for spec in _dataset_specs(lock, name)
            }
            payload["completion"] = completion is not None
            payload["license"] = entry.get("license")
            payload["checksums"] = {
                "expected_sha256": expected,
                "tree_sha256": completion.get("tree_sha256")
                if completion is not None
                else None,
            }
            payload["source"] = completion.get("source") if completion is not None else None
        components[name] = payload
    raw_datasets = lock.get("datasets", {})
    datasets = raw_datasets if isinstance(raw_datasets, dict) else {}
    dataset_licenses = {
        name: entry.get("license")
        for name, entry in datasets.items()
        if isinstance(entry, dict)
    }
    return {
        "profile": profile.name,
        "asset_root": str(root),
        "asset_lock": str(resolved_lock_path.resolve()),
        "asset_lock_sha256": sha256_file(resolved_lock_path),
        "licenses": dataset_licenses,
        "complete": all(status.verified for status in statuses.values()),
        "components": components,
    }


def asset_runtime_metadata(
    asset_root: str | Path,
    *,
    dataset_name: str,
    lock_path: str | Path | None = None,
) -> dict[str, object]:
    root = resolve_asset_root(asset_root)
    resolved_lock_path = (
        Path(lock_path) if lock_path is not None else project_root() / ASSET_LOCK_NAME
    )
    lock = load_asset_lock(resolved_lock_path)
    entry = _dataset_entry(lock, dataset_name)
    completion = _read_completion(root / str(entry["target"]))
    model_manifest = root / "models" / "manifest.json"
    models = (
        json.loads(model_manifest.read_text(encoding="utf-8"))
        if model_manifest.is_file()
        else None
    )
    return {
        "asset_root": str(root),
        "asset_lock": str(resolved_lock_path.resolve()),
        "asset_lock_sha256": sha256_file(resolved_lock_path),
        "models": models,
        "dataset": completion,
        "license": entry.get("license"),
    }


def fetch_assets(
    asset_root: str | Path | None = None,
    *,
    profile_name: str = "demo",
    component: str = "all",
    accept_licenses: bool = False,
    lock_path: str | Path | None = None,
) -> dict[str, object]:
    from slam_torch.models import fetch_models

    valid_components = {"all", "models", "tartanair", "euroc"}
    if component not in valid_components:
        raise ValueError(f"Unsupported asset component: {component}")
    if component in {"all", "tartanair", "euroc"} and not accept_licenses:
        raise PermissionError(
            "Dataset licenses must be accepted explicitly with --accept-licenses"
        )
    root = resolve_asset_root(asset_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = load_asset_lock(lock_path)
    profile = dataset_profile(lock, profile_name)

    if component in {"all", "models"}:
        fetch_models(root)

    if component in {"all", "tartanair"}:
        entry = _dataset_entry(lock, "tartanair")
        tartanair_specs = _dataset_specs(lock, "tartanair")
        target = root / str(entry["target"])
        status = _dataset_status(
            "tartanair",
            target,
            expected_frames=profile.frames,
            verify_tree=True,
            specs=tartanair_specs,
        )
        if not status.verified:
            specs = {spec.id: spec for spec in tartanair_specs}
            rgb_spec = specs["tartanair_rgb"]
            depth_spec = specs["tartanair_depth"]
            archive_dir = (root / rgb_spec.target).parent
            required_archives = _remaining_download_bytes(
                root / rgb_spec.target, rgb_spec.size_bytes
            ) + _remaining_download_bytes(root / depth_spec.target, depth_spec.size_bytes)
            _ensure_free_space(archive_dir, required_archives + 1024**3)
            rgb_archive = download_resumable(
                _ResolvedDownload(
                    rgb_spec.source, rgb_spec.size_bytes, rgb_spec.sha256, "sha256"
                ),
                root / rgb_spec.target,
            )
            depth_archive = download_resumable(
                _ResolvedDownload(
                    depth_spec.source, depth_spec.size_bytes, depth_spec.sha256, "sha256"
                ),
                root / depth_spec.target,
            )
            extract_tartanair_demo(
                rgb_archive,
                depth_archive,
                target,
                frames=profile.frames,
                trajectory=str(entry["trajectory"]),
                source_metadata={
                    "revision": rgb_spec.revision,
                    "rgb_sha256": rgb_spec.sha256,
                    "depth_sha256": depth_spec.sha256,
                },
            )
            rgb_archive.unlink(missing_ok=True)
            depth_archive.unlink(missing_ok=True)

    if component in {"all", "euroc"}:
        entry = _dataset_entry(lock, "euroc")
        euroc_specs = _dataset_specs(lock, "euroc")
        target = root / str(entry["target"])
        status = _dataset_status(
            "euroc",
            target,
            expected_frames=profile.frames,
            verify_tree=True,
            specs=euroc_specs,
        )
        if not status.verified:
            spec = euroc_specs[0]
            resolved = _resolve_dspace_download(spec)
            if resolved.size_bytes is not None:
                _ensure_free_space(
                    (root / spec.target).parent,
                    _remaining_download_bytes(root / spec.target, resolved.size_bytes)
                    + 4 * 1024**3,
                )
            archive = download_resumable(resolved, root / spec.target)
            archive_sha256 = sha256_file(archive)
            if spec.sha256 is not None and archive_sha256 != spec.sha256:
                archive.unlink(missing_ok=True)
                raise ValueError(
                    "EuRoC archive SHA-256 mismatch: "
                    f"expected {spec.sha256}, got {archive_sha256}"
                )
            extract_euroc_demo(
                archive,
                target,
                frames=profile.frames,
                source_metadata={
                    "item": spec.source,
                    "revision": spec.revision,
                    "archive_sha256": archive_sha256,
                    "official_checksum": resolved.checksum,
                    "official_checksum_algorithm": resolved.checksum_algorithm,
                },
            )
            archive.unlink(missing_ok=True)

    return assets_status(
        root, profile_name=profile_name, lock_path=lock_path, verify_tree=True
    )
