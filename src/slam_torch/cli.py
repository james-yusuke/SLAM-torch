from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from slam_torch.assets import (
    asset_runtime_metadata,
    assets_status,
    fetch_assets,
    resolve_asset_root,
)
from slam_torch.config import load_config
from slam_torch.datasets import create_source
from slam_torch.devices import (
    DeviceUnavailableError,
    RuntimeDevice,
    device_metadata,
    enable_mps_fallback,
    is_cuda_oom,
    resolve_device,
)
from slam_torch.doctor import doctor_report
from slam_torch.evaluation import evaluate_run
from slam_torch.io import create_run_directory, save_run, write_failure
from slam_torch.models import build_models, fetch_models, model_status
from slam_torch.slam import SemanticSlam


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slam-torch", description="Monocular semantic SLAM")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser(
        "doctor", help="Inspect CPU, MPS, CUDA, and model availability"
    )
    doctor.add_argument("--asset-root", type=Path, default=None)
    doctor.add_argument("--require", dest="require_device", choices=("cpu", "mps", "cuda"))
    doctor.add_argument("--model-smoke", action="store_true")

    assets = subcommands.add_parser("assets", help="Manage local models and demo datasets")
    asset_commands = assets.add_subparsers(dest="assets_command", required=True)
    assets_fetch = asset_commands.add_parser("fetch", help="Fetch and verify local assets")
    assets_fetch.add_argument("--profile", default="demo", choices=("demo",))
    assets_fetch.add_argument(
        "--component",
        default="all",
        choices=("all", "models", "tartanair", "euroc"),
    )
    assets_fetch.add_argument("--asset-root", type=Path, default=None)
    assets_fetch.add_argument("--accept-licenses", action="store_true")
    assets_status_parser = asset_commands.add_parser("status", help="Verify local assets")
    assets_status_parser.add_argument("--profile", default="demo", choices=("demo",))
    assets_status_parser.add_argument("--asset-root", type=Path, default=None)

    models = subcommands.add_parser("models", help="Manage pretrained model weights")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    models_fetch = model_commands.add_parser("fetch", help="Download and verify all model weights")
    models_fetch.add_argument("--asset-root", type=Path, default=None)
    models_status = model_commands.add_parser("status", help="Show cached model status")
    models_status.add_argument("--asset-root", type=Path, default=None)

    datasets = subcommands.add_parser("datasets", help="Validate dataset layouts")
    dataset_commands = datasets.add_subparsers(dest="datasets_command", required=True)
    validate = dataset_commands.add_parser("validate")
    validate.add_argument("--type", choices=("tartanair", "euroc"), required=True)
    validate.add_argument("--input", type=Path, required=True)

    run = subcommands.add_parser("run", help="Run offline SLAM")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, default=None)
    run.add_argument("--asset-root", type=Path, default=None)
    run.add_argument("--device", default=None, help="auto, cpu, mps, cuda, or cuda:N")
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--max-frames", type=int, default=None)

    evaluate = subcommands.add_parser("evaluate", help="Evaluate a completed run")
    evaluate.add_argument("--run", type=Path, required=True)

    visualize = subcommands.add_parser("visualize", help="Open the saved point cloud")
    visualize.add_argument("--run", type=Path, required=True)
    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _resolve_runtime_device(
    requested: str,
    *,
    precision: str,
    asset_root: Path,
) -> RuntimeDevice:
    runtime = resolve_device(requested, precision=precision)
    if not runtime.is_cuda:
        return runtime
    report = doctor_report(asset_root, require_device="cuda")
    requirement = report.get("requirement", {})
    satisfied = isinstance(requirement, dict) and requirement.get("satisfied") is True
    if satisfied:
        return runtime
    raw_errors = requirement.get("errors", []) if isinstance(requirement, dict) else []
    errors = raw_errors if isinstance(raw_errors, list) else [raw_errors]
    message = "CUDA startup validation failed: " + "; ".join(str(item) for item in errors)
    if requested != "auto":
        raise DeviceUnavailableError(message)
    warnings.warn(message + "; trying MPS then CPU", stacklevel=2)
    for fallback in ("mps", "cpu"):
        try:
            return resolve_device(fallback, precision=precision)
        except DeviceUnavailableError:
            continue
    raise DeviceUnavailableError(message + "; no fallback backend is available")


def _run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    asset_root = resolve_asset_root(args.asset_root, config.assets.root)
    config.assets.root = str(asset_root)
    if args.device is not None:
        config.device.requested = args.device
    if args.output is not None:
        config.output.root = str(args.output)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        config.dataset.max_frames = args.max_frames
    raw_input = args.input if args.input is not None else config.dataset.path
    if raw_input is None:
        raise ValueError("Dataset input is required via --input or dataset.path in the config")
    input_path = Path(raw_input).expanduser()
    if not input_path.is_absolute():
        input_path = asset_root / input_path
    input_path = input_path.resolve()
    config.dataset.path = str(input_path)
    output_root = Path(config.output.root).expanduser()
    if not output_root.is_absolute():
        output_root = asset_root / output_root
    config.output.root = str(output_root.resolve())
    run_dir = create_run_directory(config.output.root)
    runtime: dict[str, object] = {
        "requested": config.device.requested,
        "resolved": None,
        "precision": config.device.precision,
        "asset_root": str(asset_root),
    }
    try:
        if config.device.allow_mps_fallback:
            enable_mps_fallback()
        runtime_device = _resolve_runtime_device(
            config.device.requested,
            precision=config.device.precision,
            asset_root=asset_root,
        )
        runtime = device_metadata(runtime_device)
        runtime["assets"] = asset_runtime_metadata(
            asset_root, dataset_name=config.dataset.type
        )
        source = create_source(
            config.dataset.type,
            input_path,
            max_frames=config.dataset.max_frames,
        )
        report = source.validate()
        if not report["valid"]:
            errors = report.get("errors", [])
            details = (
                "; ".join(str(item) for item in errors)
                if isinstance(errors, list)
                else str(errors)
            )
            raise ValueError("Dataset validation failed: " + details)
        depth, detector = build_models(config, runtime_device)
        result = SemanticSlam(config, depth, detector).run(source)
        if runtime_device.is_cuda:
            import torch

            runtime["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        save_run(run_dir, result, config, runtime)
        try:
            if (run_dir / "groundtruth.tum").is_file():
                evaluate_run(run_dir)
        except ValueError as exc:
            runtime["evaluation_warning"] = str(exc)
        _print({"status": "complete", "run": str(run_dir), "metrics": result.metrics.to_dict()})
        return 0
    except BaseException as exc:
        if is_cuda_oom(exc):
            runtime["cuda_oom"] = True
        write_failure(run_dir, exc, runtime)
        print(f"SLAM failed; diagnostic saved to {run_dir / 'failure.json'}", file=sys.stderr)
        raise


def _visualize(run_dir: Path) -> int:
    import open3d as o3d

    path = run_dir / "map.ply"
    if not path.is_file():
        raise FileNotFoundError(f"Missing point cloud: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    o3d.visualization.draw_geometries([cloud], window_name=f"SLAM-torch: {run_dir.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        report = doctor_report(
            args.asset_root,
            require_device=args.require_device,
            model_smoke=args.model_smoke,
        )
        _print(report)
        return 0 if report["ok"] else 1
    if args.command == "assets":
        if args.assets_command == "fetch":
            result = fetch_assets(
                args.asset_root,
                profile_name=args.profile,
                component=args.component,
                accept_licenses=args.accept_licenses,
            )
        else:
            result = assets_status(args.asset_root, profile_name=args.profile)
        _print(result)
        if args.assets_command == "fetch" and args.component != "all":
            components = result.get("components", {})
            selected = components.get(args.component, {}) if isinstance(components, dict) else {}
            return 0 if isinstance(selected, dict) and selected.get("verified") else 1
        return 0 if result["complete"] else 1
    if args.command == "models":
        root = resolve_asset_root(args.asset_root)
        _print(fetch_models(root) if args.models_command == "fetch" else model_status(root))
        return 0
    if args.command == "datasets":
        source = create_source(args.type, args.input)
        report = source.validate()
        _print(report)
        return 0 if report["valid"] else 1
    if args.command == "run":
        return _run_command(args)
    if args.command == "evaluate":
        _print(evaluate_run(args.run))
        return 0
    if args.command == "visualize":
        return _visualize(args.run)
    return 2
