# SLAM-torch

English | [日本語](README-ja.md)

SLAM-torch is an offline semantic SLAM system that combines PyTorch-based monocular depth estimation and general object detection with ORB/PnP tracking and pose-graph optimization. It generates a camera trajectory, a colored 3D point cloud, and 3D object positions from a monocular RGB sequence.

> [!WARNING]
> This is a research MVP. Metric scale from monocular depth is an estimate and must not be used directly for flight control, safety decisions, or collision avoidance. ROS 2, live cameras, Jetson, and real-time guarantees are outside the scope of this initial release.

## Supported platforms

- macOS Apple Silicon: PyTorch MPS with CPU fallback
- Linux x86_64: NVIDIA CUDA 12.8 with CPU fallback
- Windows x86_64: NVIDIA CUDA 12.8 with CPU fallback
- Python 3.11 and uv 0.12 or newer

On Linux and Windows, uv selects the PyTorch CUDA 12.8 wheels. Compiling against a local CUDA Toolkit is not required, but a [CUDA 12.8-compatible NVIDIA driver](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html) is. The minimum compatible versions are `525.60.13` on Linux and `528.33` on Windows; a 570-series or newer driver is recommended for CUDA 12.8. Run `slam-torch doctor --require cuda` to inspect the actual environment. Jetson is not supported because JetPack uses different PyTorch packages from the standard x86_64 wheels.

## Setup

Install uv on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra dev
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --frozen --extra dev
```

The same `uv.lock` automatically selects PyTorch from PyPI on Apple Silicon and the CUDA 12.8 build from the explicit `pytorch-cu128` index on Linux/Windows x86_64, following [uv's PyTorch index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/). Then run diagnostics and fetch the pinned assets:

```bash
uv run slam-torch doctor
uv run slam-torch assets fetch --profile demo --accept-licenses
uv run slam-torch assets status --profile demo
```

`assets fetch` places the following items under the repository's `models/` and `data/` directories. The first run downloads approximately 14 GB of official archives; at least 20 GB of free space is recommended for downloading and extraction. Downloads resume from `.part` files. After hash verification and extraction of the 300-frame subsets, the source archives are deleted. Normal SLAM runs operate offline.

- Depth Anything V2 Metric Outdoor Small (Apache-2.0)
- TorchVision SSDLite320 MobileNetV3 COCO
- First 300 frames of TartanAir `OldTownFall/Data_easy/P000/lcam_front`
- First 300 frames of EuRoC `MH_01_easy/cam0`

Sources, pinned revisions, sizes, checksums, and licenses are recorded in `assets.lock.yaml`. TartanAir is licensed under CC BY 4.0. EuRoC permits non-commercial use, so `--accept-licenses` is required before downloading the datasets. Large binary assets are excluded from Git.

The asset root is resolved in this order: `--asset-root`, the `assets.root` configuration value, `SLAM_TORCH_HOME`, and the repository root. The legacy `SLAM_TORCH_CACHE` variable remains available as a migration fallback. Compatibility commands for fetching models only are also supported:

```bash
uv run slam-torch assets fetch --component models
uv run slam-torch models fetch
```

## Datasets

### TartanAir V2

The default demo aligns RGB images, depth maps, and poses to the same first 300 indices from the official outdoor Easy sequence. Only RGB is used as SLAM input; ground-truth depth and poses are used exclusively for evaluation.

`tartanair-demo.yaml` explicitly sets `depth.scale_factor: 0.49` as a fixed scale calibration for the synthetic camera domain. Normal runs do not consult ground-truth depth, and the factor is recorded in `run.yaml`. Calibrate this value independently using a known distance or another sensor when using a real drone camera.

```bash
uv run slam-torch datasets validate \
  --type tartanair \
  --input data/tartanair/oldtownfall-easy-p000-300
uv run slam-torch run --device auto --config configs/tartanair-demo.yaml
```

The following directory names are detected automatically:

- RGB: `image_lcam_front`, `image_left`, `lcam_front`
- Depth: `depth_lcam_front`, `depth_left`, `lcam_front_depth`
- Poses: `pose_lcam_front.txt`, `pose_left.txt`, `pose.txt`

### EuRoC MAV

The default demo extracts only `MH_01_easy` from the official Machine Hall archive in the ETH Research Collection. This initial version uses `cam0` only; the IMU and right camera are not used.

```bash
uv run slam-torch datasets validate \
  --type euroc \
  --input data/euroc/MH_01_easy-300
uv run slam-torch run --device auto --config configs/euroc-demo.yaml
```

External full sequences can also be used by overriding the configured path with `run --input PATH`.

## GPU and precision settings

`--device` accepts `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`.

- `auto` selects the first available backend in this order: CUDA, MPS, CPU.
- An unavailable explicitly requested CUDA device is an error.
- A CUDA out-of-memory error during a run writes `failure.json` and stops; it never falls back to CPU implicitly.
- `balanced` uses FP16 autocast for CUDA depth estimation and FP32 for object detection.
- `deterministic` uses FP32 with TF32 disabled for comparison and testing.

Check `runtime.resolved` in `run.yaml` and `metrics.json` to verify the selected backend. `metrics.json` also records the GPU name, compute capability, peak VRAM, and per-model inference time.

On an NVIDIA machine, the following diagnostic verifies the CUDA 12.8 PyTorch build, driver, tensor operations, and that both models and their inputs are placed on CUDA. It exits with a non-zero status on failure.

```bash
uv run slam-torch doctor --require cuda --model-smoke
uv run slam-torch run --device cuda --config configs/tartanair-demo.yaml
```

## Outputs

Each run is saved under `runs/<UTC-timestamp>-<ID>/`.

- `trajectory.tum`: estimated camera trajectory in TUM format
- `groundtruth.tum`: ground-truth trajectory when provided by the dataset
- `map.ply`: colored 3D point cloud
- `objects.json`: object classes, confidence scores, 3D centers, approximate AABBs, and dynamic state
- `metrics.json`: tracking rate, FPS, depth error, ATE/RPE, inference times, and device information
- `run.yaml`: resolved configuration and reproducibility metadata
- `failure.json`: initialization, dataset, CUDA OOM, and other failure details

```bash
uv run slam-torch evaluate --run runs/<run-id>
uv run slam-torch visualize --run runs/<run-id>
```

Poses use `T_world_camera`, with the first camera frame as the world origin. Image coordinates follow the OpenCV convention: x points right, y points down, and camera z points forward.

## Pipeline

1. Convert a dataset-specific format into calibrated RGB `Frame` objects.
2. Extract ORB features and match descriptors on every frame.
3. Build 3D features from keyframe depth and estimate poses with PnP RANSAC.
4. Relocalize against previous keyframes after tracking failure.
5. Geometrically verify non-neighboring keyframes and optimize an SE(3) pose graph.
6. Rebuild the point cloud and object map from the optimized poses and depth maps.

Dynamic classes such as people and vehicles are excluded from the persistent point cloud and the 3D features used by SLAM. They remain in `objects.json` as dynamic observations with their last observed positions.

## Development and testing

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Tests that require real models, official datasets, or a CUDA GPU are separated from the regular CPU CI jobs. The self-hosted CUDA workflow reuses a persistent asset directory and processes all 300 TartanAir frames with an explicitly requested CUDA device.

```bash
uv run pytest -m model
uv run pytest -m benchmark
uv run pytest -m cuda
```

The research baseline targets are a tracking rate of at least 95%, Sim(3)-aligned ATE below 10% of path length, and depth AbsRel at most 0.35 on the TartanAir outdoor Easy sequence. For EuRoC `MH_01_easy`, the targets are a tracking rate of at least 80% and Sim(3)-aligned ATE below 15% of path length. Results vary by model, hardware, and sequence.

## License

This repository is licensed under the MIT License. Pretrained models and datasets remain subject to their respective upstream licenses, and dataset binaries are not redistributed.

- [TartanAir V2](https://tartanair.org/): CC BY 4.0
- [EuRoC MAV](https://www.research-collection.ethz.ch/items/bcaf173e-5dac-484b-bc37-faf97a594f1f): In Copyright - Non-Commercial Use Permitted
- [Depth Anything V2 Metric Outdoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf): Apache-2.0
- [TorchVision SSDLite320 MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html): BSD-3-Clause (COCO dataset terms also apply)
