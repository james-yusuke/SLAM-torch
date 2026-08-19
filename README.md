# SLAM-torch

PyTorchによる単眼深度推定と一般物体検出を、ORB/PnP・姿勢グラフ最適化と組み合わせたオフラインSemantic SLAMです。単眼RGB系列から、カメラ軌跡、色付き3D点群、物体の3D位置を生成します。

> [!WARNING]
> これは研究用MVPです。単眼深度のメートル尺度は推定値であり、飛行制御、安全判断、衝突回避にはそのまま使用できません。ROS 2、ライブカメラ、Jetson、実時間保証は初版の対象外です。

## 対応環境

- macOS Apple Silicon: PyTorch MPS、CPUフォールバック
- Linux x86_64: NVIDIA CUDA 12.8、CPUフォールバック
- Windows x86_64: NVIDIA CUDA 12.8、CPUフォールバック
- Python 3.11、uv 0.12以降

Linux/WindowsではuvがPyTorch CUDA 12.8ホイールを選択します。ローカルCUDA Toolkitのコンパイルは不要ですが、[CUDA 12.8互換NVIDIAドライバー](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)が必要です。互換下限はLinux `525.60.13`、Windows `528.33`で、CUDA 12.8には570系以降を推奨します。`slam-torch doctor --require cuda`で実際の状態を確認してください。JetsonのJetPack用PyTorchは通常のx86_64ホイールと異なるため未対応です。

## セットアップ

macOS / Linuxでは次の手順でuvを導入します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra dev
```

Windows PowerShellでは次の手順です。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --frozen --extra dev
```

Apple SiliconではPyPI版PyTorch、Linux/Windows x86_64では`pytorch-cu128`のCUDA 12.8版を、[uvのPyTorch向けインデックス分岐](https://docs.astral.sh/uv/guides/integration/pytorch/)と同じ`uv.lock`から自動選択します。その後、診断と固定アセット取得を実行します。

```bash
uv run slam-torch doctor
uv run slam-torch assets fetch --profile demo --accept-licenses
uv run slam-torch assets status --profile demo
```

`assets fetch`は次の一式をリポジトリ内の`models/`と`data/`へ配置します。初回は公式配布物を約14GB取得するため、展開領域を含め20GB以上の空き容量を推奨します。ダウンロードは`.part`から再開でき、ハッシュ検証と300フレーム抽出の完了後に元アーカイブを削除します。通常のSLAM実行中はネットワークへ接続しません。

- Depth Anything V2 Metric Outdoor Small（Apache-2.0）
- TorchVision SSDLite320 MobileNetV3 COCO
- TartanAir `OldTownFall/Data_easy/P000/lcam_front`先頭300フレーム
- EuRoC `MH_01_easy/cam0`先頭300フレーム

取得元、固定リビジョン、サイズ、チェックサム、ライセンスは`assets.lock.yaml`に記録されています。TartanAirはCC BY 4.0、EuRoCは非商用利用条件のため、データ取得には`--accept-licenses`が必須です。大容量ファイルはGit管理外です。

配置先は`--asset-root`、設定の`assets.root`、`SLAM_TORCH_HOME`、リポジトリルートの順で決まります。旧`SLAM_TORCH_CACHE`も移行用に利用できます。モデルだけを取得する互換コマンドも残しています。

```bash
uv run slam-torch assets fetch --component models
uv run slam-torch models fetch
```

## データセット

### TartanAir V2

標準demoは公式Hugging Face配布の屋外Easy系列から、RGB、深度、姿勢を同じ300インデックスへ揃えて配置します。SLAM入力に使うのはRGBだけで、正解深度と姿勢は評価専用です。

`tartanair-demo.yaml`では合成カメラ領域に対する固定尺度校正として`depth.scale_factor: 0.49`を明示しています。この係数は通常実行中に正解深度を参照せず、`run.yaml`へ保存されます。実ドローンのカメラでは既知距離や別センサーで個別に校正してください。

```bash
uv run slam-torch datasets validate \
  --type tartanair \
  --input data/tartanair/oldtownfall-easy-p000-300
uv run slam-torch run --device auto --config configs/tartanair-demo.yaml
```

次のディレクトリ名を自動認識します。

- RGB: `image_lcam_front`、`image_left`、`lcam_front`
- 深度: `depth_lcam_front`、`depth_left`、`lcam_front_depth`
- 姿勢: `pose_lcam_front.txt`、`pose_left.txt`、`pose.txt`

### EuRoC MAV

標準demoはETH Research CollectionのMachine Hall公式アーカイブから`MH_01_easy`だけを選択抽出します。初版では`cam0`だけを使用し、IMUと右カメラは使用しません。

```bash
uv run slam-torch datasets validate \
  --type euroc \
  --input data/euroc/MH_01_easy-300
uv run slam-torch run --device auto --config configs/euroc-demo.yaml
```

外部の完全系列も従来どおり`run --input PATH`で設定値を上書きして利用できます。

## GPUと精度設定

`--device`は`auto`、`cpu`、`mps`、`cuda`、`cuda:N`を受け付けます。

- `auto`: CUDA → MPS → CPUの順で利用可能なデバイスを選択します。
- 明示したCUDAが使えない場合はエラーにします。
- 実行途中のCUDAメモリ不足ではCPUへ暗黙移行せず、`failure.json`を保存して停止します。
- `balanced`: CUDA深度推定のみFP16 autocast、物体検出はFP32です。
- `deterministic`: FP32、TF32無効の比較・試験用設定です。

CUDAが選ばれたかは`run.yaml`と`metrics.json`の`runtime.resolved`で確認できます。`metrics.json`にはGPU名、Compute Capability、ピークVRAM、モデル別推論時間も記録されます。

NVIDIA実機では次の診断で、CUDA 12.8ビルド、ドライバー、Tensor演算に加え、両モデルのパラメータと入力がCUDA上にあることを検査します。失敗時は非ゼロで終了します。

```bash
uv run slam-torch doctor --require cuda --model-smoke
uv run slam-torch run --device cuda --config configs/tartanair-demo.yaml
```

## 成果物

各実行は`runs/<UTC時刻>-<ID>/`へ保存されます。

- `trajectory.tum`: TUM形式の推定カメラ軌跡
- `groundtruth.tum`: データセットに正解姿勢がある場合のみ
- `map.ply`: 色付き3D点群
- `objects.json`: 物体クラス、信頼度、3D中心、粗いAABB、動的状態
- `metrics.json`: 追跡率、FPS、深度誤差、ATE/RPE、推論時間、デバイス情報
- `run.yaml`: 解決済み設定と再現情報
- `failure.json`: 初期化、データ、CUDA OOM等で失敗した場合

```bash
uv run slam-torch evaluate --run runs/<run-id>
uv run slam-torch visualize --run runs/<run-id>
```

座標は最初のカメラを原点とする`T_world_camera`です。画像座標はx右、y下、カメラz前方のOpenCV規約です。

## 処理構成

1. データセット固有形式を較正済みRGB `Frame`へ変換
2. 全フレームでORB特徴抽出・記述子照合
3. キーフレーム深度から3D特徴を作り、PnP RANSACで姿勢推定
4. 追跡失敗時に過去キーフレームから再ローカライズ
5. 非近傍キーフレームを幾何検証し、SE(3)姿勢グラフを最適化
6. 最適化済み姿勢と深度から点群・物体地図を再構築

人物・車両などの動的クラスは恒久点群とSLAM用3D特徴から除外し、`objects.json`には最終観測位置を持つ動的観測として残します。

## 開発とテスト

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

実モデル、公式データ、CUDA GPUを必要とするテストは通常のCPU CIから分離しています。self-hosted CUDAワークフローは永続アセット領域を再利用し、TartanAir 300フレームを明示CUDAで処理します。

```bash
uv run pytest -m model
uv run pytest -m benchmark
uv run pytest -m cuda
```

研究ベースラインの目標値は、TartanAir屋外Easyで追跡率95%以上・Sim(3)整列ATE/経路長10%以下・深度AbsRel 0.35以下、EuRoC `MH_01_easy`で追跡率80%以上・Sim(3)整列ATE/経路長15%以下です。モデル、ハードウェア、取得した系列により結果は変わります。

## ライセンス

本リポジトリはMIT Licenseです。学習済みモデルとデータセットには各配布元のライセンスが適用され、データ本体は再配布しません。

- [TartanAir V2](https://tartanair.org/): CC BY 4.0
- [EuRoC MAV](https://www.research-collection.ethz.ch/items/bcaf173e-5dac-484b-bc37-faf97a594f1f): In Copyright - Non-Commercial Use Permitted
- [Depth Anything V2 Metric Outdoor Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf): Apache-2.0
- [TorchVision SSDLite320 MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html): BSD-3-Clause（COCOデータセット条件も適用）
