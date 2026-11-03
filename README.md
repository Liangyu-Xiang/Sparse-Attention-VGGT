> History-fix evaluation: repaired single-probe heads128 warmed median **14.634211 s**, four-probe **15.532334 s**. Complete quality results, limitations, and stage timings: [GPU evaluation report](outputs/gpt6_history_fix_eval_20261103/report.md). Historical failed12.83s snapshot remains archived separately.

> Experimental fastest sparse snapshot: heads128 warmed median **12.832573 s**; **severe quality regression**. See [results and limitations](outputs/fastest_single_probe_snapshot_20261103/report.md).

<div align="center">

<h1>SAF3R: Dynamic Sparse Attention for Feed-Forward 3D Reconstruction Transformers</h1>

[![Paper](https://img.shields.io/static/v1?label=Paper&message=arXiv-2607.03612&color=red&logo=arxiv)](https://arxiv.org/abs/2607.03612)

</div>

SAF3R is a training-free dynamic sparse attention framework that accelerates existing feed-forward 3D reconstruction models, such as VGGT, by reducing the computational cost of global attention. It exploits head-wise sparsity heterogeneity to replace each full global attention head with the sparse attention kernel that best matches its attention pattern.

<p align="center">
  <img src="assets/teaser.png" alt="Teaser" width="80%">
</p>

### What's in this repo
* **Visualization tools** for analyzing head-wise global attention patterns in feed-forward 3D reconstruction models. Currently supported models include three offline models (VGGT, Pi3, and DA3) and one streaming model (StreamVGGT).
* **SAF3R offline profiling code** that automatically assigns each global attention head to one of four predefined sparse attention patterns.
* **SAF3R inference patches** that enable dynamic sparse attention inference for VGGT, Pi3, and DA3.
* **Benchmark and evaluation tools** for evaluating model performance and efficiency. Currently supported benchmarks include DA3-Bench, Co3D-v2, RealEstate10K, and ScanNet.

## Table of Contents
- [Installation](#installation)
- [Analysis Tools](#analysis-tools)
- [Model Inference](#model-inference)
  - [Running Inference Demo](#running-inference-demo)
  - [Minimal Code Snippet (for SAF3R-VGGT)](#minimal-code-snippet-for-saf3r-vggt)
- [Evaluation Benchmarks](#evaluation-benchmarks)
  - [Supported Benchmarks & Datasets](#supported-benchmarks--datasets)
  - [Datasets Preparation](#datasets-preparation)
  - [Running Evaluation](#running-evaluation)
  - [Benchmarking Efficiency](#benchmarking-efficiency)
- [Profiling Attention Heads](#profiling-attention-heads)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)


## Installation

```bash
conda create -n saf3r python=3.10 -y
conda activate saf3r
git clone https://github.com/jndeng/SAF3R
cd SAF3R
pip install -e .
```


## Analysis Tools

We provide visualization and analysis tools for frame/global attention patterns for each supported model under `tools/`. For each model, the corresponding tool is implemented as a standalone Jupyter notebook and can be run independently. The two main use cases are shown below.

<p align="center">
  <img src="assets/vggt_attn_img_overlay.png" height="400" alt="Attention Distribution Overlay" />
  &emsp;&emsp;
  <img src="assets/vggt_attn_map_heads.png" height="400" alt="Attention Maps of Layer Heads" />
  <br>
  <em>Left: Semi-interactive visualization of the attention distribution for each selected query token (blue box), overlaid on the image. Right: Attention maps of all heads in a specific layer.</em>
</p>


## Model Inference

We provide inference example code for using SAF3R on different 3R models.

> [!NOTE]
> Checkpoints will be automatically downloaded to the local cache directory `checkpoints/` during the first run.
> They can also be manually downloaded from [VGGT](https://huggingface.co/facebook/VGGT_tracker_fixed), [Pi3](https://huggingface.co/yyfz233/Pi3), [DA3-GIANT](https://huggingface.co/depth-anything/DA3-GIANT), and [StreamVGGT](https://huggingface.co/lch01/StreamVGGT).

### Running Inference Demo

To run a demo inference script using a specific model on your image directories:
```bash
python scripts/inference_demo.py --model vggt --data_dir data/courthouse
```
Supported options for `--model` are `vggt`, `pi3`, and `da3`. By default, the predicted point clouds will be exported under `tmp_plots/` as `.ply` files and browser-viewable interactive `.html` visualizations.

> [!NOTE]
> The first run may take longer due to JIT compilation of the custom Triton kernels. The compiled kernels are cached and reused in subsequent runs.

<p align="center">
  <img src="assets/demo_vis.png" alt="Vis" width="80%">
  <br>
  <em>Example interactive HTML visualization results.</em>
</p>

### Minimal Code Snippet (for SAF3R-VGGT)

<details>
<summary>Show code</summary>

```python
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model, infer_model

# 1. Configure the model and SAF3R patch settings
model_cfg = Dict(
    name="VGGT", 
    dpt_only=True, 
    ckpt_path="checkpoints/vggt/model_tracker_fixed_e20.pt",
    sparse_config_path="configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json",
    patch_module=Dict(
        type="headsparse",
        topk_mode="token",
        lazy_dino_topk=True,
        topk=4
    )
)

# 2. Build and automatically patch the model with SAF3R kernels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(model_cfg, device)

# 3. Setup input data (image file paths)
scene_data = Dict(
    image_files=[
        "data/courthouse/000000.png",
        "data/courthouse/000001.png",
        ...
    ]
)

# 4. Perform dynamic sparse attention inference
pred_data, stats = infer_model(model, model_cfg, scene_data)

# Extract unified predictions
extrinsics = pred_data.extrinsics     # [N, 3, 4]
intrinsics = pred_data.intrinsics     # [N, 3, 3]
depth      = pred_data.depth          # [N, H, W]
depth_conf = pred_data.conf           # [N, H, W]
```
</details>


## Evaluation Benchmarks

We provide evaluation code for multiple tasks and benchmarks.

### Supported Benchmarks & Datasets
* DA3-Bench Datasets (7Scenes, ETH3D, ScanNet++, HiRoom, DTU64, DTU)
  - Camera pose estimation
  - Video depth estimation
  - 3D point-cloud reconstruction
* Co3D-v2
  - Camera pose estimation
* RealEstate10K
  - Camera pose estimation
* ScanNet (v2)
  - Camera pose estimation
  - 3D point-cloud reconstruction

### Datasets Preparation
Please follow the corresponding instructions to prepare each dataset.
* DA3-Bench (7Scenes, ETH3D, ScanNet++, HiRoom, DTU64, DTU)
    - Follow the [DA3-Bench dataset download instructions](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md#-dataset-download) to download the datasets. This should create the `workspace/benchmark_dataset/` directory under the project root.
* Co3D (v2)
    - Follow the [VGGT preparation instructions](https://github.com/facebookresearch/vggt/tree/evaluation/evaluation#dataset-preparation) to prepare the dataset and place it under `workspace/benchmark_dataset/`.
* RealEstate10K
    - Follow the [Pi3 preparation scripts](https://github.com/yyfz/Pi3/blob/evaluation/datasets/preprocess/prepare_re10k.sh) to prepare the dataset and place it under `workspace/benchmark_dataset/`.
* ScanNet (v2)
    - Follow the [ScanNet instructions](http://www.scan-net.org/ScanNet/) to download the dataset and place it under `workspace/benchmark_dataset/`. The list of the 50 evaluation scenes can be found [here](https://github.com/mystorm16/FastVGGT/blob/main/eval/scannet_50.yaml).

The downloaded datasets should be organized under `workspace/benchmark_dataset/` as follows:


<details>
<summary>Show dataset structure</summary>

```text
workspace/benchmark_dataset/
├── 7scenes/
│   └── 7Scenes/
│       ├── chess/
│       └── ...
├── eth3d/
│   ├── courtyard/
│   ├── electro/
│   └── ...
├── scannetpp/
│   ├── 09c1414f1b/
│   └── ...
├── hiroom/
│   ├── data/
│   ├── fused_pcd/
│   └── selected_scene_list_val.txt
├── dtu/
│   ├── Rectified/
│   ├── Cameras/
│   ├── Points/
│   ├── SampleSet/
│   └── depth_raw/
├── dtu64/
│   ├── Cameras/
│   ├── scan105/
│   └── ...
├── co3dv2/
│   ├── vggt_testset
│   │   ├── apple/
│   │   └── ...
│   └── vggt_anno
│       ├── apple_test.jgz
│       └── ...
├── realestate10k/
│   ├── data/re10k
│   │   ├── 005dd9a58df1ba3c/
│   │   └── ...
│   └── datasets/seq-id-maps
│       └── Re10K_relpose_seq-id-map_seed42.json
└── scannetv2/
    ├── scans/
    │   ├── scene0000_00/
    │   ├── scene0013_02/
    │   └── ...
    └── scannet_50.yaml
```
</details>


### Running Evaluation
Run the unified launcher with the desired configuration file:
```bash
bash scripts/evaluate.sh eval_saf3r_vggt
```

### Benchmarking Efficiency
To measure inference latency and peak memory usage across different sequences:
```bash
bash scripts/benchmark_efficiency.sh eval_saf3r_vggt 300
```
This runs the efficiency benchmark using the specified configuration file and sequence length.


## Profiling Attention Heads

We provide profiled global-attention head configurations under `configs/sparse_attn/`. 

To generate these profiling results from scratch:

1. Download the calibration dataset (e.g., ETH3D) following the instructions in [DA3-bench](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md).
2. Run the profiling script using the desired configuration file under `configs/profile/`:
   ```bash
   bash scripts/profile.sh profile_saf3r_vggt
   ```


## Local History80 Research Extension

The currently selected local sparse variant is **original History80 with residual
cluster-population compensation**. See the [current implementation and reproduction guide](docs/experiments/current_history80_population.md).
The middle-ratio calibration candidate is experimental and is **not** the selected version.
This extension is separate from official SAF3R;no lossless-quality or deployment-speed claim is implied.

## Acknowledgements
This repository builds upon several excellent open-source projects, including [VGGT](https://github.com/facebookresearch/vggt), [Pi3](https://github.com/yyfz/Pi3), [DA3](https://github.com/ByteDance-Seed/depth-anything-3), [StreamVGGT](https://github.com/wzzheng/streamvggt), [LingBot-Map](https://github.com/robbyant/lingbot-map), [FastVGGT](https://github.com/mystorm16/FastVGGT), [SparseVGGT](https://github.com/brianwang00001/sparse-vggt), and [Speed3R](https://github.com/Visual-AI/speed3r). We sincerely thank the authors and contributors for making their code publicly available.


## Citation
If you find SAF3R useful for your research or project, please consider citing:

```bibtex
@article{deng2026saf3r,
  title={SAF3R: Dynamic Sparse Attention for Feed-Forward 3D Reconstruction Transformers},
  author={Deng, Jianing and Li, Yuanzhe and Wang, Jialu and Wang, Song and Chen, Tianlong and Yang, Huanrui and Hu, Jingtong},
  journal={arXiv preprint arXiv:2607.03612},
  year={2026}
}
```

## Current head-mean history implementation (2026-11-02)

The active implementation uses head-mean cross-layer conditional statistics and a single selected-edge QK pass. Every layer still updates Value clusters and states; total retention is capped at20%. Probability-mass compensation remains enabled by default. The no-compensation version is a separate ablation and is not a quality-preserving replacement.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/evaluate_zdrop_history_states.py \
  --dataset 7scenes --method zdrop_history_states --scene heads --frames 128 \
  --config configs/evaluation/reuse_zdrop_history_states.json \
  --output-dir /absolute/path/to/fresh/results
```

The original checkpoint and dataset workspace layout are required as before. Dense and SAF3R reference scores are reused, not rerun. Current code is byte-identical to the evaluated frozen head-mean source; unrelated working-tree edits are not included.

Results and limitations:
- `outputs/headmean_single_qk_quality_20261102/report.md`: full7Scenes and ETH3D quality, runtime and memory; ETH3D quality regresses and full-forward speed does not improve.
- `outputs/no_mass_compensation_quality_v2_20261102/report.md`: removing mass compensation hurts reconstruction, particularly ETH3D; preserve as an ablation.
- `outputs/no_mass_compensation_profile_20261102/report.md`: exclusive timing; no double-counting of parent/child scopes.
- `docs/experiments/headmean_single_qk.md` and `docs/experiments/no_mass_compensation.md`: formulas, implementation, validation and scope.

Source archives, raw per-scene six-metric JSON, timing batches, memory and macro tables are retained. No paired Dense speedup is claimed for these batches. Independent estimator/execution causal ablations are complete: `outputs/headmean_execution_causal_20261102/report.md`. They locate reconstruction regressions in the grid-grouped execution path; head-mean alone improves reconstruction but changes some pose metrics. The current default is retained for reproducibility, not advertised as an accepted speed/quality improvement.

Large raw timing traces are stored as `.json.gz` / `.csv.gz`; decompress a copy with `gunzip -k` when rerunning legacy report generators. Quality metrics remain plain JSON and CSV.

For the independently measured cause of quality regression and slowdown, see `docs/experiments/headmean_execution_causal.md`. Both variants and adverse results are retained.
