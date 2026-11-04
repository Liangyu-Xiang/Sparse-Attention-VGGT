# Shared Value Loss + Reduced Cluster Count — Server Migration

**Authoritative Git branch:** `codex/shared-value-cluster-scaling-20261104`

This branch contains the tested shared Cluster–Grid Value-distortion loss, 256-query sampled-history probability completion, four-state budget selection, and **20% / 10% / 5% flat Cluster** configurations. The selected-Key Retention budget remains **20% in all three configurations**. The existing `main` branch has not been overwritten.

## Restore code on another server

Fresh clone:

```bash
git clone --single-branch --branch codex/shared-value-cluster-scaling-20261104 \
  https://github.com/Liangyu-Xiang/Sparse-Attention-VGGT.git
cd Sparse-Attention-VGGT
git status --short
git rev-parse --short HEAD
```

If a local copy already contains uncommitted changes, **do not switch or reset that checkout**. Add an isolated worktree:

```bash
cd /path/to/existing/Sparse-Attention-VGGT
git fetch origin codex/shared-value-cluster-scaling-20261104
git worktree add --detach ../Sparse-Attention-VGGT-shared-value FETCH_HEAD
cd ../Sparse-Attention-VGGT-shared-value
```

## Files used by the experiments

- `saf3r/patch/reuse_residual_exploration.py` — install/forward integration, budget and seed handling.
- `saf3r/triton/reuse_shared_value_states.py` — query-shared Cluster–Grid Value distortion statistics; sparse per-query probability corrections.
- `saf3r/patch/reuse_cluster_seed.py` — deterministic unique first-layer Keys when Cluster count is smaller than Key budget.
- `saf3r/triton/reuse_drop_moment.py` — population bins including width up to 512.
- `saf3r/patch/reuse_sampled_history.py` — 256-query sampled history and uniform grid probability.
- `saf3r/patch/reuse_window_graph.py` — CUDA Graph and selector with unit selected-Key bias.
- `configs/evaluation/reuse_shared_value_loss_256.json` — nominal ~20% Cluster count.
- `configs/evaluation/reuse_shared_value_loss_clusters10.json` — 10% clusters.
- `configs/evaluation/reuse_shared_value_loss_clusters5.json` — 5% clusters.
- `scripts/validate_shared_value_loss.py` — numerical and CUDA mask/Graph checks.
- `scripts/evaluate_zdrop_history_states.py` — individual geometric quality and forward-time test.
- `scripts/evaluate_shared_value_six_scenes.py` — six-scene experiment harness.
- `scripts/profile_shared_value_cluster_scaling.py` — portable, warmed forward and CUDA Event stage profiler.

## Important: datasets and checkpoint are NOT in Git

The previous machine used external persistent files, including:

- Model checkpoint: `/mnt/rfs/users/xiangliangyu/checkpoints/vggt-1b-model.pt` (about 4.7 GB), linked into the repository as `checkpoints/vggt/model_tracker_fixed_e20.pt`.
- 7Scenes: `/mnt/rfs/users/xiangliangyu/datasets/7scenes`, linked at `workspace/benchmark_dataset/7scenes`.
- ETH3D: `/mnt/rfs/users/xiangliangyu/datasets/eth3d`, linked at `workspace/benchmark_dataset/eth3d`.

If the new server mounts the same persistent `/mnt/rfs` volume, these files may already be available. If not, transfer or remount them independently. **A Git clone by itself will not supply the checkpoint or datasets.** Restore an appropriate Python/CUDA environment based on the repository's `pyproject.toml` and tested model dependencies.

Verify the required paths before running:

```bash
test -f checkpoints/vggt/model_tracker_fixed_e20.pt && echo CHECKPOINT_OK
test -d workspace/benchmark_dataset/7scenes && echo 7SCENES_OK
test -d workspace/benchmark_dataset/eth3d && echo ETH3D_OK
python -m scripts.validate_shared_value_loss
```

## Run a single test

From the repository root, with a free, exclusively held CUDA GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_zdrop_history_states \
  --dataset 7scenes --scene heads --frames 128 \
  --method zdrop_history_states \
  --config configs/evaluation/reuse_shared_value_loss_clusters10.json \
  --output-dir outputs/new_machine_clusters10_heads \
  --warmup 1 --repeats 3
```

Profiling the same algorithm at larger frame lengths (this script does **not** calculate reconstruction quality):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_shared_value_cluster_scaling \
  --frames 160 --clusters 10 --repeats 3 \
  --output outputs/new_machine_cluster_scaling/10_160
```

Change `--clusters` to `20` or `5` to match the corresponding fixed configurations. Do not run other CUDA programs on the timing GPU.

## Previous experimental results

- `outputs/shared_value_loss_six_20261104/report.md`: original 6-scene quality and old-versus-new paired speed.
- `outputs/shared_value_cluster_scaling_20261104/report.md`: 128, 160 and 192-frame timing, CUDA Event stage breakdown, memory limits, and quality tradeoffs.

The detailed original `result.json`, `layers.json`, CUDA profiling records and exclusive-GPU monitor data were also saved on the previous machine at:

`/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/shared_value_cluster_scaling_20261104/`

These bulky raw artifacts are **not** all uploaded to Git. Transfer them separately if the new server does not share the same persistent volume.

**Caveat:** reduced Cluster count has not preserved geometric quality on all tested sequences. 10% is faster on 128/160-frame tests, 5% has a memory advantage at 192 frames but has a larger ETH3D quality regression; see reports. Do not present this experimental branch as a validated accuracy-equivalent replacement for Dense/SAF3R.
