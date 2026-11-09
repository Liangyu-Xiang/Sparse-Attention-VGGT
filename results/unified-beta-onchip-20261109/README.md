# Unified-Beta FP32 on-chip history reduction

Experimental snapshot based on 66bbffe9. The ScanNet50 500-frame forward diagnostic averaged 23.971 s across four H800 scenes, down from 26.049 s for the preceding sparse candidate. The 19 full-length scenes in the first-20 quality campaign averaged 23.975 s. The corresponding historical Dense ratio was 4.892×. Paired Dense speedup is unavailable because there was no contemporaneous paired run. The quality gate failed on both pose AUC metrics. See the diagnostic report and the first-20 acceptance report in this directory.

Build the CUDA Welford/RoPE kernel on Hopper before enabling the supplied settings.

```bash
nvcc -O3 -arch=sm_90 --shared -Xcompiler -fPIC saf3r/triton/norm_rope.cu -o saf3r/triton/norm_rope.so
set -a
source results/unified-beta-onchip-20261109/config.env
set +a
python scripts/profile_scannet500_beta_local.py --worker --diagnostic --dataset scannet50 \
  --scene scene0000_00 --method fast_subgroup --frames 500 \
  --data /path/to/benchmark-data --checkpoint /path/to/model_tracker_fixed_e20.pt \
  --checkpoint-sha256 b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969 \
  --output /path/to/results
```

For the first-20 quality run, use `scripts/dispatch_goal_quality.py` with the same environment. Its checkpoint and dataset defaults point to the original experiment layout. It records pose and unposed reconstruction.
