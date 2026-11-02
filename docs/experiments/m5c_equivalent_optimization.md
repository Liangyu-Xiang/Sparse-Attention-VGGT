# M5c implementation optimization, 2026-11-02

This is an isolated implementation of the repaired RMS Drop + historical-state cleanup. It does not replace the unrelated newer experimental implementations in the main working tree. The mathematical estimator, 20% retention, adaptive per-layer clustering, historical Rep rule, full History state, Exact eligibility, 16 budget iterations, and probability-mass transfer are unchanged.

Changes:

- Fuse historical selected-QK reconstruction, M5c correction and probe summation, with separate per-head outputs followed by the original Torch head mean.
- Keep compact FP32 calibration statistics on GPU, releasing each old window before caching the corresponding new window. No CPU round-trip. Remove the unread cached special-probability field.
- Fuse current calibration exponentiation and grid accumulation. Keep the Attention second QK because its grid statistics are still consumed.

Create a runnable fork from the verified cleanup source with:

```bash
python scripts/setup_m5c_equivalent.py --destination /tmp/zdrop_m5c_local
```

The default source is `/tmp/zdrop_history_cleanup_20261102`, checked against its frozen manifest. Choose a new destination; existing directories are rejected. Datasets, checkpoints and the existing project environment remain required. The setup script also corrects the inherited cache-device metadata label and includes the optimized modules in future evaluator source snapshots. The timed fork predates those metadata-only fixes; its actual cache is GPU-resident, as recorded in `history_calibration_gpu_bytes`.

The changed modules are `saf3r/patch/reuse_m5c_equivalent.py` and `saf3r/triton/reuse_m5c_equivalent.py`. Tests: `scripts/validate_m5c_equivalent.py` and `scripts/validate_m5c_real128.py`. Timing: `scripts/run_m5c_equivalent128.py`, with `scripts/profile_m5c_equivalent128.py`. These diagnostic scripts preserve the local snapshot/output paths and reject existing output directories where applicable.

Measured on heads/128 frames, one RTX5090 D, sequential exclusive GPU lease, 1 warmup + 3 complete forward measurements per implementation: median 53.63070 -> 33.10490 s, 1.6200x relative to the old cleanup (38.27% time reduction). Peak allocated 19.61151 -> 19.06346 GiB; peak reserved 23.04297 -> 24.98047 GiB. The device cache is 1.52590 GiB. No Dense or SAF3R rerun; contemporaneous Dense-relative speed N/A.

Twelve synthetic FP32/BF16 cases pass probability/cache/normalization, History gates and four-budget state-choice checks. Real 128-frame first/last-window shadow checks produce prior relative L2 below 7.90e-8, but 7 History candidate threshold decisions differ near floating-point ties. This is mathematical equivalence, not bitwise equivalence or verified preservation of quality. No GT tuning or quality evaluation was performed; AUC@3, AUC@30, ACC, Comp, CD and F1 are all N/A because this is a numerical/timing diagnostic.

Full raw timing, source archives/hashes, output differences and profile: `outputs/m5c_equivalent_heads128_20261102/report.md`. Numerical microbenchmark: `outputs/m5c_equivalent_20261102`. Real-input checks: `outputs/m5c_equivalent_real128_20261102`.
