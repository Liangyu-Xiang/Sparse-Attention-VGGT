# SAF3R peak-memory recheck and 300-frame comparison

Scope: 7Scenes Chess seq-01, uniform sampling, 392×518 input, BF16 autocast; one warmup, three measured complete forwards, no layer timing hooks. Includes fresh per-forward selection and state reset. Pose only; not seven-scene macro metrics.

| Configuration | Seconds ↓ | Allocated peak GiB ↓ | Reserved peak GiB | AUC@5 % ↑ | AUC@15 % ↑ | AUC@30 % ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Dense 128 | 16.613 | 14.359 | 17.555 | 39.695 | 77.367 | 88.466 |
| SAF3R Top-128, release stale indices | 9.658 | 22.453 | 25.660 | 40.037 | 77.457 | 88.501 |
| SAF3R Top-128, retain stale indices | 9.491 | 29.882 | 31.213 | 40.037 | 77.457 | 88.501 |
| SAF3R Top-4 (official default) 128 | 8.123 | 14.612 | 17.566 | 39.459 | 77.207 | 88.379 |
| Dense 300 | 70.530 | 25.090 | 31.449 | 39.500 | 77.147 | 88.317 |
| Fused 20% 300 | 62.714 | 26.532 | 35.213 | 39.429 | 76.191 | 87.654 |
| SAF3R Top-128 300 | oom | — | — | — | — | — |
| SAF3R Top-4 (official default) 300 | 25.033 | 26.480 | 32.973 | 40.068 | 77.287 | 88.365 |

## Interpretation

The previous 29.882 GiB SAF3R Top-128 peak is reproduced with stale indices retained. Releasing all obsolete wrapper references before fresh selection lowers the peak to 22.453 GiB without changing any measured pose fingerprint. This is a benchmark lifecycle correction, not cached selection or a different sparse mask.

At 300 frames Top-128 requires a 310800 × 38400 int32 selection-index tensor (44.46 GiB alone). The unmodified selector fails on the 48-GiB device. The failed allocation is not included in PyTorch peak allocated memory; its recorded pre-failure peak is NOT the memory requirement for a successful run.

Top-4 is an explicitly separate baseline from the repository evaluation YAML, not a replacement result for Top-128. The current Triton version uses the existing blockwise PyTorch selector compatibility path; sparse attention remains SAF3R.

Allocated and allocator-reserved peaks are different metrics. All peaks in the table are medians of the three measured per-forward peaks; per-sample and cold-start values are in the individual result.json files.
