# 7Scenes, 128 frames: hierarchical Value + validated Attention

IMPORTANT: the full-layer 10% scheme has a large accuracy regression (AUC30
40.393% vs dense 84.293%). It is not currently a usable full-layer acceleration
result. Only zero-based layer 14 receives 1 or 3 Attention edges/query; layer 15
returns to Value-only and does not reuse the new history.

Independent retained diagnostics: [four tables and dense-reference errors](diagnostics/report.md).
All seven 128-frame diagnostic predictions were checked against the formal
saved predictions. Small Chess frames 0,20 raw snapshots (24 layers, about
1.2 GiB) are in diagnostics/chess_frames_0_20/chess/. Diagnostic timings and
dense-oracle work are NOT included in the formal timing table below.

Paired inputs; natural global L1--L24 (zero-based 0--23) change.
All other global and frame layers remain dense. One layer-shared selection/history,
head-mean Q/K/V for selection; actual per-head attention outputs are not averaged.
Hierarchy: 15% fine clusters, coarse ceil(N_patch/128), two local iterations,
Three coarse Lloyd steps; initialization=uniform. Uniform bootstrap is Value-only.
Final total edge density is 10% INCLUDING all mandatory special keys.
SAF3R retains the original profile and is not matched to a 10% budget.

SAF3R correspondence TopK=128 PER FRAME (lazy QK selector); this is
effective in the full 24-layer run, unlike the legacy L3--L9-only run.

Full inference includes the lightweight first-sparse-layer initialization.
No L2 global PAM time is removed, because no PAM initialization is used.
Timings: scene macro mean of per-scene medians, one warmup/three repeats
(actual settings are recorded in each input.json). AUC: scene macro mean (%).

| Method | Scenes | Selected global layers ms | Overall s | Peak GiB | AUC30 % | AUC15 % | AUC5 % | AUC3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7 | 11319.9625 | 19.1419 | 14.3488 | 84.2927 | 70.5458 | 36.6465 | 23.0420 |
| saf3r | 7 | 4080.7610 | 11.9958 | 22.4425 | 84.5763 | 71.2942 | 38.1370 | 24.1687 |
| hierarchy | 7 | 7733.9335 | 16.2510 | 15.1644 | 40.3932 | 25.4967 | 9.4544 | 5.3062 |

## Per-scene comparison

| Scene | Method | Selected global layers ms | Overall s | AUC30 % | AUC5 % |
|---|---|---:|---:|---:|---:|
| chess | dense | 11502.865 | 19.386 | 88.466 | 39.695 |
| chess | saf3r | 4158.724 | 12.055 | 88.501 | 40.037 |
| chess | hierarchy | 7761.222 | 16.582 | 37.854 | 1.100 |
| fire | dense | 11095.078 | 18.836 | 94.523 | 70.017 |
| fire | saf3r | 4021.386 | 11.940 | 94.626 | 70.753 |
| fire | hierarchy | 7790.751 | 16.037 | 1.853 | 0.002 |
| heads | dense | 11513.114 | 19.389 | 94.007 | 67.281 |
| heads | saf3r | 4155.293 | 12.100 | 94.133 | 68.172 |
| heads | hierarchy | 7764.266 | 16.368 | 90.221 | 50.167 |
| office | dense | 11289.322 | 19.265 | 77.134 | 23.679 |
| office | saf3r | 4067.313 | 12.051 | 75.876 | 22.982 |
| office | hierarchy | 7782.751 | 16.234 | 3.631 | 0.084 |
| pumpkin | dense | 11209.423 | 18.920 | 75.208 | 18.664 |
| pumpkin | saf3r | 4038.176 | 11.912 | 74.909 | 18.514 |
| pumpkin | hierarchy | 7711.630 | 16.276 | 70.584 | 11.951 |
| redkitchen | dense | 11488.732 | 19.314 | 79.768 | 21.499 |
| redkitchen | saf3r | 4103.653 | 12.039 | 80.180 | 21.813 |
| redkitchen | hierarchy | 7683.565 | 16.280 | 14.978 | 0.145 |
| stairs | dense | 11141.203 | 18.883 | 80.942 | 15.691 |
| stairs | saf3r | 4020.782 | 11.874 | 83.809 | 24.688 |
| stairs | hierarchy | 7643.349 | 15.979 | 63.633 | 2.731 |

## Layer-shared adaptive budgets (scene macro means)

| Layer (zero-based) | Alpha | A patch keys | V patch keys | Total keys | Connection density % |
|---|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 1 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 2 | 0.000001 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 3 | 0.000001 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 4 | 0.000001 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 5 | 0.000003 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 6 | 0.000001 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 7 | 0.000002 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 8 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 9 | 0.000002 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 10 | 0.000003 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 11 | 0.000015 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 12 | 0.000014 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 13 | 0.000013 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 14 | 0.000124 | 1.857143 | 12683.142857 | 13325.000000 | 10.000150 |
| 15 | 0.000010 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 16 | 0.000001 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 17 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 18 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 19 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 20 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 21 | 0.000002 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 22 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |
| 23 | 0.000000 | 0.000000 | 12685.000000 | 13325.000000 | 10.000150 |

## Interpretation and limitations

- This validates pose AUC only; no GT depth or reconstruction claims.
- This is an algorithmic/correctness prototype, not a fused production selector.
  Sorting, integer allocation, scalar synchronizations and hierarchy regrouping
  are included in timing. A low connection density does not guarantee acceleration.
- Attention history is validated from the SAME computed exact sparse logits.
  There is no second history QK evaluation and no old weights in state.
- A source-description boundary case is handled explicitly: after exhausting
  fine representatives, exploration uses unique deterministic member tokens
  from the SAME coarse group, preserving token-capacity budgets and uniqueness.
- Config: configs/evaluation/eval_hierarchical_value_vggt.yaml.
- Method details: docs/experiments/hierarchical_value_attention.md.
- Tests: tests/test_hierarchical_value_attention.py (including CUDA/reference
  logits/output agreement, uniqueness/capacity/coverage/anchors, reset, validation).
