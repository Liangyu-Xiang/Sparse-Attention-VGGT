# Paired 7Scenes128: all 24 global layers

Completed scenes (7): chess, fire, heads, office, pumpkin, redkitchen, stairs.
One warmup, three measured repeats per method/scene; tables use per-scene timing medians, then scene macro means.
Check result.json input metadata for exact repeats, input indices, checkpoint configuration and settings.
Same checkpoint/frames/GPU per scene, BF16 autocast. Frame attention remains dense.
All initialization, Value clustering, routing, online probe QK/output evaluation, selection, history and fallback costs are INCLUDED.
SAF3R is the full profiled method, correspondence topk=128 per frame, not a global 128-key mask.
Fixed10 has block-rounding; adaptive begins at 10% but may expand to 100%. Density includes mandatory special keys.
AUC is a pose metric, not depth/reconstruction accuracy. Seven selected sequences are not an entire-dataset generalization claim.

| Method | Global24 ms | Whole model s | AUC30 % | AUC5 % | Actual edges % | Dense heads % |
|---|---:|---:|---:|---:|---:|---:|
| dense | 11332.209 | 18.985628 | 84.293 | 36.647 | 100.000 | — |
| saf3r | 4084.840 | 11.883286 | 84.576 | 38.137 | profile-dependent | — |
| fixed10 | 4448.753 | 12.771357 | 74.205 | 26.293 | 10.038 | 0.000 |
| adaptive | 8904.548 | 16.756438 | 83.299 | 34.287 | 38.479 | 2.716 |

## Speed comparisons

Ratios below divide macro-mean times; >1 means the candidate is faster.

| Candidate | Global vs Dense | Whole vs Dense | Global vs SAF3R | Whole vs SAF3R |
|---|---:|---:|---:|---:|
| fixed10 | 2.547x | 1.487x | 0.918x | 0.930x |
| adaptive | 1.273x | 1.133x | 0.459x | 0.709x |

## Per scene

| Scene | Method | Global24 ms | Whole s | AUC30 % | AUC5 % |
|---|---|---:|---:|---:|---:|
| chess | dense | 11545.514 | 19.627781 | 88.466 | 39.695 |
| chess | saf3r | 4155.436 | 12.151193 | 88.501 | 40.037 |
| chess | fixed10 | 4303.847 | 13.448558 | 85.703 | 37.052 |
| chess | adaptive | 8854.258 | 17.102163 | 88.225 | 38.986 |
| fire | dense | 11086.769 | 18.962199 | 94.523 | 70.017 |
| fire | saf3r | 4022.420 | 11.967704 | 94.626 | 70.753 |
| fire | fixed10 | 4418.575 | 12.796643 | 91.927 | 60.827 |
| fire | adaptive | 8756.753 | 16.779240 | 94.065 | 68.024 |
| heads | dense | 11530.535 | 19.541712 | 94.007 | 67.281 |
| heads | saf3r | 4152.828 | 12.474663 | 94.133 | 68.172 |
| heads | fixed10 | 4438.735 | 13.434243 | 88.054 | 41.713 |
| heads | adaptive | 8917.167 | 17.236384 | 92.426 | 59.230 |
| office | dense | 11291.319 | 19.191860 | 77.134 | 23.679 |
| office | saf3r | 4062.708 | 12.229375 | 75.876 | 22.982 |
| office | fixed10 | 4358.072 | 13.047182 | 32.156 | 1.708 |
| office | adaptive | 8979.459 | 17.272750 | 77.105 | 22.960 |
| pumpkin | dense | 11242.241 | 19.160061 | 75.208 | 18.664 |
| pumpkin | saf3r | 4039.712 | 12.206168 | 74.909 | 18.514 |
| pumpkin | fixed10 | 4332.983 | 12.923761 | 72.314 | 14.692 |
| pumpkin | adaptive | 8945.892 | 17.264297 | 74.516 | 17.173 |
| redkitchen | dense | 11485.883 | 19.485762 | 79.768 | 21.499 |
| redkitchen | saf3r | 4100.116 | 12.383960 | 80.180 | 21.813 |
| redkitchen | fixed10 | 4804.718 | 13.662743 | 77.468 | 21.073 |
| redkitchen | adaptive | 9224.513 | 17.560427 | 79.373 | 21.324 |
| stairs | dense | 11143.201 | 16.930020 | 80.942 | 15.691 |
| stairs | saf3r | 4060.663 | 9.769937 | 83.809 | 24.688 |
| stairs | fixed10 | 4484.342 | 10.086370 | 71.811 | 6.986 |
| stairs | adaptive | 8653.796 | 14.079805 | 77.383 | 12.315 |

## Interpretation / safeguards

- C*S is not used as an edge fraction. Attention-active counts and actual mixtures are saved per head.
- Budget-selection probes measure current sparse-prefix inputs; they do not establish closeness to the original Dense model at every later layer.
- Passing sampled output-error thresholds does not certify all queries or final poses.
- Dense fallback is exact at that head current input, not an assertion of a sparse speedup.
- Prior legacy AUC30 40.393% is an archived result; new comparisons against Dense/SAF3R use this matched rerun.
- Shared Value hierarchy is retained, but masks/history/budgets are now head-specific and query-block-specific.
- Block pooling/quantization remains an approximation. Replacing the old scalar path does not prove the new pipeline beats optimized SAF3R.
- Full masks are not stored by default; per-scene/layer/head decisions, match indices, densities and history lifecycle counters are retained.
- Source/configuration and separate operator checks are documented in docs/experiments/adaptive_block_value.md.
