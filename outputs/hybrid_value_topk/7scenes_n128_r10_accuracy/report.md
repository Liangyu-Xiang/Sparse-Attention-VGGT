# 7Scenes128: representative Value / exact QK Top-K hybrid

SINGLE COLD FORWARD PER SCENE: timings INCLUDE first-use compilation, not warmed formal speed measurements. Primary objective is seven-scene accuracy validation.
Zero-based global layers: 0–2 Dense, 3–9 Value, 10–16 full-QK Top-K, 17–23 Value. Frame attention unchanged.
Value keeps real representative K/V, not Key blocks. One shared head-mean-V clustering, GPU lightweight L2 initialization and deterministic centroid-nearest member updates.
Shared hierarchy is maintained in 10–16 as well, so layer 17 inherits layer 16, not stale layer 9.
All specials mandatory and COUNT in fixed 10% total budget. TopK selects the remaining patch slots per head/query.
All initialization, bridge-layer Value updates, full FP32 QK, exact patch TopK, renormalized probabilities and gathered AV INCLUDED in timings.
No adaptive budget, online audit or extra selected-edge QK. Query chunks bound memory; masks remain token-wise per-query.
SAF3R is the full profile, correspondence topk=128 per frame, not 128 global keys.
Pose AUC only; no depth/reconstruction claim. Each scene is one designated sequence, 128 uniformly sampled frames.
Within-scene medians over measured repeats, then seven-scene macro means. Check per-scene input metadata for warmups/repeats and source hashes.

| Method | Global24 s | Whole s | AUC30 % | AUC5 % | Peak GiB |
|---|---:|---:|---:|---:|---:|
| hybrid | 118.5296 | 127.0956 | 83.8174 | 35.0777 | 14.859 |

## Hybrid components (24-layer totals)

- Dense prefix 0–2, INCLUDING L2 seed: 1.7151s.
- 14 representative-Value layers 3–9/17–23: 1.7975s.
- 7 full-QK Top-K layers 10–16, INCLUDING their Value updates: 115.0170s.
- Mean kept edges over all 24 layers: 21.25013%; sparse layers target 10%, prefix layers 100%.
- Kept-edge density is NOT computed-QK density: TopK layers still score all Keys.
- Nominal QK+AV FLOPs: 10% in Value layers, 55% in TopK layers; overall 34.375% before clustering, sorting, memory access and projection costs.
- FP32 QK/TopK and gathered AV can be much slower than tensor-core Flash Dense; nominal FLOPs do not establish latency speedup.

| Reference | Whole-model speedup (reference/hybrid) | Global speedup | Hybrid AUC30 difference (pp) |
|---|---:|---:|---:|

## Per scene

| Scene | Method | Whole s | Global24 s | AUC30 % | AUC5 % |
|---|---|---:|---:|---:|---:|
| chess | hybrid | 126.9636 | 118.3791 | 89.4845 | 48.3883 |
| fire | hybrid | 127.3778 | 118.8403 | 94.0014 | 68.7008 |
| heads | hybrid | 126.8866 | 118.4494 | 88.6823 | 41.8258 |
| office | hybrid | 127.0653 | 118.3425 | 77.5328 | 24.6531 |
| pumpkin | hybrid | 127.6802 | 118.8782 | 75.1353 | 17.8839 |
| redkitchen | hybrid | 127.0745 | 118.3775 | 81.7110 | 26.8307 |
| stairs | hybrid | 126.6216 | 118.4402 | 80.1743 | 17.2613 |

## Archived matched-input accuracy references

Reference poses are from the earlier seven-scene v4 Dense/SAF3R run. All frame paths, shapes and checkpoint paths must match.
Only accuracy is compared here; archived runtimes are NOT mixed into a new paired speed claim.

| Scene | Dense AUC30 % | SAF3R AUC30 % | Hybrid AUC30 % |
|---|---:|---:|---:|
| chess | 88.4662 | 88.5007 | 89.4845 |
| fire | 94.5230 | 94.6256 | 94.0014 |
| heads | 94.0071 | 94.1334 | 88.6823 |
| office | 77.1342 | 75.8760 | 77.5328 |
| pumpkin | 75.2083 | 74.9094 | 75.1353 |
| redkitchen | 79.7683 | 80.1796 | 81.7110 |
| stairs | 80.9416 | 83.8095 | 80.1743 |
| Macro mean | 84.2927 | 84.5763 | 83.8174 |

This is NOT the old global-PAM seed + q=8 exact-medoid experiment. It restores the representative-token attention form while using the explicitly documented GPU lightweight hierarchy.
Earlier full-layer block/adaptive results and L3–L9-only results are different policies and cannot substitute for these cumulative runs.
