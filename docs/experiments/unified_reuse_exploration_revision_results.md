# Unified Reuse / Exploration: three completed revision evaluations

All three revisions completed all seven 7Scenes scenes, 128 frames per scene, exact matched inputs/checkpoint/protocol.
All24 global layers sparse; frame/local attention unchanged. Total edge retention cap20%, including special Keys.
One warmup, three measured whole forwards including initialization, clustering, routing, history and evidence.
Quality: AUC/F1 percentages, reconstruction distances metres; CD=(ACC+Comp)/2 unsquared.
Unposed reconstruction threshold .05m; downsampling .0078125m. No test-truth parameter tuning.

## Changes and provenance

- `3afe65a`: first unified allocator, cluster-local History candidates and per-cluster cap.
- `46bd65b`: direct query-group History relation pool, no per-cluster cap, compact bank and merged representative deduplication.
- `57bdb78`: additionally scales History gains by previous explicit-relation coverage in the canonical grid.
- Every revision includes the disclosed FP32 sparse mapping / four-effective-probe padded Tensor Core compatibility overlay; see each evaluation_overlay.patch and source snapshots.
- Dataset/model files are not uploaded. Archived results and code snapshots are immutable experimental evidence, not a replacement for current method source.

## Equal-scene macro quality

| Method/revision | Complete | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGGT Dense (historical) | 7/7 | 23.0420 | 84.2927 | 0.14804 | 0.12436 | 0.13620 | 48.0537 |
| SAF3R lazy Top4 (historical) | 7/7 | 23.6027 | 84.4311 | 0.12976 | 0.12427 | 0.12701 | 49.1664 |
| History80 + population (historical) | 7/7 | 21.6547 | 83.8221 | 0.13378 | 0.12454 | 0.12916 | 49.4740 |
| Original reuse + residual (historical) | 7/7 | 22.2060 | 83.8300 | 0.12183 | 0.11292 | 0.11737 | 50.4427 |
| 3afe65a | 7/7 | 22.4128 | 84.3375 | 0.13581 | 0.12926 | 0.13254 | 47.2822 |
| 46bd65b | 7/7 | 22.3080 | 84.3211 | 0.13288 | 0.12694 | 0.12991 | 46.9800 |
| 57bdb78 | 7/7 | 22.3003 | 84.3062 | 0.13782 | 0.12741 | 0.13262 | 46.8034 |

## Timing, memory and actual edge retention

Rows are recorded in separate evaluation phases, not contemporaneous matched speed tests. No newly measured Dense speedup is available; historical ratios must not be presented as new speedups.
Timing includes selector costs; final edge density is not total computed-QK density. Peak is allocated memory, not reserved.

| Revision | Mean scene warmed median s | Max peak allocated GiB | Mean retained edges % | Paired Dense speedup |
|---|---:|---:|---:|---|
| 3afe65a | 106.617 | 15.494 | 19.9942 | N/A: no contemporary pair |
| 46bd65b | 70.481 | 15.712 | 19.8860 | N/A: no contemporary pair |
| 57bdb78 | 72.623 | 15.697 | 19.9053 | N/A: no contemporary pair |

## Per-scene quality for every evaluated revision

| Revision | Scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|
| 3afe65a | heads | 47.7731 | 93.5277 | 0.03479 | 0.03288 | 0.03383 | 83.1674 |
| 3afe65a | chess | 20.5709 | 88.4383 | 0.13010 | 0.09025 | 0.11018 | 37.1505 |
| 3afe65a | stairs | 7.0497 | 81.9082 | 0.22465 | 0.50111 | 0.36288 | 15.6202 |
| 3afe65a | fire | 53.5638 | 94.2089 | 0.04879 | 0.05144 | 0.05011 | 66.5810 |
| 3afe65a | office | 13.2177 | 77.5197 | 0.05010 | 0.05529 | 0.05269 | 58.0628 |
| 3afe65a | pumpkin | 6.4427 | 75.1743 | 0.38177 | 0.08756 | 0.23467 | 25.0691 |
| 3afe65a | redkitchen | 8.2718 | 79.5854 | 0.08049 | 0.08632 | 0.08341 | 45.3244 |
| 46bd65b | heads | 48.1709 | 93.6467 | 0.03537 | 0.03245 | 0.03391 | 82.7774 |
| 46bd65b | chess | 20.7841 | 88.5724 | 0.11704 | 0.08392 | 0.10048 | 42.5166 |
| 46bd65b | stairs | 6.5945 | 81.5244 | 0.22737 | 0.48792 | 0.35764 | 14.7030 |
| 46bd65b | fire | 52.6493 | 94.0769 | 0.04943 | 0.05316 | 0.05129 | 65.1306 |
| 46bd65b | office | 13.1275 | 77.3757 | 0.05489 | 0.06035 | 0.05762 | 51.2630 |
| 46bd65b | pumpkin | 6.5002 | 75.3859 | 0.36681 | 0.08345 | 0.22513 | 27.6897 |
| 46bd65b | redkitchen | 8.3292 | 79.6658 | 0.07926 | 0.08732 | 0.08329 | 44.7796 |
| 57bdb78 | heads | 46.8914 | 93.4555 | 0.03472 | 0.03301 | 0.03387 | 83.5083 |
| 57bdb78 | chess | 21.1245 | 88.5581 | 0.12476 | 0.08920 | 0.10698 | 36.1334 |
| 57bdb78 | stairs | 7.4393 | 82.0661 | 0.23032 | 0.47873 | 0.35452 | 14.1448 |
| 57bdb78 | fire | 52.5960 | 94.0814 | 0.05136 | 0.05243 | 0.05189 | 65.2427 |
| 57bdb78 | office | 13.2464 | 76.9193 | 0.05179 | 0.05594 | 0.05387 | 55.1104 |
| 57bdb78 | pumpkin | 6.4222 | 75.3027 | 0.39347 | 0.09910 | 0.24628 | 27.2029 |
| 57bdb78 | redkitchen | 8.3825 | 79.7601 | 0.07833 | 0.08346 | 0.08089 | 46.2812 |

## Interpretation and limitations

Neither update consistently improves all quality metrics. Latest macro F1 remains below matched Dense and both earlier revisions. Do not infer a component-level causal conclusion without controlled ablations.
SAF3R baseline uses local lazy first-correspondence-head per-frame Top4 with the ETH3D profile, NOT paper DINO feature TopK reproduction. Local measured results are not published-reference scores.
These are seven-scene128-frame evaluations, not300-frame runs or evaluation of every dataset frame.

## Full frozen reports

- [3afe65a: per-scene baseline comparison and provenance](../../outputs/global_attention/unified_refactor_20261025/comparison.md)
- [46bd65b: per-scene baseline comparison and provenance](../../outputs/global_attention/unified_refactor_46bd65b/comparison.md)
- [57bdb78: per-scene baseline comparison and provenance](../../outputs/global_attention/unified_refactor_57bdb78/comparison.md)
