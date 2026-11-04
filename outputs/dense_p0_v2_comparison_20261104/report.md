# Corrected P0 versus original Dense, paired heads128

Same GPU7,exact128 frames,checkpoint,preprocessing,BF16,TF32 disabled,onewarmup+three complete forwards. CPU-only lock and single CUDA process through full subprocess exit. Profiles are separate,not a partition of medians. No quality measured by these timing diagnostics; all six fields N/A with reasons in raw results and diagnostic tables. Full geometry quality is in ../v8_p0_quality_v3_20261104/report.md.

|method|warmed median s|Dense speedup|profile core s|profile global Attention s|profile outside global s|
|---|---|---|---|---|---|
|dense|12.642997|1.000000|7.567077|8.096077|5.480428|
|p0|21.134579|0.598214|6.460871|16.323766|4.653690|

Full24-layer rows are in per_layer.csv. Sparse core includes fused probe/Grid evidence, not sorting/reduction wrapper. Common backbone/frame Attention/FFNs/heads remain outside global Attention. Corrected P0 fails the later all-window equivalence check and is slower than original Dense. No equivalent-execution optimization success is claimed. Source is frozen by ../v8_p0_quality_v3_20261104/source_sha256.json and archive. Its method differs from initial P0 in ../dense_sparse_comparison_20261104.
