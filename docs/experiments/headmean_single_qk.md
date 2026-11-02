# Head-mean history and single-QK grid summaries

This candidate implements the user's explicit head-mean transfer rule. It starts from the frozen ready-history v5 implementation. Budgets (20%), per-layer Value clustering, state losses, historical-Rep rule, Exact eligibility, mass transfer and discrete budget allocation remain unchanged. It does not claim bitwise equality to v5: head averaging before cross-layer confidence is an intentional estimator change, and physical Key grouping changes BF16 summation order.

## Cross-layer formula

For each of the four probes separately, the producer computes each current-layer head's estimated Dense probabilities and corrected grid mass, then averages the **conditional probabilities** over heads:

`a_bar[e] = mean_h(p_hat[h,e] / corrected_grid[h,grid(e)])`.

The unobserved-Key conditional is averaged in the same way. The preceding coarse grid probability is also averaged before caching. The consumer computes the current grid probabilities per head, averages them, and only then computes the ratio and confidence:

```
ratio = (current_grid_headmean + eps) / (previous_grid_headmean + eps)
confidence = ratio if ratio < 1 else ratio ** -2
predicted[e] = current_grid_headmean[r] * (
    confidence[r] * previous_conditional_headmean[e]
    + (1 - confidence[r]) / grid_size[r]
)
```

Finally it weights the four probes as before. There is no head-index correspondence across layers. It does not average logits. Cache tensors retain a singleton head axis for compatibility, not per-head data. FP16/u14 storage is retained. On heads128, the 16-head edge bank becomes 1/16 of its old logical size and fits in the GPU cache; no pinned-CPU probability bank is needed.

## One selected-edge QK pass

The sparse attention kernel processes special Tokens, then selected Keys grouped by canonical grid. The wrapper performs a stable GPU grouping and keeps the original support permutation. Each selected Key belongs to exactly one grid. QK is computed once per selected edge, with normal tile padding; probe logits are emitted in original logical Key order.

The same online-softmax pass accumulates weighted and uncompensated exponential sums per query and grid. It writes compact per-grid log sums and the final query log denominator. A reduction kernel uses these to produce the original all-query/head averaged weighted and uncompensated grid masses. This kernel performs no QK. Dividing by Patch mass reproduces the old `previous_grid` / `previous_real_grid` definitions, which remain inputs to the Exact eligibility algorithm. No full query-by-selected-Key logit/probability bank is saved.

Dense probability calibration uses the four emitted raw probe logits, the canonical-grid estimate of omitted Keys, and special Tokens, as before. Sparse-output normalization and the estimated Dense normalization are distinct. The calibration does not repeat QK.

## Reproduction and validation

`python scripts/setup_headmean_history.py --destination /absolute/fresh/path` generates a separate runnable source tree, retaining v5 unchanged. The frozen evaluation snapshot under `outputs/headmean_single_qk_quality_20261102/source` and its source archive preserve the complete implementation. Config records `evidence_mode=fused_grid`. Full per-edge capture is rejected explicitly because this backend emits grid summaries rather than per-edge masses.

`outputs/headmean_single_qk_validation_20261102/validation.json` covers FP32/BF16 Attention, FP32/FP16 caches, empty supports, invalid padding, tail queries, original probe ordering, explicit conditional averaging and current-head permutation invariance. FP32 output relative L2 is 1.68e-7, BF16 is 0.002032 against a direct FP32 calculation; grid statistics relative L2 is below 7.60e-8. These are numerical diagnostics, with all six quality fields N/A. Small-kernel timing is slower than v5; it is not an end-to-end speed claim.

Both-dataset quality/timing run: `scripts/run_headmean_history_quality.py`; report generator: `scripts/report_headmean_history_quality.py`. Only this candidate runs; Dense, SAF3R and ready-history v5 are reused. Inputs/checkpoint/protocol are audited; no parameters change after observing GT scores. Same-batch Dense speedup is N/A, because unchanged controls are not rerun.

## Completed dataset evaluation

All 18 scenes completed pose + recon_unposed once; all input/protocol/frozen-source/GPU overlap checks passed. All 432 layer records have zero second selected-edge QK and zero CPU probability-cache bytes. Results: `outputs/headmean_single_qk_quality_20261102/report.md`, `per_scene.csv`, `per_layer.csv`, `summary.json`.

|Dataset/method|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Mean scene forward s|Peak allocated GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|7Scenes previous v5|22.13219|84.32438|0.15008|0.12460|0.13734|45.70155|24.8544|25.808|
|7Scenes head-mean/single-QK|22.64190|84.43857|0.14410|0.12298|0.13354|46.12994|27.7707|17.211|
|ETH3D previous v5|24.41852|80.00208|0.35150|0.90507|0.62828|56.71405|4.4134|14.499|
|ETH3D head-mean/single-QK|22.34480|78.41895|0.35583|0.89430|0.62507|54.91852|4.4769|11.231|

Previous timings are historical, not contemporaneous paired comparisons. Current Dense speedup is N/A. 7Scenes uses seven scenes, 128 frames each. ETH3D uses eleven scenes, all available 14–76 frames. Quality improves on 7Scenes but ETH3D pose/F1 degrade; no claim of universally preserved quality or improved speed. No GT-based parameter retuning occurred.

The current single-QK execution has grouping/sorting, grid statistics, and short padded grid tiles; removing QK recomputation did not improve total measured forward latency. An alternative four-grid block layout was checked without GT and rejected after it failed to improve microbenchmark time. Its code/results remain under `outputs/headmean_single_qk_fast_validation_20261102`, while the deployed/reproducible kernel remains byte-identical to the evaluated frozen candidate.
