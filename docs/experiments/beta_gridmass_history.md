# Actual sparse probability → beta → Grid mass compensation

Experimental modeling change based on the frozen v8 source. Opt-in config: `configs/evaluation/reuse_beta_gridmass.json`. Retention20%, adaptive-local per-layer clustering, four probes, existing state losses, Drop/Rep/History/Exact states,16-round solver and formal AV arithmetic are retained. This is separate from the layered no-Drop method. Results and acceptance status: [campaign report](../../outputs/beta_gridmass_20261104/report.md).

The preceding layer exports its actual sparse Softmax probabilities, including compensation bias and Special normalization, by reusing emitted probe logits and the executor's final LSE. Probabilities are averaged after each head's own normalization and saved in the existing FP16-u14 cache. The old Grid denominator and conditional-probability reconstruction are not called. This first experiment still exports raw probe logits; removing that large intermediate is not claimed.

For each current coarse Grid probability G and previous coarse probability Gprev, preserve the existing confidence beta: ratio=(G+EPS)/(Gprev+EPS), beta=ratio if ratio<1 else ratio^-2. Let A be the sum of cached, quantized actual sparse probabilities of previously selected keys in this Grid. For a partially selected Grid, factor=min(beta,G/max(A,EPS)); selected key i receives factor*old_probability_i. Distribute max(0,G-factor*A) uniformly over the unselected members.

For an empty selected set, all G is assigned uniformly to members. For a fully selected Grid with A>EPS, factor=G/A regardless of confidence, as there are no unselected members to receive residual mass. For a fully selected Grid whose cached probabilities all underflow to zero, use uniform G/GridSize on those selected members. Current coarse Special handling is unchanged; patch priors sum to the current coarse patch mass, not an artificially forced1. Final per-probe estimates are combined using the original probe weights, including tail duplicate handling.

This defines a normalized coarse-Grid-constrained prior; it does not assert exact next-layer sparse or Dense Softmax recovery. Actual compensated sparse P has a different semantic meaning from v8's Grid-conditional estimate. Reusing the state loss and quality transfer is an explicit modeling hypothesis under evaluation.

Implementation:

- `saf3r/triton/reuse_serial_sparseprob.py`: original serial Attention execution, exposes its final LSE and logical bias; AV kernel and staging unchanged.
- `saf3r/triton/reuse_sparse_beta_history.py`: head-mean actual sparse P export/cache and scatter of current selected estimates. Per-Grid mass sums use the same quantized values the next layer decodes.
- `saf3r/patch/reuse_sparse_beta_history.py`: beta calibration, Grid mass cap, uniform missing compensation and current cluster priors.
- Existing forward gains opt-in history/backend branches. Existing evaluator also archives the new source and exact config.

GPU numerical validation covers FP32/BF16 formal Attention, actual sparse probability with bias, Special, Padding, duplicate tail probes, cache quantization, empty/full Grids, underflow fallback and per-Grid mass conservation. The first diagnostic used an overly small absolute threshold against an unquantized FP32 reference. The preserved v2 diagnostic compares to the FP16 storage reference; model formulas were unchanged. This is a numerical-reference correction, not a geometric quality-gate adjustment.

The experiment design is frozen before model quality results: only the changed method is run on all7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames, with pose+recon_unposed. Dense/SAF3R/v8/grid_local controls are historical and reused. No test-truth parameter tuning or selection of better repeats. GPU4–7 use CPU-only file locks, wait for full process exit, and abort on any foreign CUDA process. Benchmark: heads128, GPU7, one warmup+three full forwards, then a separate profile. Unpaired Dense speedup is N/A. Raw quality, per-scene six-metric tables, macro averages, source hashes, failures and monitors are preserved.

Numerical-only/profiling runs report all six quality fields unavailable with explicit reasons. Full quality uses AUC/F1 fractions in JSON and percentages in tables; distances metres, unsquared CD=(ACC+Comp)/2. Threshold/voxel:7Scenes0.05m/0.0078125m; ETH3D0.25m/0.0390625m. SAF3R controls are local lazy first-correspondence-head Top4 with ETH3D profile, not paper DINO Top-K.

## Interpreting the remaining runtime

The changed-method warmed complete-forward median is19.553523152s. A separate detailed profile is18.741772324s: global Attention replacement modules14.904668396s, with3.837103928s outside those modules. The latter is a residual accounting category, not a newly introduced algorithm stage. Original VGGT also has the image backbone, frame Attention, all block FFNs (including global-block FFNs), residual/norm work and prediction heads. Their exact original-Dense duration is not measured in this campaign; the current residual must not be assigned unchanged to Dense.

Historical20261029 Dense heads128 complete-forward measurements are about12.2s (12.216s in the earlier compact report). They are full-model timings, not isolated Dense Attention timings, and belong to a different batch. No isolated matched Dense Attention duration is available here, and subtracting the current sparse residual from a historical Dense full-forward number would not measure it. Dense paired speedup remains N/A. No unchanged control was rerun.

The sparse Attention core with fused Grid observation remains6.404190s, routing/state losses/budget/Mask work in the exclusive reuse_and_budget range3.741345s, execution-wrapper preparation0.837655s, history production0.492844s and next-layer history consumption0.528141s. The old denominator scan is removed, but the raw per-head four-probe logits are still exported and read again. The consumer still materializes a dense token prior for current cluster statistics. This implementation tests the estimator with the original execution path; it does not establish the minimum runtime of a fully fused probability-transfer implementation. Core kernel efficiency needs its own investigation; reduced retained keys alone do not establish a proportional end-to-end saving.

All18 quality runs completed. The new estimate is not accepted: 7Scenes macro F1 falls from historical v8 46.8951% to10.9585%, and ETH3D from59.5813% to1.9670%. See the complete six-metric table and matched-input audits in the campaign report. Numerical mass conservation does not establish cross-layer estimation accuracy. No follow-up estimator was tuned on these test results.
