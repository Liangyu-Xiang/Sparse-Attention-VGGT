# Current selected version:original History80 with population compensation

This is a local research extension,not the official SAF3R method or a verified
paper reproduction. The user-selected configuration is
`configs/evaluation/parent_region_history80_population.json`.

| Global layers (zero based) | Attention patch budget | Value patch budget | Total target |
|---|---:|---:|---:|
|0|0%|100%|20%|
|1–9|20%|80%|20%|
|10–16|80%|20%|20%|
|17–23|20%|80%|20%|

Special Keys are always retained inside the total budget. Whole-Parent selection
can slightly overshoot. Ratios are unique patch-edge budgets,not output mixing
coefficients. All24 global layers are sparse;frame/local attention is unchanged.

Value clustering uses head-meanV,spatial initialization atL0,restricted
synchronous GPU updates and current-layer centroid-nearest REAL representatives.
Merging covers every patch Token.32 adjacent Queries share Attention support;
head/Query weights remain independent. Query-group probes rank pooled complete
Parents,then all member Keys in selected Parents are retained. Historical edges
are hard-kept after deduplication;outgoing history is capped to80% of the NEXT
layer's Attention budget. There is no Dense seed or Dense fallback.

## Population compensation

For each Value representative,add `log(cluster_size - real_attention_members)`
to its final QK logit. Attention real Keys and specials have unit mass. The
residual count uses the current merged partition and Query-group-specific
support,so already-expanded members are not double-counted. Same-pass weighted
evidence produces history. A SINGLE online softmax normalizes the combined
Value/Attention/special support. This is an approximation,not an exact Dense
operator unless the represented members have appropriate matching K/V.

Core implementation: `saf3r/patch/value_region_av.py`,
`saf3r/triton/fused_shared_av.py`. Defaults remain opt-in for old reproduction;
load the selected JSON explicitly:

```python
import json
from saf3r.patch.value_region_av import ValueRegionAVConfig, install_value_region_av

# model must be original VGGT,not already patched SAF3R.
with open('configs/evaluation/parent_region_history80_population.json') as stream:
    config = ValueRegionAVConfig(**json.load(stream))
state = install_value_region_av(model, config)
```

## Evidence and limitations

See `history80_population_ablation.md` for all six quality metrics,thresholds
and full24-layer output comparison. Same-input Dense/SAF3R baselines were reused
explicitly.7Scenes uses seven default128-frame scene entries,not every original
video;ETH3D uses11 configured entries with14–76 real images. Distances metres,
CD=(ACC+Comp)/2 unsquared. Speedup is N/A without paired warmed uninstrumented
full-forward measurements;captured diagnostic times are not speed benchmarks.
Quality gates have NOT all passed. Output similarity is not a task-quality proxy.

The middle A20/V80 calibration candidate is retained in
`parent_region_history80_population_middle20.json` ONLY as a failed-task-quality
experiment:its outputs get closer to Dense,but reconstruction degrades. It is
NOT the selected current version. See `history80_middle_parameter_calibration.md`.

## Reproduction entry points

After the existing README dataset/checkpoint setup,run fresh outputs (do not
overwrite prior directories):

```bash
# Complete Dense/SAF3R/uncompensated History80 diagnostic baseline.
python scripts/run_full_global_output_comparison.py \
  --output-dir outputs/global_attention/fresh_baseline --gpus 0 1 2 3 4 5 6

# Original History80 plus residual population compensation;reuse exact baseline.
python scripts/run_history80_population_comparison.py \
  --baseline outputs/global_attention/fresh_baseline \
  --output-dir outputs/global_attention/fresh_population \
  --gpus 0 1 2 3 4 5 6 --wait-busy-gpus
```

Baselines must be complete and exact frame lists match. If an original attempt
fails,retain it,use unchanged retries,and construct a documented consolidated
baseline via `scripts/consolidate_full_global_output_comparison.py`.
Source snapshots and explicit canonical inputs are part of each experiment.

## Single-Dense-layer ablation

`history80_single_dense_restoration.md` defines72 interventions plus3 fresh
baseline reproductions on heads,stairs,chess at128frames. Exactly one global
layer returns Dense output;its shadow sparse operator STILL maintains the
original clustering and sparse history. Remaining masks evolve naturally.
No Dense-derived history is introduced. These are causal diagnostics,not a new
recommended sparse deployment or an acceleration benchmark.

```bash
python scripts/run_single_dense_history80.py \
  --baseline outputs/global_attention/fresh_population \
  --output-dir outputs/global_attention/fresh_single_dense \
  --gpus 0 1 2 3 4 5 6
```

The queue writes all six metrics and timing-unavailable reasons,checks reproduced
baselines,uses idle GPUs,overlaps GPU forward only with CPU-only quality,and
preserves failed attempts. It runs independently of the chat. A repository sync
does not imply the background experiment has finished:completion and analysis
are reported only when requested separately by the user. No mutable progress
logs,weights,datasets or large cached Dense tensors are included in this sync.
