# Attention-first controlled campaign

The previous Dense-teacher ratio search and new two-frame execution smoke have
finished. This campaign freezes sources and configuration before launching
subprocesses on idle GPUs 4–6. Old results and old implementation are preserved.

## Frozen schemes and scope

- Dense: original VGGT, same checkpoint and image preprocessing.
- Old: whole Parent + original executor, manual Attention ratios
  `[0]+[.2]*9+[.8]*7+[.2]*7`, target20% support, history80%.
- New: independent causal allocation, Attention first, residual Value next;
  `configs/evaluation/attention_first_residual_value.json`, no GT ratio tuning.
- Main evaluation and layer-output comparisons:128 uniformly sampled frames,
  seed0. All seven scenes in the main Dense/new evaluation.
- Visualization:2 uniformly sampled frames from each selected scene, same
  frames for Dense/old/new. All24 global layers, no frame/local attention change.

Diagnostic scenes are selected **before the new campaign** from the previous
seven-scene old-vs-Dense results. Rank by
`max(AUC30 drop in pp /1, F1 drop in pp /3,0)` and take3. This chooses
**stairs, heads, office**. This is failure-analysis scope selection, not fitting
the new controller on test truth.

## Ordered phases

1. Paired Dense/new seven-scene quality evaluation. Each method has one warmup
   and three full forwards; each scene's pair runs on one GPU worker. Report
   AUC@3, AUC@30, ACC, Comp, CD, F1, timing, peak memory and matched speedup.
   A partial subset is not labelled a full-dataset macro.
2. For the three selected scenes, save true two-frame Dense attention
   probabilities (each head softmax then average) and the actual calculated
   old/new support. Plot in original token coordinates, Queries on rows, Keys
   on columns. White mask pixels are calculated edges; no reordering to make
   masks look block-contiguous. Save24 per-layer PNGs and three eight-layer
   contact sheets per scene, raw probabilities/masks and mask overlap metrics.
3. For the full128-frame input, perform independent autoregressive Dense, old
   and new forwards. Save every layer's attention-module output and full global
   block output to CPU/disk. Compare **every token and channel**, without Query
   sampling, for old-vs-Dense, new-vs-Dense and directly new-vs-old.
4. Evaluate old/new shallow-only, middle-only and deep-only sparse variants on
   the three scenes:18 ablations, with full task-quality metrics. Shallow0–9,
   middle10–16, deep17–23. The remaining two stages return original Dense output.

For phase4, inactive layers still update the selected sparse scheme's own Value
and historical state, then discard that layer's proposed sparse output and run
original Dense attention. Thus a stage transition is not accidentally converted
into a history reset or Dense-derived seed. This diagnostic includes auxiliary
selection/state and duplicate QKV costs in timing. Its speed is **not** an
optimized deployment estimate. Each ablation obtains its own same-worker Dense
timing rather than using slow hooked diagnostic forwards as its baseline.

## Layer error definitions

For flattened full-layer tensors X (scheme) and D (reference):

- Relative L2 = `||X-D||_F / ||D||_F`.
- Relative squared L2 = `||X-D||_F² / ||D||_F²`.
- Cosine = `<X,D> / (||X||_F ||D||_F)`.
- Mean-token cosine averages the per-token channel-vector cosines.

Computation streams chunks in FP32 with FP64 aggregate sums. Both-zero vectors
have cosine1; only-one-zero vectors have cosine0. Direct new-vs-old uses old as
the relative-error denominator. Comparisons are **not Dense teacher forcing**;
they include accumulated previous-layer changes. Stage ablations are needed to
distinguish early error propagation from an intrinsically poor middle/deep stage.

Full Dense/old layer tensors are retained on disk for reproducibility; they can
occupy tens of GiB per scene. No existing outputs are deleted. Diagnostic hooks,
Dense probability recomputation and disk I/O invalidate performance timing;
those runs mark all six task-quality fields/speed as unavailable with a reason.

## Execution and artifacts

```bash
python scripts/run_attention_first_campaign.py \
  --output-dir outputs/attention_first_residual_value/campaign_7scenes128 \
  --frames 128 --gpus 4 5 6
```

Source/config hashes are checked before every subprocess. The controller waits
for the previous search status to become
complete and for an idle GPU. Restarting the identical command resumes existing
completed results; changed sources require a fresh output directory.

Artifacts under the campaign directory:

- `manifest.json`, `source_snapshot/`, `controller.json`, `status.json`, `logs/`.
- `quality/{dense,attention_first}/{scene}/`: raw/task metrics, six-metric table,
  paired speedup, actual per-layer budgets, source snapshots and frame lists.
- `quality_comparison.{json,md}`: progressively updated table; full macro only
  when all14 paired runs exist.
- `diagnostics/visualize/{scene}/L00_comparison.png` etc.; `layers_00_07.png`,
  `layers_08_15.png`, `layers_16_23.png`; raw `.npy/.npz`, mask comparisons.
- `diagnostics/deviation/{scene}/deviations.md`, per-method errors JSON and full
  Dense/old output tensors. Attention and block-output scopes are distinct.
- `ablations/{shallow,middle,deep}/{old,attention_first}/{scene}/` and
  `ablation_comparison.{json,md}`.

There is no silent Dense fallback, budget relaxation, ratio search on7Scenes GT
or claim of unchanged accuracy/2x acceleration. Failures remain explicit in
status and missing-result lists, and must not be presented as completed quality
evaluations.
