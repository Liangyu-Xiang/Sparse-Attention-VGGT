# Coarse-sensing corrections and controlled re-evaluation

Historical first-stage report. The subsequent complete policy/kernel redesign
and cumulative re-evaluation are documented in `adaptive_block_value.md`.
The scope statements below describe this controlled experiment, not all later work.

## Scope confirmed by user

The user selected the referenced chat's first-stage task: **coarse sensing and
diagnostics/operator re-evaluation only**, before changing budgets, masks or
running a new cumulative sparse evaluation. Consequently the failed legacy
full-forward configuration is preserved for reproduction, not silently replaced
with an unvalidated new policy. There is **no new sparse end-to-end AUC or speedup**.
Layer 0 remains uniform Value-only in the inference method; its sensing probes
here are explicitly counterfactual measurements.

## Implemented corrections

`saf3r/patch/coarse_sensing.py` provides reusable matched-head `Q_h @ Kbar_h.T`
sensing. Shared Value groups pool each head's own Keys separately. The default
relevance softmax has **no log-population prior**. Population is returned
separately for future capacity/allocation. Entropy statistics retain the head axis
and average only queries. The old mean-Q/mean-K and size-biased variants are
explicit ablation controls, not defaults of this helper. CS is exposed as an
uncalibrated statistic, not asserted to be a valid budget fraction.

The previous diagnostic's `coarse_group_recall_top20pct` was dense **Top-K count
recall**, not retained attention mass. This evaluation separately computes:

- True patch mass = selected groups' dense probability / total patch probability.
- Dense-total mass with the special-key denominator retained.
- Dense patch Top-K count recall.
- Token coverage and mass lift over that coverage (averaged per head/query).

An apparent gain from large groups covering more tokens is therefore visible.
Special keys are always retained in operator tests and count toward the total
10/15/20% budget. Group localization metrics normalize patch-only probabilities
and separately report global probabilities; the definitions are not interchanged.

## Controlled experimental design

1. Chess seq-01 frames 0,20: all queries, all 24 dense-prefix global layers,
   same captured QKV and shared fine partition for every 2x2 sensing comparison.
2. Mean-head versus matched-head, crossed with +/− log population bias.
3. Coarse sizes 32/64/128: regroup the SAME 15%-fine partition. New mappings do
   not feed into later-layer features or the reference hierarchy, avoiding a
   cumulative-feedback confound.
4. Seven 7Scenes sequences, each 128 frames, 64 fixed evenly spread query probes
   per layer against ALL Keys. All seven dense model pose predictions match the
   saved formal dense baseline. Sampling is not full-query certification.
5. Isolated operator budgets 10/15/20% on identical current inputs. These use
   ONE common **20%-fine** partition so 20% unique representative budgets fit;
   a 15%-fine partition could not support them. This is a diagnostic control,
   not an apples-to-apples rerun of the previous cumulative 15%-fine method.
   Controls are shared Value, uniform, and dense-score Top-K with mandatory
   specials. The last is not a cheap deployable selector or minimum-NMSE oracle.

No inference head is averaged in its actual output; no alpha multiplier, minimum
Attention budget, adaptive total budget, new per-head Mask/history policy, or
optimized indexed kernel is introduced in this stage.

## Results: seven 128-frame sequences, 64 query probes/layer

Coarse size 128, macro mean over all 24 layers, heads and seven scenes:

| Sensing | Size bias | C | S | Top20% true patch mass | Top20% Top-K recall | Token coverage | Mass lift |
|---|---|---:|---:|---:|---:|---:|---:|
| Mean Q/K | yes | 0.019859 | 0.000586 | 41.024% | 41.045% | 35.260% | 1.170 |
| Mean Q/K | no | 0.001575 | 0.000588 | 38.524% | 46.616% | 19.063% | 2.021 |
| Matched heads | yes | 0.126212 | 0.049305 | 59.366% | 73.030% | 27.952% | 2.264 |
| Matched heads | no | 0.112607 | 0.049469 | 53.419% | 79.544% | 19.453% | 2.755 |

The head-averaging issue is supported: matched-head specificity is about 84x
the old biased mean-Q/K baseline. **Size bias is NOT confirmed as the main cause
of specificity compression**: removing it changes S only slightly at fixed
head mode. It substantially changes concentration and coverage. Biased scores
can retain more raw mass by selecting more tokens; the prior should be kept
distinct from relevance rather than declaring all biased allocations harmful.

| Coarse size (matched heads, no bias) | Top20% true patch mass | Top20% Top-K recall |
|---|---:|---:|
| 32 | 55.116% | 84.712% |
| 64 | 54.336% | 82.242% |
| 128 | 53.419% | 79.544% |

Smaller groups help moderately, but 32 costs about 4x coarse QK work versus 128
at equal heads and full queries. This does not justify a production default
without measuring its complete mask-generation/execution cost.

Shared-Value isolated mean output NMSE: **0.887728 / 0.462086 / 0.277611** at
10% / 15% / 20%. More budget reduces local error, but some heads/layers remain
poor. Uniform controls average slightly lower NMSE in this experiment. Higher
retained mass from dense Top-K also does not imply lower output NMSE, because
renormalization and Value cancellation differ across heads. Per-head tables
retain these differences; no global safety threshold is invented.

Two-frame and 128-frame statistics must not be conflated: the latter samples
64 queries and uses a much larger context; its entropy specificity is a sampled
estimate. All variants within each run use identical query samples.

## Saved artifacts and reproduction

- `outputs/coarse_sensing_review/chess_0_20/report.md`: all-query small diagnostic.
- `outputs/coarse_sensing_review/chess_0_20/chess/layer_00.pt` ... `layer_23.pt`:
  normalized current Q/K/V, fine labels/reps, every size/mode/bias probability
  matrix and statistics, coarse mappings/populations, dense Top-K indices.
  Approximately 1.6 GiB; retained locally for offline follow-up.
- `outputs/coarse_sensing_review/7scenes_n128_q64/report.md`: cross-sequence probe.
- Both roots include sensing aggregate and scene/layer/head CSVs, operator
  aggregate and per-layer CSVs, complete JSON records and dense pose predictions.
- Settings: `configs/analysis/coarse_sensing_review.json`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/evaluate_coarse_sensing.py --scene chess --frame-indices 0 20 --save-tensors

PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/evaluate_coarse_sensing.py --gpus 0 1 2 3 4 5 6 \
  --frames 128 --query-samples 64 \
  --output-dir outputs/coarse_sensing_review/7scenes_n128_q64

PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/report_coarse_sensing.py outputs/coarse_sensing_review/chess_0_20
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/report_coarse_sensing.py outputs/coarse_sensing_review/7scenes_n128_q64
```

Remaining stages are budget calibration on the repaired signal, a head-aware
mask/history policy and a performant query-specific execution path, followed by
new cumulative end-to-end AUC/timing. None is claimed solved by this first stage.
