# Audited, head-preserving block Value/Attention redesign

[完整中文版方案](adaptive_block_value_zh.md)

This supersedes the **implementation policy**, not the historical measurements,
of `hierarchical_value`. The legacy implementation/results remain available for
reproduction. It is a block-quantized method, not an exact implementation of the
old per-query medoid mask. All 24 **global** attention layers are eligible; frame
attention is unchanged. Value clustering is still shared using head-mean V.
Attention decisions now retain individual heads, as requested by the latest
implementation critique; they do not average Q and K before multiplication.

## What changed

| Identified issue | New behavior |
|---|---|
| `alpha=C*S` almost always rounds to zero | Compare Value-only, C-based mixture and Attention-only masks using actual sampled output error; S remains diagnostic, not a budget multiplier. |
| Mean Q/K cancels head information | Matched-head QK, softmax first; average probabilities from two routing queries per query block. |
| Population prior enters reliability | No log-population prior in relevance, C or S. True populations are used for Value coverage and actual edge counts only. |
| Value groups assumed to preserve Keys | Value hierarchy determines coverage priorities only. Sensing pools each head's Keys over execution-aligned contiguous patch blocks, independently of Value groups. This alternative pooling is still approximate, **not** proof that all QK peaks survive. |
| All layers forced to 10% | Start at 10%, test 20/35/50/75%, then exact dense fallback. Selection uses online output probes, not ground truth poses. Fixed-10% remains an explicit control. |
| Shared mask harms heterogeneous heads | Per-head budgets/masks/history, shared Value hierarchy. No averaging of actual multi-head attention outputs. |
| A few Attention edges select a very slow scalar path | One tensor-core block-sparse QK/softmax/AV path regardless of A/V split. Evidence is exported from that same QK pass. |
| History disappears whenever A=0 | After bootstrap, keep at least four validated blocks (limited by available slots); use current relevance for cross-layer head matching and reuse filtering, not old weights. |
| Dense fallback allocates quadratic attention memory | Explicit 4-D batched SDPA inputs, enabling FlashAttention on the evaluation GPU. |
| Budget search synchronizes thousands of times | Vectorized candidate evaluation across heads; at most one host decision synchronization per candidate budget. |
| Rare repeat-inference divergence from floating atomic centroids | New method uses stable token sorting and segmented means in fine representative updates and coarse Lloyd updates; integer counts stay exact. Legacy defaults are preserved. |

## Precise policy

Defaults: query/key blocks 64/64, shared fine ratio 15%, shared coarse target
size 128, two synchronous sibling updates and three coarse Lloyd iterations.
Representatives are nearest to the cluster centroid, **not exact Euclidean
medoids**. Layer 0 initializes a uniform Value partition and uses uniform
Value-only block priorities, with no Attention routing or history branch.
It still uses output probes to decide its safe **total** budget.

Layer l's masks depend only on current Q/K/V and prior-layer indices/hierarchy.
Routing samples two actual queries per block (offsets 1/4, 3/4); it does not use
mean-query times mean-key across heads. Sensing is a heuristic, not an oracle.
The audit samples 32 other evenly distributed queries (offset 1/2), against all
Keys. It pools exact dense weighted Values/probabilities by patch block, so many
candidate masks can be evaluated without recomputing QK.

For each head, the first candidate total budget satisfying aggregate output
NMSE <= .02 and 90th-percentile per-query NMSE <= .10 is accepted. At that
budget choose the passing A/V mixture with the lowest aggregate error. If none
passes, execute dense SDPA. These are **sampled** safeguards, not a guarantee
about unsampled queries or final pose accuracy. Calibration queries are not an
independent validation set for the candidate they select.

A slots reuse at most half their capacity from previous validated blocks that
are currently in the top-2A relevance region; remaining slots explore current
relevance. V slots fill previously uncovered blocks by shared Value coverage
priority. The union contains exactly the chosen number of **unique blocks**.
Key blocks use frame-major patch-token order; special tokens are separately
mandatory. Per-head cross-layer matching is a Hungarian assignment maximizing
current probability enrichment of old indices; enrichment <=1 disables reuse.
No prior attention weights or dense attention matrices are retained.

After sparse attention, history is selected by mean log block attention mass,
computed inside the same QK kernel. All currently selected blocks may enter
history, including those initially chosen by Value coverage. Dense-fallback
heads retain empty history because Flash SDPA does not export that evidence.

## Honest budget and timing accounting

The target patch-block count is ceil((round(rho*N)-Nspecial)/BK), at least one,
capped by the number of patch blocks. A partial last block has its real size.
Thus fixed-10% is **block-rounded**, not exactly 10.000%. Report actual unique
Key count including special tokens, averaged over heads/queries/layers. Adaptive
results are not labeled 10% retention; dense fallback counts as 100%.

End-to-end benchmarks include initialization, all Value updates, routing,
online dense probe QK/output checks, candidate selection, history matching,
mask assembly, sparse execution and dense fallback. There is no excluded
layer-2 global clustering. Compare full 24-global-layer attention modules and
whole VGGT separately. A faster sparse kernel alone does not establish model
speedup.

Leading QK+AV arithmetic is approximately 4*H*N^2*d for Dense and rho times
that for selected edges, where rho is the measured density, including dense
fallback heads and special keys. With 128 frames, N=133248 and 132608 patch
tokens. Two routing queries per 64-query block against 64-key block means cost
roughly 1/4096 of Dense QK+AV FLOPs; 32 full-key calibration queries add roughly
32/N of that Dense arithmetic. Candidate checks, Value clustering, mask sorting,
memory traffic, launches and reduced kernel efficiency are additional costs.
This nominal arithmetic reduction is **not** a measured latency reduction.
In particular, expensive per-head masks and validated evidence need not achieve
the utilization or contiguous accesses of Dense FlashAttention or SAF3R.

## Verification and evaluation

- Unit tests compare block sparse output with explicit same-mask FP32 softmax,
  including tail blocks, varying head budgets, FP16/BF16 and same-pass evidence.
- Batched audit masks must match the actual production mask construction.
- Full retention must recover Dense and reset all 24 layers across sequences.
- Controlled sensing evidence/limitations remain in `coarse_sensing_review.md`.
- Formal final run: `outputs/adaptive_block_value/7scenes_n128_audit02_v4`, seven
  scenes, 128 uniformly sampled frames each, one warmup and three repeats.
  Dense, SAF3R full profile (correspondence topk=128 **per frame**), fixed10 and
  adaptive use identical inputs/checkpoint and the same scene GPU/process.
  SHA256 source manifests are saved per scene. An extra untimed forward checks
  64 independent queries/head/layer and records stage CUDA events. Its predicted
  poses must agree with formal predictions (maximum absolute difference <=1e-4).
  Online calibration probes remain enabled and charged in every formal sample;
  only these extra validation/profiling probes are outside the formal samples.
  The three measured pose encodings must have identical SHA256 fingerprints.
  v1 is an earlier completed policy/kernel version; v2 failed a CUDA Bool
  scatter-reduce regression, and v3 failed the Stairs diagnostic-repeat check.
  Those failed runs must not be pooled into the final seven-scene statistics.

Reproduce:

```bash
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/benchmark_adaptive_block_value.py --gpus 0 1 2 3 4 5 6 \
  --frames 128 --warmup 1 --repeats 3 --diagnostics-after-timing \
  --output-dir outputs/adaptive_block_value/7scenes_n128_audit02_v4
```

Interpret the final aggregate/report, not the single-scene JIT-inclusive
preflight. Accuracy preservation and a speed advantage over SAF3R are empirical
questions; the code changes alone do not establish either.

## Final v4 outcome (seven scenes, 128 frames)

| Method | Actual edges | AUC30 % | Global24 ms | Whole model s |
|---|---:|---:|---:|---:|
| Dense | 100% | 84.2927 | 11332.209 | 18.9856 |
| SAF3R | profile-dependent | 84.5763 | 4084.840 | 11.8833 |
| New fixed10 | 10.0384% | 74.2049 | 4448.753 | 12.7714 |
| New adaptive | 38.4790% | 83.2991 | 8904.548 | 16.7564 |

The known implementation problems were addressed, but the **full accuracy/speed
goal is NOT achieved**. Adaptive improves on the archived legacy 40.393% AUC30,
but remains 0.994 percentage points below Dense; whole-model speedup is 1.133x
over Dense, while it is 1.410x slower than SAF3R (2.180x in global layers alone).
Fixed10 still loses substantial accuracy; it is not a recommended final policy.
Even with head-aware sensing, shared-within-64-query-block masks and Key-block
pooling remain approximations, not verified replacements for all dense peaks.

Independent same-current-input query validation has mean output NMSE .010569
(calibration .010442); 94.903% of head/layer/scene checks pass both thresholds.
That is not 100% certification. Local errors also accumulate: Stairs AUC30 falls
from 80.942% Dense to 77.383% adaptive, despite low sampled operator error.
The three measured predictions per method/scene are bitwise identical, and all
14 extra diagnostic predictions match their formal predictions exactly.

Diagnostic 24-layer stage totals for adaptive: sparse QK/AV/evidence **5266.7ms**,
Value hierarchy **1087.0ms**, head matching/budget selection **592.0ms**, Mask
assembly **239.8ms**. The dominant remaining latency is sparse execution, not
Mask assembly or exact medoid search. Merely decreasing local-update overhead
will not erase the gap to SAF3R. Stage events are from separate diagnostics,
not replacements for formal timing samples.

Full report: `outputs/adaptive_block_value/7scenes_n128_audit02_v4/report.md`;
held-out errors/stages: `diagnostics_report.md` in that directory. Per-head,
per-layer, per-scene decisions, independent errors, prediction checks, source
hashes and an exact source snapshot are saved there. All tests: **200 passed,
1 skipped** on the evaluation environment. Raw two-frame Chess 0/20 replay
captures are stored separately under `outputs/adaptive_block_value/chess_0_20_replay`;
they are explicitly not 128-frame performance evidence.
