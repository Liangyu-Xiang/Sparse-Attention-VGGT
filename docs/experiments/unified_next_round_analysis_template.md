# Unified Next-Round Analysis Template

Copy this file to:

`outputs/unified_next_round/analysis.md`

and fill it only from committed diagnostic outputs.

## 1. Dense adjacent-layer reuse ceiling

| Pair | Prev Top20 -> current Top20 recall | Prev Top40 recall | Prev Top50 recall | Prev Top80 recall | Current attention mass covered by Prev Top20 |
|---|---:|---:|---:|---:|---:|
| L8->L9 | | | | | |
| L9->L10 | | | | | |
| L10->L11 | | | | | |
| L11->L12 | | | | | |
| L12->L13 | | | | | |
| L13->L14 | | | | | |
| L14->L15 | | | | | |
| L15->L16 | | | | | |
| L16->L17 | | | | | |

State whether a 50–80% **candidate reuse** regime is supported.  Always distinguish candidate density from target recall.

## 2. Current History pipeline

| Layer | previous_full recall | history_candidates recall | history_selected recall | final_support recall | candidate density |
|---|---:|---:|---:|---:|---:|
| L9 | | | | | |
| L10 | | | | | |
| L11 | | | | | |
| L12 | | | | | |
| L13 | | | | | |
| L14 | | | | | |
| L15 | | | | | |
| L16 | | | | | |

Supported case: **H1 / H2 / H3 / H4**

Evidence:

- 
- 
- 

## 3. Query-group granularity

| Pair | group32 recall | group16 | group8 | group4 | per-query |
|---|---:|---:|---:|---:|---:|
| L9->L10 | | | | | |
| L10->L11 | | | | | |
| L11->L12 | | | | | |
| L12->L13 | | | | | |
| L13->L14 | | | | | |
| L14->L15 | | | | | |

Does smaller query sharing materially improve recall at identical 20% candidate density?

## 4. Value-cluster compressibility

### Metric ranking

| Signal | Rank correlation with importance-weighted diagnostic error | Precision@20% |
|---|---:|---:|
| Value relative L2 | | |
| Value cosine distance | | |
| Key relative L2 | | |
| Key cosine distance | | |
| QK logit std | | |
| Representative affinity error | | |

Report:

- fraction/examples of **low Value distortion + high Attention-side error** clusters;
- current Rep vs exact-like metric distributions;
- whether Value-only Rep/Exact decision is over-aggressive;
- one recommended cheap compressibility signal, or "none".

## 5. History reuse scaling

| History ratio | Exploration ms/window | Solver ms/window | Total ms/window | Projected all-window ms | Peak MiB |
|---:|---:|---:|---:|---:|---:|
| 0% | | | | | |
| 25% | | | | | |
| 50% | | | | | |
| 75% | | | | | |
| 80% | | | | | |

Does current implementation satisfy:

`more reliable History -> less exploration work -> lower mask-construction time`?

**Yes / No**

## 6. Causal conclusions

### History

Most likely bottleneck:

Evidence:

One next change only:

### Rep / Exact

Most likely bottleneck:

Evidence:

One next change only:

## 7. What NOT to change yet

List all plausible changes that are intentionally deferred so the next end-to-end ablation remains causal.

- 
- 
- 

## 8. Next end-to-end experiment

Specify exactly one History modification and at most one Rep/Exact modification.

Do not change total retention, population compensation, query grouping, History scoring, and Value criterion all in the same run.
