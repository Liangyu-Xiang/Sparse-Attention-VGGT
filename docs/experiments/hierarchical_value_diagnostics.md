# Retained diagnostics for iterative algorithm improvement

Implements the logging request in the latest referenced chat. Actual selection
stays **layer-shared**. Per-head fields are observational error/oracle analysis,
not per-head budgets or separate selection state.

## Completed artifacts

Root: `outputs/hierarchical_value/7scenes_n128_r10_all24_uniform/diagnostics/`.

- `report.md`: the four core tables, plus small-input dense oracle analysis.
- `runtime_breakdown.csv`: each stage's synchronized milliseconds and layer-time
  percentage; the sum of detailed components is not a formal inference benchmark.
- `mask_composition.csv`: K_A/K_V, reuse/explore/V-only/AV fractions.
- `history_quality.csv`: survival, exploration precision, incoming/validated counts.
- `accuracy_breakdown.csv`: per-layer mean/median/p90 NMSE and final MODEL AUC.
- `accuracy_per_scene_layer_head.csv`: per-scene, per-layer, per-head errors.
- `full128/<scene>/summary.json`: all seven scenes, each 128 frames, plus hierarchy
  occupancy distributions, migration/representative-change fractions, original
  coarse-sibling search range, tensor sizes and peak allocated memory.
- `full128/<scene>/status.json`: confirms diagnostic predictions match the formal
  saved predictions (all seven verified). Diagnosis does not change selection.
- `chess_frames_0_20/chess/layer_00.pt` through `layer_23.pt`: 24 raw diagnostic
  snapshots, approximately 1.2 GiB total. GT task metrics on two frames are not
  meaningful dataset estimates; use full128 formal metrics for final accuracy.

## Raw tensor schema

| Field | Meaning |
|---|---|
| `fine_cluster_id` | Patch-local token -> fine cluster |
| `fine_rep_index` | Fine cluster -> representative patch-local token |
| `fine_to_coarse` | Fine cluster -> coarse group |
| `fine_cluster_sizes`, `coarse_group_sizes` | Current occupancies |
| `patch_token_global_indices`, `special_global_indices` | Map local patch tokens to full sequence |
| `coarse_attention` | Shared FP16 [query, coarse group]; null for forced Value-only layer 0 |
| `group_budget` | Shared INT32 [query, coarse group], row sum K_A |
| `statistics` | C, S, alpha, K_A, K_V |
| `previous_validated_edges`, `validated_edges` | Patch-local Key indices, no head axis/weights |
| `final_sparse_indices` | Full-sequence INT32 Key indices [query, total K], including specials |
| `selection_origin` | UINT8: 0=V, 1=A-reuse, 2=A-explore, 4=mandatory special |
| `provenance` | Same codes, with validated V upgraded to 3=AV |
| `validated_slots` | Slots of patch edges that survived mean-head exact-logit ranking |
| `exact_sparse_logits` | FP16 saved mathematical FP32-oracle logits [head, query, total K] |
| `dense_topk_indices` | Full-sequence indices from mathematical FP32 dense-logit oracle |
| `retained_dense_mass`, `dense_topk_recall` | [head, query] on that same current input |
| `dense_output`, `sparse_output`, `per_query_head_output_nmse` | Same-input local operator error |

An A-origin edge is **not automatically validated**. Always inspect
`validated_slots` or `validated_edges`; provenance retains its original reason
so failed candidates remain analyzable. Mandatory specials are outside the patch
A/V split. Missing rates (no reused or explored samples) are null/N/A, not 0 or 1.

The raw sparse logits are reconstructed in the **separate small diagnostic**
from the current Q/K dense oracle, not another QK added to formal forward timing.
Dense output references cast normalized Q/K to V's dtype (BF16), matching formal
autocast SDPA and keeping memory bounded. Dense-logit oracle math is FP32.
NMSE is squared error energy / dense-output energy; relative L2 is its square
root and is also retained. This distinction avoids the referenced chat's
ambiguous use of “NMSE” for a non-squared norm ratio.

## Timing protocol

`ValueDiagnostics.mark` synchronizes CUDA before/after each stage boundary and
uses wall-clock intervals. Stages include fine assignment, representative update,
coarse grouping, hierarchy mapping, pooled Key, coarse QK/softmax/statistics,
budget, history lookup/filter/dedup, exploration, V fill, mask assembly, fused
sparse QK+softmax+AV, validation, QKV and projection. Source tensors/oracles are
exported **after closing** each layer's measured pipeline interval. Diagnostic
bookkeeping is explicitly identified. QK/softmax/AV are fused in the real operator,
so their individual times are not fabricated. No diagnostic or dense-reference
time is inserted into the formal warmed inference samples.

## Observations retained, not silently fixed

The all-24-layer, uniform-start 10% scheme is currently **not accurate enough**:
formal scene-macro AUC30 is 40.393%, versus dense 84.293% and full SAF3R 84.576%.
Only layer 14 gets a nonzero K_A (1 or 3 patch edges/query); layer 15 returns to
zero. Thus no actual history reuse occurs in this data. Reuse/explore lifecycle
is implemented and tested synthetically, but its benefit is not established here.
The instrumented layer-14 query-specific path is the main runtime bottleneck.
These facts do not isolate the causal source of final accuracy loss, nor justify
changing the requested alpha formula without another experiment.

## Reproduce

```bash
# Formal paired 24-layer run, one warmup + three measured forwards.
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/benchmark_hierarchical_value.py --gpus 0 1 2 3 4 5 6

# Seven independent 128-frame diagnostics; not formal timing samples.
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/diagnose_hierarchical_value.py --gpus 0 1 2 3 4 5 6 \
  --verify-formal outputs/hierarchical_value/7scenes_n128_r10_all24_uniform

# Small raw diagnostic, frame 0 and frame 20 (not uniform-to-2 sampling).
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/diagnose_hierarchical_value.py --scene chess --frame-indices 0 20 \
  --raw --dense-oracle \
  --output-dir outputs/hierarchical_value/7scenes_n128_r10_all24_uniform/diagnostics/chess_frames_0_20

PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/report_hierarchical_diagnostics.py \
  outputs/hierarchical_value/7scenes_n128_r10_all24_uniform/diagnostics
```

Captured P/C/S/labels/indices/provenance permit offline inspection and adjustment
of selection decisions. Exact logits outside the saved support are NOT fully
stored, and changing a mask changes subsequent features. Therefore new end-to-end
AUC cannot generally be evaluated without rerunning the model.
