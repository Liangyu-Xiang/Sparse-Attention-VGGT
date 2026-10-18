# Middle-layer Attention isolation: spatial region history

This document describes the historical309cd09policy. The subsequent
[history-first corrections and controlled oracles](region_history_corrections.md)
have a separate opt-in installer and results; archived masks below are unchanged.

This opt-in experiment supersedes neither the saved old implementation nor its
results. The earlier128-frame residual-history queue and its report waiter were
terminated at the user's request; its incomplete files remain intact.

## Backbone and inputs

- Original VGGT global attention0–9: Dense, including layer9.
- Global10–16: the new Attention-only branch. No Value edges are mixed in.
- Global17–23: Dense. Frame/local attention is unchanged everywhere.
- First validation: Chess seq-01 frames0/10, same checkpoint and processing as
  the previous two-frame visualizations,2082tokens including10specials.
- Final edge budgets:10% and20%, including all special Keys in each budget.
  They correspond to208/416total Keys and198/406patch Keys per Query.

## Spatial terminal cells and unchanged global coarse filtering

The old feature-driven Value partition is NOT used as Attention's terminal
fine cells. Attention uses a fixed, connected frame-local spatial grid:
14x18cells per image,504cells in total (~25% of2072patch tokens).
The cells average4.11patch tokens, with one actual spatial-center representative.
The Value hierarchy continues updating separately, even in Dense layers. Its
original15%-fine, feature-driven coarse construction (~17coarse groups here)
supplies global coarse filtering. Thus we preserve the previously tested coarse
Key recipe rather than silently replacing it with a new spatial coarse gate.
Each global coarse Key is the unweighted mean of its Value-fine representative
Keys. Heads retain their own Q/K; per-head softmax precedes averaging.

## History and search

Dense layer9 executes ordinary Dense SDPA. A chunked recomputation of its Dense
weights constructs a budget-sized patch seed support; a transfer ratio chooses
the highest-weight subset. This layer9 seed computation is diagnostic overhead,
not a claim that standard fused Dense attention returns its full probabilities.

At layers10–16:

1. Compute current global coarse probabilities; retain history edges in the
   top50% of coarse GROUP COUNT, not50% probability mass.
2. Preserve every such edge, including multiple distinct Keys in one fine cell.
   Count edges, not occupied cells, against the final budget. If history exceeds
   the requested budget, expand the budget instead of dropping history.
3. Remove all history-covered fine cells from exploration. Pool only unresolved
   fine representative Keys, separately for each Query.
4. A: use spatial2x5-fine parents (~10fine/parent), rank pooled-parent attention,
   and select descendants until the missing edge budget is filled.
5. B: rank those same parents and keep enough to cover at least50% of unresolved
   fine cells (or enough for the missing budget, whichever is greater). Refine
   retained parents into connected children containing2–3fine cells; rank pooled
   child Keys and select descendants to fill the remaining budget.
6. Each newly selected fine cell contributes ONE real representative Key.
   A last partly used parent/child is trimmed in fixed numeric cell order. There
   is no Q-to-fine-representative ranking pass and no fine-member expansion.
7. Compute actual indexed multi-head QK/softmax/AV only on the final mask.

The reference's50%/80% cuts are illustrative budget ratios, not simultaneous
fixed guarantees. Under a fixed10%/20% final budget and changing history size,
the final group cut is budget-driven. This boundary policy is explicit and
deterministic; it is not oracle fine selection.

History ablations:0%,25%,50%,100% of the final PATCH budget. Specials always
remain present but are not history candidates. Only int32 Key indices recur,
not old weights or score ranks. An additional explicit adaptive trial uses
`C(i)=1-m80(i)/coarse_count`, `history_count=floor(C(i)*patch_budget)`.
This concentration uses the global coarse distribution, never sparse entropy.

## Validation protocol

Two scopes distinguish selector errors from accumulated backbone changes:

- Controlled: every selector sees original Dense layer inputs. Its private
  Value/history state still advances, but its outputs do not replace Dense.
- Native: sparse10–16outputs feed subsequent layers; all other layers are Dense.
  Layer10Q/K must be bitwise equal to original Dense layer10Q/K.

For each middle layer report:

- Recall of original Dense top10% PATCH Keys (207targets per Query), fixed
  across the10%/20% budgets, with patch Chebyshev radii0,1,2,4. Neighborhoods
  never cross image boundaries. Mandatory specials do not inflate this recall.
- Same-budget exact patch recall (198/406targets); full masks include specials.
- Pre-projection multi-head output relative Frobenius error against Dense on the
  SAME layer input; native also reports error against original Dense trajectory.
- Real density, retained history, explored edges, intermediate logical probes.

Large neighborhood recall alone is not evidence of success: a radius4window
contains up to81patches, so a same-budget random support can already cover most
target neighborhoods. Report that baseline, not just the raw neighborhood score.
Two-frame timing includes diagnostic capture and oracle comparison, and cannot
be advertised as a128-frame speed benchmark or7Scenes accuracy evaluation.
Intermediate-probe counters count nonempty logical query/group relations, not
the exact FLOPs of the padded batched pooling implementation. Unresolved-Key
pooling and masked/padded work are additional costs, not free operations.

## Entry points

- `saf3r/patch/region_history_attention.py`: opt-in production branch.
- `scripts/evaluate_region_history_two_frames.py`: controlled/native two-frame sweep.
- `scripts/report_region_history_two_frames.py`: comparisons and exact-mask images.
- `tests/test_region_history_attention.py`: spatial connectivity, history rules,
  global concentration, pooling, and unmodified Dense-layer numerical tests.

Example:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_region_history_two_frames.py \
  --output-dir outputs/region_history_attention/chess_two_frames/budget20_A \
  --retention .2 --variant A
```
