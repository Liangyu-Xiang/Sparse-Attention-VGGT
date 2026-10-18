# History-first region Attention corrections and controlled oracles

## Scope

This revision responds to the referenced “分析远程仓库问题” review of309cd09.
It does not silently replace archived results or resume the cancelled128-frame
queue. Test inputs remain Chess seq-01frames0/10, final10%/20%budgets including
mandatory specials, Dense global layers0–9/sparse10–16/Dense17–23.

The legacy `install_region_history_attention` remains available for exact replay.
The new `install_corrected_region_history_attention` is an explicit non-oracle
entry point. It defaults to A(one-level exploration); B is optional. Value
hierarchy maintenance, spatial fine grid, per-head softmax-then-mean selection,
and the non-sparse Dense layers are preserved.

## Non-oracle implementation changes

1. **Hard history**: every incoming history edge is mandatory, even outside the
   current coarse gate. Distinct edges sharing a fine cell all survive; that
   cell is excluded from exploration. History is the previously defined
   high-confidence subset (fixed50%in this experiment), not all previous edges.
   If retained history exceeds the nominal patch budget, preserve it and expand
   the budget. No historical attention weights are passed between layers.
2. **Population bias**: use `Q dot mean(representative Keys) + log(real eligible
   member count)` before per-head group softmax. Population is the true number
   of tokens in unresolved fine cells, not merely their fine-cell count. This
   fixes missing cardinality but is still NOT logsumexp and cannot recover
   intra-group logit variance. Global coarse concentration also uses population
   bias in the corrected installer; fixed-ratio ablations do not depend on C.
3. **Query-dependent terminal bridge**: after selecting fine cells, evaluate
   only their local real-member QK(typically2–6members/cell). Each head performs
   a softmax within that cell; mean-head probabilities choose ONE actual Key.
   This is not a Dense-best-member oracle: it has no global Dense normalizers,
   and does not rank all fine representatives or all remaining Keys. It removes
   the forced spatial-center landing point while preserving one new edge/cell.
4. **Avoid unnecessary default refinement**: A is the corrected default. B
   remains for controlled comparison, not an assumption that extra hierarchy
   always improves quality. In this example B's20%gain is negligible, while
   its10%output error is slightly better than A.

The local terminal costs additional real QK. Its winning logits are currently
recomputed during final indexed attention; counters explicitly include the
terminal probes. On the two-frame B run, final+terminal QK alone amount to
~29.78%/60.04%of Dense real QK at10%/20%final retention, before pooled probes
and pooling costs. These are logical executed relation counts, not a FLOPs or
latency benchmark. There is NO demonstrated speedup claim in this revision.

## Four controlled diagnostics

All methods see original Dense layer inputs. Each recursive method has its own
history initialized from Dense layer9, with identical50%transfer. Old masks
must reproduce309cd09bitwise; old output errors must reproduce within1e-5.

- **Hard-history only** changes just the gate rule.
- **Exact group-mass** changes just parent/child scoring to real eligible-member
  logsumexp. It uses Dense member logits explicitly and is diagnostic-only.
  Per-head mass normalizes over current eligible groups BEFORE head averaging.
- **Best-member frozen** freezes the source policy's retained history and
  selected fine cells at every layer. Only the terminal Key is replaced by the
  member with highest Dense mean-head attention. Its changed Keys do not alter
  the source policy's later histories. This avoids conflating region selection
  and history evolution with the terminal substitution.
- **Value-aware diagnostic** compares uniform, shared Dense attention TopK and
  shared mean-head single-deletion influence ranking at identical total budget:
  `a/(1-a) * norm(V - O_dense)`. Influence uses FP32 Dense probabilities/output.
  The identity is exact for deleting ONE edge; top-ranked influences are not
  an optimal multi-edge support oracle and may interact after renormalization.

Additional ablations combine hard history, cardinality, exact mass and terminal
selection. “Full oracle” combines hard history/exact mass/Dense-best-member but
is still NOT an optimal output-error oracle. No Dense oracle is reachable from
the production installer; oracle inputs must be supplied explicitly.

## Metrics and factual corrections to the review

- Primary: relative Frobenius output error before the attention output
  projection, averaged over the7middle layers. All methods share the same
  BF16Dense SDPA reference on the same inputs. Mean-head NMSE is also saved,
  but it is a DIFFERENT aggregation and cannot be substituted numerically.
- Exact recall targets the original Dense top10%PATCH Keys(207/Query) in both
  budgets; specials do not artificially increase recall. Same-budget recall is
  separately recorded. Retained total mass and patch-only mass are distinct.
- The existing309cd09selector already computed per-head softmax before head
  averaging. The review's statements about mean-Q/K predictors refer to other
  diagnostics and must not be attributed to this implementation.
- Current same-protocol diagnostics do NOT reproduce the quoted conclusion
  that uniform outperforms Dense attention TopK: in this input/layer/mask/metric
  protocol TopK is better than uniform, while influence improves further.
- Hard-history alone yields a small improvement. Exact group mass alone does
  not improve this sample, so “weak coarse gating is the main failure cause”
  and “exact group scoring will fix it” are hypotheses, not established facts.
- Four errors do not add linearly; ablations expose interactions rather than
  a mathematically additive error decomposition. No128-frame/AUCclaim is made.

## Results and reproduction

See [complete report](../../outputs/region_history_corrections/report/report.md).
Controlled B output errors at10%/20%:

| Method |10% |20% |
|---|---:|---:|
| Legacy |41.41% |31.43% |
| Hard-history only |41.14% |31.33% |
| Exact mass, fixed centers |42.15% |31.61% |
| Original regions + Dense-best-member frozen |37.63% |27.60% |
| Corrected B, no Dense oracle |37.51% |27.61% |
| Shared Dense attention TopK |38.82% |26.09% |
| Influence TopK diagnostic |36.67% |24.71% |

Native sparse propagation separately validates exact history preservation for
EVERY Query, unique budgets/specials, Dense prefix bitwise identity, and repeated
pose bitwise identity. Its local errors differ from controlled inputs and are
reported separately. These are TWO-FRAME diagnostics, not dataset accuracy.

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_region_history_oracles.py \
  --retention .1 --output-dir outputs/region_history_corrections/chess_two_frames/budget10
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python scripts/diagnose_region_history_oracles.py \
  --retention .2 --output-dir outputs/region_history_corrections/chess_two_frames/budget20
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python scripts/validate_corrected_region_history.py \
  --output-dir outputs/region_history_corrections/native_two_frames
PYTHONPATH=. python scripts/report_region_history_corrections.py outputs/region_history_corrections
PYTHONPATH=. pytest -q tests/test_region_history_corrections.py \
  tests/test_region_history_attention.py tests/test_region_stage_capture.py
```

Outputs refuse existing completed result paths; use a fresh directory for reruns.
Legacy results, source snapshots and old stage masks remain untouched.
