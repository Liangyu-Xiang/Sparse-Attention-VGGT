# Guarded shared-bank acceleration: preregistered experiment

User-approved goal: seven-scene128-frame evaluation, speed relative to a NEW
same-input/GPU Dense warmed timing >=1.5x if possible; quality is mandatory.
No extra dataset and no parameter search using7Scenes truth. Exactly one final
seven-scene quality evaluation after implementation-only validation/freezing.

Quality reference is the completed compact implementation in
outputs/reuse_compact_7scenes128_20261029. Each of its six equal-scene macros
defines a relative1% gate: AUC/F1 >=.99*reference, distances <=1.01*reference.
The exact unrounded reference values, source hashes, inputs and thresholds
must be frozen before the final quality test. CD=(ACC+Comp)/2 unsquared metres;
F1 threshold.05m and voxel.0078125m remain unchanged. Depth is additional only.

## Fixed algorithm settings

- Original maximum20% fine-cluster count; layer caps15% atL0–9/L17–23 and20%
  atL10–16, including all special keys. L0 reduced-cap selection is Rep/Drop.
- Eight dual iterations normally; complete16-iteration20% allocation on a
  guard failure. All eleven states, Top-4 cap, cosine objective and transfer
  assignments retained.
- Reuse support/weights for at most one intervening layer. Force refresh at
  L0, L10, L17, two-layer age or when a cached cap exceeds the requested cap.
- Each layer checks four deterministic rotating query groups, one query at
  offset16 (clamped in a short tail), using CURRENT Q/K/V and actual incoming
  history. Candidate output is compared with a complete20%,16-step allocation
  on the same sampled groups, or original full Rep seed atL0.
- Guard uses MAXIMUM per-group relativeL2 over all heads: <=.005. Near-zero
  reference norms clamp at1e-8. Nonfinite or larger errors trigger a whole-layer
  full refresh; fallback cost is measured, not excluded. No GT enters guards.
- Layer budget, actual support density, cache age, guard errors/probes,
  fallback counts and both sampled allocation/attention costs are recorded.

## Shared attention and exact evidence

Current representatives form a contiguous global bank, sorted by canonical
grid. Extras use per-group indexed segments, also grouped by grid, with their
original logical support slot preserved. This includes Exact members which
happen to be current Reps in the shared bank. A physical inactive key is not
an observed semantic edge. Grid routing never duplicates a selected key.

The initial64-query common-bank kernel was too slow in truth-free synthetic
microbenchmarks. The equivalent production tiling uses16-query x64-key tiles,
four warps, eight disjoint grid partitions; two16-query tiles still belong to
the SAME32-query group. Extras use32-query x32-key tiles. This changes only
execution tiling, not groups, budgets or guard thresholds. Both retain
FP32 normalizers and accumulators and merge with log-sum-exp into one softmax.
Specials are always in the shared bank and excluded only from patch summary
normalization. Biased and raw within-grid log normalizers recover the same
group-mean weighted and unit-key grid masses WITHOUT a second QK pass.
Raw relative logits are generated in forward and returned in logical order,
maintaining historical tie breaking despite physical reordering.

Scratch is bounded by the64-group window; no full query x selected-key logit
tensor or per-group duplication of the full representative K/V bank. Original
indexed backend and defaults are preserved as numerical references.

## Validation and reporting

CPU/CUDA references cover partition softmax, evidence, empty patch support,
specials, tails, masks, ordering, cache cap transitions and both guard branches.
Real128-frame intermediate checks compare current-input old/new executors
(FP32 small cases1e-5; BF16 relativeL2 <=.005). No quality score is consulted
while selecting equivalent implementation tiles or fixing numerical issues.

One warmup and three full-forward samples per scene/method. Dense is timed
fresh without recomputing its quality; the existing Dense quality is explicitly
reused. Pair exact input/hardware/protocol, preserve source snapshots, all raw
samples, peak allocated/reserved memory and six quality fields. Diagnostics,
OOM or interrupted runs have six N/A fields with concrete reasons.
Freeze then evaluate new method once on all seven scenes. Never adjust the
fixed approximation settings after reading that final quality. A missed
quality or speed target is reported as a failure, not relaxed or hidden.

## Exact allocation optimizations before the final freeze

Equal-cost states are contracted to their minimum error, preserving ORIGINAL
state-index tie breaking (including Exact duplicating a small integer cost).
Sixteen-step reference decisions are unchanged. Only states already invalid
with infinite cost/error skip useless raw-loss atomic additions.

The NORMAL eight-step solver uses a deterministic feasible upper bracket.
For each cluster let e1 be its minimum unit-cost error. The largest
(e1-e_s)/(cost_s-1) over cost>1 states excludes extra-cost winners. When B<F,
also use the (B+1)th largest positive Drop-to-unit benefit as a bound; take the
smaller of this feasible upper and the original all-Drop upper. This is derived
from current operating points, not a calibrated hyperparameter. The full16-step
reference retains its original interval.

Cold/failed preflight and intermediate kernel benchmarks remain diagnostics,
not quality experiments or contemporaneous full-forward speed claims. The first
preflight completed forward but failed vector-metadata serialization; vector
records are now serialized as lists of scalar tensors outside forward timing.
