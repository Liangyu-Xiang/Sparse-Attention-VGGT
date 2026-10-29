# Rep-baseline history and budget revision

Design source: [latest complete chat summary](reuse_rep_revision_reference.md).
The later history-estimation answer was unfinished and supplies no further definition.

All quality runs use seven 7Scenes scenes, identical uniform 128 frames/seed 0,
checkpoint, preprocessing, pose and unposed reconstruction settings. F1 threshold
0.05 m; voxel downsampling 0.0078125 m; CD=(ACC+Comp)/2 unsquared metres.
No settings are chosen using test truth. Canonical grid 2x2, query groups 32,
budget <=20%, clustering unchanged, four non-Rep history candidates/cluster,
same exact-expansion eligibility and 16 dual iterations. The budget is an upper
bound. No exact knapsack optimality is claimed.

| Run | Change from previous stage |
|---|---|
| dense | Already completed this cohort before user correction; retained for paired timing |
| saf3r_dino64 | Already completed this cohort before user correction; SAF3R DINO per-frame Top-64, ETH3D calibrated profile, not lazy Top-4 |
| original | Reuse previous completed output-proxy quality results; historical timing |
| cosine_distortion | Reuse previous completed cosine quality results; historical timing. Prior implementation: Drop=2pi, genuine-only history, estimated mass transfer |
| rep_a | Drop=pi, Rep-relative gain objective |
| rep_b | All previously executed patch edges are observed history, including seed and ordinary Reps |
| rep_c | Raw relative-logit evidence fused into sparse attention; current-grid calibration |
| rep_d | Same selector and relative evidence as C, original population execution instead of estimated mass transfer |

## Objective and duplicate keys

Rep is the feasible baseline: one key per fine cluster (F=B patch keys).
G(s)=D(Rep)-D(s), deltaB=cost(s)-1. Drop's total D=pi and gain is
-sum_j a_j*cos(z_j,z_rep). Negative similarities are NOT truncated.
History, History+Rep and Exact keep the previous tokenwise distortion formulas.
The allocator minimizes -G+lambda*cost with the same dual interval as the
absolute-loss formulation; subtracting a cluster constant does not itself
change the mathematical optimum. Defaults and historical controls are preserved.

In B/C/D all valid executed patch keys enter the observation bank. Keys equal
to the current Rep inform its prior but are excluded from the extra non-Rep
candidate list: Rep already makes that key selectable, and H+Rep must never
execute or charge for it twice. Candidate cap limits selectable extras, not
the full evidence bank. Special tokens retain their separate original path.
B saves measured conditional patch probability divided by execution weight,
separating single-key evidence from population/transfer compensation.

## Compact relative evidence and current-layer estimate

C/D compute delta_h=mean_(heads,queries in group)(raw QK_h/sqrt(d)
- raw QK_pooled_grid(h)/sqrt(d)) directly inside the existing forward kernel.
Pooling and mean use this layer's original K/Q, not population-biased logits.
One FP32 scalar per group/selected patch key is stored, with no full
head x query x selected-key logit bank. This is a geometric likelihood ratio
across heads/queries, not a measured next-layer probability. Arithmetic mean
Q is used for the reference (linearity); the sparse QK matmul uses the same
numerics as actual attention. Head-specific information is compressed away.

Next layer: R_h=exp(clamp(delta_h,-30,30)); unobserved tokens have R=1.
For each CURRENT canonical grid g, a_j=p_g*R_j/(n_g-m_g+sum_observed R).
This preserves current coarse grid mass even for clusters spanning grids.
The fine cluster mass pi is recomputed as sum_(j in cluster) a_j.
All observed keys, including present Reps and candidates outside Top-4, inform
the denominator. No test truth or extra current historical-QK oracle is used.
The fixed exponential clamp is solely numerical range protection.

Exact conditional grid mass summaries are still collected by the existing
second selected-QK pass, in C/D as well as the controls. Fused relative
statistics add no further QK pass. Eliminating that existing pass would require
changing these summaries or a separate attention kernel design; this experiment
does not claim to have eliminated all repeated QK. Its cost is included.

## Mass-transfer ablation and diagnostics

C uses assigned estimated mass / own estimated mass multipliers. D uses the
original packer: Rep population n-t, history unit weight, Exact unit weights;
History-only therefore has no residual compensation. No dropped cluster is
assigned to another cluster. C and D use the same selector algorithm, but full
network runs can naturally diverge at later layers. A separate diagnostic
compares both executions at IDENTICAL current Q/K/V and selected support,
without feeding the diagnostic output into later layers or quality scores.

Per-layer diagnostics include previous observed keys, current Rep overlap,
candidate count/cap discards, finite positive/negative gain sums and extrema,
selected gain, state counts and unused budget. Layer 0 is Rep seed and has no
allocation gains; its corresponding fields are zero by definition.

Unit tests cover cross-grid prior normalization, gain equivalence, unclipped
negative similarities, fused raw residuals independent of execution bias,
all observed Reps entering history, CPU/CUDA finite outputs and duplicate-free
supports. Full-size validation precedes quality runs. Diagnostic-only runs
record six unavailable metrics with an explicit reason.

Timing is one warmup plus three full forwards on the same scene/GPU for all
methods. All initialization/clustering/routing/relative evidence/mass transfer
and summaries count. Seven-scene macro speedup is the ratio of mean scene
warmed medians; per-scene speedups also accompany six raw quality metrics.
Previously published paper scores and timings are not local measurements.

User correction during execution: only A/B/C/D are scheduled. Redundant old original reruns were stopped and recorded as interrupted with six N/A fields; previous completed original/cosine quality is linked through portable relative symlinks. The already finished Dense/SAF3R controls are retained. Original/cosine timings are historical and not used as contemporaneous speed comparisons.
