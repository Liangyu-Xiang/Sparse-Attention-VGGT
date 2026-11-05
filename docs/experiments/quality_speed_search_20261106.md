# RGB-only quality / latency search on corrected History A

Base:94d29a14,History A+all five special queries+Exact mask stride repair.
The user requests a finite hyperparameter search and better pose/reconstruction
at no additional inference latency, preferably with fewer retained keys.
This study does not claim a global hyperparameter optimum.

Search60 frozen configs:cluster fraction4/6/8/10/12/16%;Value features
head_mean or head_sketch4 (16heads*4=64dimensions, same width as head_mean);
retention profiles middle10/22/10%,flat16%,front20/16/12%,lowmiddle8/16/8%,
lowflat12%. Profiles use layer partitions0–5/6–17/18–23;every cap <=25%.
History loss/probability/execution unchanged, except the explicitly searched
feature representation and layer/cluster budgets. Source width32,group256,
16dual iterations,one local update and all engineering options remain fixed.

Calibration:64 RGB frames ScanNet scene0000_00,64 RGB frames TUM sitting
halfsphere, disjoint by exact path from both final evaluation frame lists.
Verify image decoding first; corrupted RGB excluded with concrete reason.
Validation:24 available chessseq03 frames (different sequence from benchmark
seq01),128 TUM walking halfsphere frames. No GT loaded in either stage.
These are independent RGB diagnostics, not held-out geometry quality results.

Per candidate:one warmup+two full forwards for screening, then a separate
instrumented forward. Rank by equal-sequence mean of24-layer Wo-projected
relative local Attention squared error. Exact targets use current sparse QKV
and32 deterministic patch queries, full key bank. They are sampled same-layer
teachers, not a full Dense model trajectory and not a guarantee of final scores.
All six geometric fields N/A on diagnostic runs. Dense/SAF3R controls not rerun.
No7Scenes/ETH3D test quality is read to select candidates. Current configuration
is measured in each search shard for same-card calibration speed filtering.
Freeze finalists before the first target quality run. Time on128-frame RGB
validation using1warmup+3repeat medians,discard anything slower than current.
Final quality:once each7Scenes7x128 and ETH3D11x14–76,all six metrics. Preserve
failure outcomes;do not tune against them. Existing controls are historical.

Key diagnostics compare selected dense probability mass,unweighted/weighted
output errors,and adding16 omitted high-probability Keys as an offline oracle.
The latter increases budget and never runs inside the inference algorithm;it
cannot be reported as a deployable speed/quality improvement. Results identify
whether support omission or compensation accounts for the larger local error.
Cost of all diagnostics/search is separate from frozen-candidate inference.

No extra Dense timing authorized. Formal relative Dense speedup is N/A.
Comparison to previous6.090130s7Scenes/1.264160sETH3D batch is historical unless
the changed algorithm's reference is paired inside the current experiment.

## Independent modelling phase2 (before target quality)

A's loss uses estimated p(h), but population-hcount compensation assumes
uniform member probabilities. Add opt-in rep_history_execution=prior_residual.
Rep-state weight=pi_cluster/prior_rep. History-state non-Rep weights1;
Rep weight=(pi_cluster-sum(nonrep historical prior))/prior_rep. Exact weights1.
The old A support/loss/budget/selector stay unchanged, with a different causal
trajectory due to these weights. Uniform prior recovers original population
weights;single Rep history equals Rep. Predicted cluster mass is conserved,
not necessarily true Dense mass. Positive numerical guard clamps ratio >=1.
Historical mass is reduced in the existing correction kernel, no extra QK or
per-member nearest-neighbor ownership.15 CPU/CUDA loss/layout/special tests pass.

Also test candidate_fraction=1 against.25. All clusters can compete for Exact
within identical edge budgets. Top-F eligibility is implemented as an all-true
mask, mathematically avoiding sparse-mm,full sort and mask scatter for this case.
This is a modelling change in eligibility, with an equivalent faster execution
of the all-eligible case. Both changes are default-off,separate factorial trials.
26 phase2 configurations are frozen:8/10/12% clusters,population/prior_residual,
Exact25/100% eligibility,mean16/12% retention,and two10% head-sketch all-eligible
trials motivated by phase1 RGB error. This extension uses no target GT.

## Phase3 and frozen final candidates

Seven trials (including current):refine10% head-sketch/all-Exact middle budget
18/19/20/21%,and16% head-sketch/all-Exact flat14/15%. All selected using prior
RGB evidence,never target GT. Independent24-frame chessseq03 and128-frame TUM
walking validation uses1warmup+3medians. Require both lower local error and no
higher latency on BOTH validation sequences. Freeze the lowest-error endpoint
(p3_sketch10_mid21) and fastest improved endpoint(p3_sketch10_mid19),before
any target test quality. Source seed7351,sketch4dimensions/head. Primary and
alternative roles are fixed;do not pick a test-score winner retrospectively.

Mean retention caps15.5%/14.5% vs old16%;fine count10% vs old12%. First/last
six layers remain10%,middle12 layers21/19%. No Dense fallback or frozen layer.
Keep original population compensation:the new prior_residual alternative was
not selected. All original probability and A loss formulas remain for endpoints.

Kernel tuning54 block/warp/stage combinations at two retention shapes,including
history probability accuracy,keeps original64-query/128-key/4warp/2stage winner.
Microbenchmark times are not full-model acceleration claims.

Opt-in skip_dead_exact_observation removes only unread full-eligible computations:
cluster-grid overlap sparse-coalescing,coarse residual demand,patch-only coarse
normalization,and fallback counts/arrays that are fully overwritten by actual
history evidence at stride1. It requires the supported full-eligible shared path.
Real8-frame four-forward validation found identical masks,pose,depth and
confidence bit-for-bit,including self repeat. Defaults remain false. This
engineering change is included in both frozen endpoint configs.

GPU7 was lost by NVML at the end of one validation. Exclude its validation
entirely;retry on healthy GPU0. Other queues use independent monitored GPUs.
The15 CPU/CUDA loss/layout/special tests pass again on GPU0. No reset of the
faulty shared device is attempted. Monitor queries only the assigned GPU index
so unrelated GPU7 errors do not corrupt healthy-card process monitoring.

## Independent Query-128 extension

The aligned fixed-budget Key diagnostic shows most helpful missed keys are in
clusters already represented,and most were not previous-layer history. A single
Probe/Mask over256Queries can miss query-specific peaks. Open a separate19-row
RGB-only family (including unchanged current reference):Query groups128 and
aligned fully-sampled history interval128;clusters6/8/10%;head_mean/head_sketch;
flat12%,flat14%,or10/18/10% retention. Fewer clusters/edges offset twice the
number of decisions;Query count and sparse Attention kernels are unchanged.
All-Exact eligibility and validated removal of unread observations remain.

The existing Query-256 primary/alternative stay frozen,run once and are not
retuned. Query-128 candidates are ranked ONLY on independent ScanNet/TUM RGB
local error and same-card time,then separate RGB validation if eligible. This
is not choosing budgets against7Scenes/ETH3D test quality. No additional control
quality/timing reruns. The API now accepts history_query_interval128 using the
existing generic stride and probe kernels;defaults remain unchanged.

## Current-probe backend:independent information/cost experiment

The frozen Query256 endpoints and Query128 trials remain separate. Their full
results are retained even where geometry regresses. The aligned independent
single-Key counterfactual found917 helpful candidate omissions,876 in a cluster
already represented and799 not previous history. This motivates current evidence,
not any7Scenes test-based budget choice.

Opt-in current_probability_probes=1/4 observes full-Key Softmax for1/4 sampled
Queries per group. A Triton online-LSE pass and a streamed head/probe-mean pass
produce only GxPatch float32 probabilities,never HxGxKey probabilities. Aux QK is
P*G*(N+Patch) per head;the main sparse Attention QK/AV is executed once with no
history export. Exact sampled probabilities replace the Grid/M5c estimator;
previous support IDs still supply History A candidates. Four-state distortions,
support H union Rep and16-round budget solver remain. No dense layer fallback.

With current p,Drop and Rep scalar weighted distortions are reduced directly
from cluster members. History A subtracts exact current historical p times Rep
distance. Population and predicted-residual compensation are separate trials.
Layer0 now uses the same three available states Drop/Rep/Exact under current
probabilities,not the old global uniform seed;this is an explicit modelling
change. All5 special Query outputs remain dense-corrected. VV clustering updates
with current features each layer;no frozen cross-layer partition.

19 configs including current:8/10% clusters,1/4 probes,population/prior_residual,
head_sketch,flat14% or10/18/10% budgets,plus two10% head_mean ablations. Query
sharing256. Graph cache cap24 for these configs within existing2GiB device cap.
This is a family comparison,not an isolated proof for only one formula change.

Crucially,calibration and independent validation score HELD-OUT Query positions
1/16,5/16,9/16,13/16 of the shared group. Observed probes are1/8,3/8,5/8,7/8.
Both the current reference and candidates use the held-out scoring positions.
No local exact-observation teacher is evaluated only at its own observed rows.
No final target geometric truth participates in ranking.6 CUDA numerical tests
pass:FP32/BF16 exact-probe probabilities versus full softmax,full weighted state
statistics,A support/predicted-mass conservation,and no-export sparse AV.

This improves information acquisition by adding small full-Key probe QK while
removing older probability-estimation machinery. Total physical QK including
both probe passes is reported separately;fewer main retained Keys does not by
itself prove fewer total QK. Both passes and all preparation count in full time.

## Diagnostic tail correction (frozen model unchanged)

Initial nominal held-out Query positions could clamp to the final token and
coincide with an observed tail Probe. Correct the diagnostic to exclude EVERY
observed clamped row,for both candidate and reference. Supplement the ALREADY
FROZEN current-probe candidate only;do not reselect parameters or rerun target
quality. Strict RGB validation still passes (chessseq03 time/error ratios
.89215/.89519,TUM128 .94485/.55265). Preserve original diagnostic files and
record the correction separately in current_probe_strict_validation.json.

Probability accuracy must likewise use the actual four clamped observed Query
IDs,including duplicate tails and any special Queries,not a deduplicated
Patch-only diagnostic set. Corrected real64-frame/24-layer candidate-pool max
relative probability difference is2.19e-6. Old derived2.1% discrepancy was a
mismatched diagnostic target;raw old/corrected records are both retained.
The new method retains709/917 old helpful candidate edges (77.32%),covering
97.57% of the sum of their individual gains. This is finite-pool support overlap
across different trajectories,not a proof of additive/joint final geometry gain.
Coordinates and original RGB paths are in critical_keys_current_probe.csv.

21 CUDA tests now pass across new probability kernels,A/history compensation,
old shared layout/special paths and sparse AV. Full new backend assessment is
18 complete quality runs;previous two changed endpoints have36 runs. All six
fields are present for quality and explicitly N/A for diagnostic/test records.

## Last bounded budget-only sweep

With strict independent RGB validation5.746s vs6.081s,measure whether spare time
can buy better fidelity under the SAME original latency ceiling. Keep all
current-probe probability/state/cluster settings fixed. Six schedules:10/20/10,
10/21/10,flat16,flat17,12/21/12,13/21/13. Every layer cap <=21%,even with both
probe scans plus dense-special work total logical QK <25%. Also retain original
A12 and frozen14% references (8 rows including references). Rank only independent
RGB held-out errors/time;validate before any further changed-method quality.
The14% version and its complete quality are reused,never rerun for selection.
This is the final planned parameter expansion;no7Scenes test-score tuning.
