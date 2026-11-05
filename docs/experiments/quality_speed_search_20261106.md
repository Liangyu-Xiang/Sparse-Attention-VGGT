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
