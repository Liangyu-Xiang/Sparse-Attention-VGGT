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
