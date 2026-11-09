Current unified-beta historical-importance candidate
===================================================

This is an opt-in experimental snapshot, not a promoted default. It did not
meet the requirement to dominate SAF3R DINO Top128 and FastVGGT in all six
quality metrics. No 7Scenes test truth was used to tune its parameters.

Core code
---------
saf3r/patch/beta_fast_subgroup_forward.py: current forward path.
saf3r/patch/beta_unified_importance.py: unified all-layer beta and global correction.
saf3r/patch/beta_fast_importance.py: fused historical head-mean normalization.
saf3r/triton/beta_subgroup_reuse_attention.py: reuse formal attention exponentials.
saf3r/patch/beta_unified_selection.py: same four-state selector via refreshed graphs.
saf3r/patch/beta_head_batching.py: DPT frame batching, retaining all heads/precision.
configs/evaluation/beta_unified_subgroup_20261109.json: exact frozen configuration.

Model
-----
C_l(g,b) is the current head-mean ordinary-Key block mass; u_l(g,j)=C_l(g,b)/n_b.
The current observation uses four contiguous 64-Query statistical means inside
each 256-Query routing group. This covers all ordinary Queries; it is not four
individual Query probes. Each head/subgroup is normalized before aggregation.
Key target256 is rounded into balanced frame-local blocks: 207-208 ordinary
Keys/block on the recorded7Scenes inputs,231-232 on recordedETH3D inputs.

beta_b=(min(C_l,C_prev)/max(C_l,C_prev))^0.5, identical in all24 layers.
For previous actual support S:
    w_l(j)=u_l(j)*(h_prev(j)/u_prev(j))^(0.5*beta_b), j in S;
    w_l(j)=u_l(j), j outside S.
Normalize across ordinary Keys before the unchanged selector. There is no
per-block mass conservation lock or residual centering.

h_prev is estimated from actual previous sparse QK exponentials, with an
observed-plus-coarse-unknown partition per head/subgroup, then head-mean.
It is not selected-only SparseSoftmax or true Dense probability. No cross-layer
head alignment, inherited unexecuted members, older rolling cache, or extra
current fineQK probes are used. Reps enter history only if actually executed.

Value clustering, Drop/Rep/History/Exact,10/18/10% layer budgets,cluster_fraction
0.1,and sqrt(population*prior-residual) execution compensation remain unchanged.
Prediction heads remain FP32,backbone BF16 autocast. 7Scenes uses head chunk32;
ETH selection evidence used original head chunk8. Alpha=tau=0.5 in both.

Run one scene with complete quality and full-forward timing
---------------------------------------------------------
Use the existing VGGT environment/dataset dependencies and the same checkpoint.
The expected-input JSON verifies exact128-frame IDs,shape,and checkpoint hash.
Run from repository root, selecting an allowed GPU4,5,or6 exclusively, e.g.:

CUDA_VISIBLE_DEVICES=4 python experiments/beta_unified/evaluate_current.py \
  --dataset 7scenes --scene chess \
  --expected-input outputs/beta_unified_current_20261109/7scenes/chess/input.json \
  --config configs/evaluation/beta_unified_subgroup_20261109.json \
  --output outputs/my_unified_run/chess

For ETH3D use --dataset eth3d and its saved scene input JSON. Input paths/checkpoint
must refer to the corresponding locally installed files. Output must be new.
Run each method in a separate process: the experimental installer patches the
process-local current-probe forward dispatcher. The repository default is unchanged.

Reported local quality
----------------------
outputs/beta_unified_current_20261109 contains unrounded six-metric tables,
per-scene raw results/configuration/layer statistics,selection provenance and
numerical checks. AUC/F1 are fractions in JSON/CSV,percentages in human tables;
distances metres,CD=(ACC+Comp)/2 unsquared.

7Scenes all7 x128: AUC3=22.75%,AUC30=84.36%,ACC=0.15502m,Comp=0.12869m,
CD=0.14185m,F1=46.87%. Threshold0.05m,downsample/voxel0.0078125m,TSDFtrunc0.04m.
ETH3D all11:26.09%,79.06%,0.30878m,0.85127m,0.58003m,61.96%,respectively.
ETH threshold0.25m,downsample/voxel0.0390625m,TSDFtrunc0.2m.

The original final7 quality batch measured5.788927s/128frames,19.660893GiB warm
peak,1cold+3warm fullforwards. This is a historical snapshot,not the newly
authorized paired comparison. The paired same-GPU Dense/quick/candidate timing
campaign is recorded separately under outputs/paired_fullforward_7scenes128_20261109.
It does not rerun pose/reconstruction; all new timing-only quality fields are N/A.
Completed paired results: Dense12.328655s(1.00x),quick5.760001s(2.140x),
candidate5.785011s(2.131x),equal-scene mean of warmed medians. Candidate is0.434%
slower than quick in this batch; strict macro speed gate is not passed. Warm
allocated peaks:14.360GiB,14.908GiB,19.723GiB,respectively. No precision or model
parameters changed during this timing-only campaign. Existing SAF3R/FastVGGT
timings remain separate batches and are not given paired speedups here.

SAF3R comparison is DINO-feature Top128 PER FRAME,not officialTop4 or lazy
correspondence-head TopK. FastVGGT comparison is local native upstream6526e275,
merging0,ratio0.9,not claimed paper-equivalent. No published scores or H100 timings
are substituted for local measurements. See seven-scene report and strict gates.
