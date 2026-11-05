# Correct baseline: History Loss A + all-special Query correction +12% clusters

This task corrects the mistaken legacy-main base of the previous speed campaign.
Base commit10e5656f contains the exact A_all_specials reference (ETH3D macro:
AUC3 22.429048%,AUC30 78.868713%,ACC .317107m,Comp .864698m,CD .590902m,
F1 55.604616%). It already includes Exact mask stride repair.

The candidate keeps rep_history_loss=A:History supports H union Rep,non-Rep
historical Keys weight1,Rep weight population-(union_count-1). It keeps all five
special Queries per frame dense-corrected and all special Keys permanently
retained. Cluster fraction is .12. Layer retention remains10%/22%/10%,query
sharing256,one current-Value local update,16 allocation iterations. No new loss,
probability formula,mask pruning or GT-guided tuning is added.

Only engineering options from the previous fast implementation are ported:
masked GPU clustering stores,allocator cache reuse,fused RoPE with native dtype
roundings and per-input raster extent,fused residual with explicit FP32 rounding,
Attention pipeline stage2,and cuDNN autotuning. Defaults remain off;correct
reference.json is kept separately. Autotuning cold latency/memory is recorded.
The A loss/window selector/history/grid/Exact stride/special-correction source
modules are byte-identical to10e5656f;semantic_source_audit.json records hashes.

Use the actual reference tracker-fixed checkpoint,b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969,
from pinned official revision4d9d9494b2211c90ab9cd3c8c6aa188fe363103d. The previous
main checkpoint symlink to a different file SHA is not overwritten. Both current
SAF3R timing and changed-method quality use the b08 file;do not mix checkpoint
hashes or transplant the previous6.161s number onto this method. A full tensor
comparison found that the two checkpoint files differ only in track_head keys;
all1403 equal tensors and zero differing nontracking tensors are recorded. The
tracking head is not called by these pose/reconstruction runs, so file-SHA
differences alone do not explain prior geometry quality differences.

17 History/layout/special/RoPE tests,18 residual cases,16 cluster trajectories
pass. Real pipes14-frame diagnostic passes all24 layers,70 dense special Queries
per layer and12% clusters;all six geometry scores are N/A for this diagnostic.

Frozen quality evaluation:all7 existing7Scenes entries128frames and11 ETH3D
entries14–76frames;pose+recon_unposed,six metrics,per-scene and equal-scene macros.
Only changed-method quality runs. Unchanged SAF3R gets authorized paired timing
only;Dense is not rerun. One warmup+3 complete-forward measurements;GPU4/5/6
only,strict same-card scene pairs. Historical quality is explicitly separate.
7Scenes has no existing complete matched A_all_specials reference;do not use the
legacy-main score as its2% degradation reference. Historical ETH3D reference is
RTX4090/Torch2.8 vs current5090D/Torch2.9;reproduction drift is not a controlled
same-hardware effect of engineering changes. Results are archived under outputs/history_A_special5_cluster12_speed_20261106;
report.md and summary.json contain completed quality, timing and audit outcomes.

Completed:18 changed-method full quality evaluations plus18 authorized SAF3R
timing-only runs. Audits pass for input/checkpoint/protocol equality, same-card
exclusive timing, 24-layer cluster counts and frozen source/config.
7Scenes:6.090130s vs SAF3R6.705480s (1.1010x),peak warm allocated14.847GiB.
ETH3D:1.264160s vs SAF3R1.603858s (1.2687x),peak warm allocated10.996GiB.
All scene pairs are faster. Dense paired speedup N/A:no authorized current run.

Quality acceptance is NOT achieved:ETH3D historical-reference ACC +13.191%,
Comp +6.010%,CD +7.937%,F1 -6.333%. These exceed2%;hardware/framework differs
from historical reference,so this is not an isolated engineering ablation.
7Scenes has no matched complete A_all_specials12% reference to establish a2%
change. No quality-guided retuning or extra controls quality reruns were made.
Cold autotuning averages20.975s/20.509s for7Scenes/ETH3D,reported separately.
