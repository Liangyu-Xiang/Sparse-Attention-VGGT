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
hashes or transplant the previous6.161s number onto this method.

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
same-hardware effect of engineering changes. All results currently pending.
