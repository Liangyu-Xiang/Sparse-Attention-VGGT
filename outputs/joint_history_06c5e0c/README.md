# Joint History-conditioned optimization evaluation

Pinned remote:06c5e0c760b34b34c3f1a715a99417dd0406fd76.
Default config physical_overlap_accounting=true,joint_history_exploration=true,
retention.20,group32,four probes,canonical2x2,unchanged Value and dual solver.
The fixed candidate-slot/fresh-QK reuse plan is NOT implemented in this commit.
No production files/configurations are modified by this evaluation.

Stage0:15 upstream CPU/GPU tests passed with the existing evaluation-only FP32
sparse.mm/four-effective-probe Tensor Core compatibility wrapper. py_compile passed.
First run a real two-frame24-global-layer preflight. Then run seven scenes with
exact historical128-frame inputs/checkpoint/preprocessing/protocols,seed0.
Each scene worker runs fresh Dense1warmup+3measured whole forwards (timing-only,
six quality fields explicitly N/A), then the new method with identical input,
same GPU UUID and timing protocol plus pose/recon_unposed quality evaluation.
Monitor foreign GPU process IDs; flag contaminated timing pairs and report
speedup N/A rather than treating them as uncontended measurements.
All idle non-quarantined GPUs allowed,one worker/card,no unrelated termination.

Quality: AUC@3,AUC@30,ACC,Comp,CD,F1 per scene/equal-scene macro only after7/7.
Reconstruction metres,CD=(ACC+Comp)/2 unsquared,threshold.05m,downsample.0078125m.
Full latency includes every initialization,clustering,selection,history/reordering
step. Dense fresh timings are separate from historical Dense quality records.
Report peak allocated/reserved,cold/warm latency,actual density and paired speedup.
SAF3R frozen local lazy per-frame Top4/ETH3D profile,not paper DINO reproduction.
No7Scenes-truth parameter search,no dropped layers/no silent budget changes.
Old quality methods are not rerun. Data and weights are never uploaded.
Evaluation output remains local unless the user subsequently requests remote sync.
