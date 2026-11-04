# Integrated one-Probe sparse history at Query intervals32 and256

Base `b0819004`,independent branch `codex/sampled-history-integration-20261104`.
The user requested integrating both32/256 Query intervals into the full algorithm
and measuring runtime and quality. Configs:
`configs/evaluation/reuse_sampled_history_32.json` and
`configs/evaluation/reuse_sampled_history_256.json`.

## Frozen method

One historical Probe at Query offset4,per Group of32Queries.32 samples every
Group;256 samples every8Groups. Windows remain64Groups and align with both phases.
The inline exporter computes no additional selected-Key QK. It exports debiased
log sparse scores,not the probability-only variant used for the earlier ~3%
kernel benchmark. Full integrated timing must be measured independently.

For sampled Groups,export `logmeanexp_head(raw_QK - final_compensated_sparse_LSE)`.
Complete missing Keys using pooled-Key scores in the same per-Head sparse-LSE
scale. Stable per-Grid log-sum-exp produces conditional estimates and unobserved
uniform density. Cache ONLY sampled Groups,ONE Probe,in signed FP16 log format,
plus FP32 coarse/unobserved statistics. Next layer applies unchanged M5c beta(1,2)
to these conditionals and its current coarse Probe0 mass. This is an explicit
one-Probe estimator change versus the previous four-Probe probability model.

Unsampled Groups have NO measured Attention probability history. Their prior is
their own current coarse Grid mass divided uniformly among that Grid's members.
Their own previous support remains eligible for the existing selector. No neighbor
IDs/probabilities are copied across Groups with different sparse masks. Residual
candidate summary for these Groups uses own coarse patch mass and selected-count
fraction;sampled Groups use the actually exported compensated/unit-weight scores.

The coarse observer retains FOUR Probes for residual candidate exploration;
history uses coarse Probe0 consistently. No extra probability QK does NOT mean
no coarse Grid QK. Four coarse Probe QKs are still included in the full forward.
Retention20%,adaptive-local clustering,Drop/Rep/History/Exact,16-round budget,
state losses and mass-transfer compensation remain fixed. No GT tuning or method
changes after quality freeze. No default method promotion.

Compact history scales as ceil(G/stride)*A half values,plus small Grid arrays.
Compared to the previous four-Probe-per-Group cache,this is1/4 for32 and1/32
for256. The compact cache has an explicit8GiB device guard,no hidden CPU spill.

## Verification and campaign

No-GT numerical oracle checks sampled conditionals,own-Grid fallback,multiple
windows,compact cache shape,probability mass,cluster sums and fresh CUDA Graph
inputs. The first harness incorrectly required bitwise equality of atomic
cluster sums;the correction preserves exact prior equality and allows FP32
atomic rounding(~1e-8),without an estimator change. Both2-frame complete24-layer
model diagnostics pass. Diagnostics report all six geometric fields N/A.

Full campaign:only the two changed methods,all7 7Scenes scenes128frames and
all11 ETH3D scenes14–76frames,pose+recon_unposed. Each method has per-scene raw
six metrics and equal-scene macros. Same checkpoint,exact images,shape and
protocol are audited against the previous complete probability4 campaign.
ACC/Comp/CD in metres,CD=(ACC+Comp)/2 unsquared;AUC/F1 fractions in JSON and
percentages in tables. Threshold/voxel7Scenes.05/.0078125m,ETH3D.25/.0390625m.
SAF3R controls are historical local lazy first-correspondence-head Top4 with
ETH3D profile,not paper DINO Top-K.

Both timings use GPU7,same heads128 input,one warmup+three full forwards and a
SEPARATE profiled forward. Include clustering,routing,history,reordering and
prediction heads. GPU4–7 leases are CPU-only and held until full child exit;
one CUDA program per physical GPU. Dense/SAF3R/other unchanged controls are
reused. End-to-end Dense speedup N/A without a new authorized paired Dense run.
Historical probability4 median15.641640s is a separate batch,not a paired timing.

Run `python -m scripts.run_sampled_history_integration`,then
`python -m scripts.report_sampled_history_integration`. Frozen full source,
hashes,logs,exclusive monitors,raw quality and all tables are under
`outputs/sampled_history_quality_20261104`.
