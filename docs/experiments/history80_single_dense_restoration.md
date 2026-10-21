# Original population History80:single Dense-layer restoration

User request:discard the middle-A20/V80 experimental change,return to original
population-compensated History80,and separately replace EACH global layer with
Dense while retaining the sparse scheme everywhere else. Three sequences only.

## Frozen method and scene selection

Original config:configs/evaluation/parent_region_history80_population.json.
L0 pureValue;L1–9/L17–23 A20/V80;L10–16 A80/V20;20% total edge target,
history80%,whole Parent selection,original expanded Tensor Core operator,
population compensation ON,no shift,no adaptive gate,no Oracle/Dense seeds.
Frame/local attention unchanged. This is NOT the latest middle-tuned candidate.

Use the exact prior uniform128-frame lists,checkpoint and preprocessing for:
- heads:largest7Scenes AUC@3 degradation of original population configuration.
- stairs:largest7Scenes AUC@30 AND F1 degradation of that configuration.
- chess:non-degraded normal control.

This is a selected diagnostic subset,not a7Scenes dataset macro. Scene selection
uses known regressions as explicitly requested;this is not parameter fitting
against7Scenes truth. No parameter is tuned in this experiment.

## Intervention semantics

75 fresh runs:3 current all-sparse baselines +3 sequences x24 one-layer Dense
restorations. The original Dense quality rows are reused same-input references,
not new speed measurements. A fresh all-sparse baseline must reproduce all six
prior metrics within1e-5 absolute and match exact input/protocol before any
intervention jobs for that sequence are allowed to run.

At the chosen layer,execute the UNCHANGED sparse operator first. It updates
Value labels/representatives,hierarchy,selection and compensated sparse history.
Discard only its projected attention output,and execute the original Dense
attention on the same current input to supply the residual branch output.
Dense weights NEVER generate or initialize sparse history. All subsequent layers
run the original sparse algorithm on their actual changed upstream features,
with masks naturally regenerated,not frozen. All24 sparse state records remain
present even though one layer returns Dense output. L0 andL23 are included.

This is a controlled diagnostic with deliberate extra shadow computation,not a
deployment-speed optimization. Same-input sparse/Dense output metrics at the
intervention are captured after output projection,before residual addition.
One-at-a-time task effects show marginal sensitivity conditional on the rest of
the sparse network;they do NOT prove a uniquely faulty layer or additive effects.

## Quality,resources and provenance

All runs evaluate pose and recon_unposed:AUC@3,AUC@30,ACC,Comp,CD,F1.
Distances metres,CD=(ACC+Comp)/2 unsquared;7Scenes F1threshold0.05m,
downsample0.0078125m. Same checkpoint,inputs,alignment/fusion and fixed seed0.
Raw metrics plus six-field Markdown tables saved per run. OOM/interrupted/pending
entries have all six unavailable fields with explicit reasons,never invented.

Report task deltas against the freshly reproduced all-sparse baseline for each
sequence,with separate AUC3/AUC30/F1 rankings and all six absolute fields. No
combining three selected scenes into a claimed complete7Scenes dataset macro.

Use all available GPUs without interrupting other jobs. At most one GPU-forward
per device;allow a second process only when the first has freed all model GPU
memory and is doing CPU-only quality. Cap two active processes/device to limit
CPU/RAM pressure. GPU process ownership is checked before launching work.
Any OOM retry has a fresh attempt directory;failed attempts remain recoverable.

Sources frozen before queue launch;each worker snapshots actual sources/config
and records exact frames,geometry and runtime state. Cold forward/peak VRAM
include shadow Dense restoration and its output comparison,recorded separately
from quality. No paired warmed speed benchmark;Dense1.00x reference and all
sparse/ablation speedups N/A. No historical/noisy timing division.

Artifacts:outputs/global_attention/history80_single_dense_20261021.
Queue/report: scripts/run_single_dense_history80.py.
Worker:scripts/evaluate_single_dense_history80.py.
Wrapper:saf3r/patch/single_dense_av_ablation.py.
Root report.md/results.csv/summary.json/queue_status.json retain all75 jobs and
their attempt paths;per-scene24-layer restoration plots and metric-specific
rankings summarize completed interventions. Middle layers are scheduled first
for timely diagnostic feedback,but all24 layers must complete.
