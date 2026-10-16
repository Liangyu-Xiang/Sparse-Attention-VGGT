# Revised layer-shared A/V attention (2026-10-16)

This is a new implementation, `layer_shared_av`, not a modification of the
head-aware adaptive/block policy. The exact reference conversation is preserved
in `layer_shared_av_reference.md`. Latest Layer-0 clarification takes precedence.

## Algorithm

All 24 **global** attention layers (zero-based 0–23) use fixed 10% final edges.
The model's separate per-frame/local attention remains unchanged. Policy, mask,
fine partition, and history are shared across heads; actual attention remains
multi-head. No Dense-output probes, adaptive total budgets, Dense fallback,
head matching, separate Key blocks, or Query-block shared masks are used.

1. Layer 0: partition each frame's 2D patch grid into approximately equal spatial
   fine cells, about 15% as many cells as patch tokens. Choose fixed spatial
   center tokens. Uniform spatial coarse cells are unions of fine cells. **No
   Value-driven representative update or reassignment at this layer.** For
   28×37 patch grids, the fine grid is 11×14 and the coarse grid is 2×4.
2. Layers 1–23 inherit only fine labels and representative indices. Using
   current head-mean V, recompute centroid-nearest real-token representatives;
   build temporary coarse groups from these current representatives; perform
   one synchronous within-group reassignment with fixed anchors; update
   representatives; rebuild final coarse groups from the resulting current V.
   Coarse grouping uses two anchored Lloyd iterations, about Npatch/128 groups.
   Neither temporary nor final coarse grouping is inherited by the next layer.
   This is lightweight approximate Value clustering, **not exact k-medoids**.
3. Average normalized/RoPE-transformed Q and K over heads **before QK**.
   Pool mean-head K over current Value coarse groups. Compute coarse logits
   `meanQ @ pooledMeanK.T / sqrt(d)`. There is no log-population bias.
4. Uniformly sample `ceil(eta*Nq)` query centers. Let P be their coarse softmax.
   `C = 1 - mean entropy(P)/log(number_of_coarse_groups)`;
   `R = mean cosine(P_i, mean(P))`; `D = 1-R`;
   `alpha_raw = 2*C*D/(C+D+1e-12)`.
5. With dominance threshold tau, snap alpha to 0 at `alpha_raw <= 1-tau`,
   to 1 at `alpha_raw >= tau`, otherwise retain it. Split the fixed patch
   budget into `KA=round(alpha*Kpatch)` and `KV=Kpatch-KA`.
6. Value-only: stop after sampled sensing, select representative Value support,
   and use shared-K/V SDPA. No full-query coarse QK, per-query routing, or
   history validation. History is empty, exactly as KA=0 specifies.
7. Mixed/Attention-only: independently compute each query's coarse P (chunks
   are memory batching only). Capacity-constrained Hamilton rounding allocates
   KA across groups. Reuse prior validated key indices, remapped into current
   coarse groups; fill missing slots with uncovered fine representatives.
   Complete KV with largest-population fine representatives not already covered.
8. Execute actual multi-head sparse attention on each query's shared support.
   Export logits and per-head LSE from that **same QK pass**, including special
   keys in normalization. Average normalized sparse attention over heads, then
   keep top KA **patch indices** as next-layer history. Value edges may be
   promoted. Do not rank mean logits: head normalizers differ.

### Explicit engineering choices for underspecified boundary cases

- Mandatory special keys are inside the total final-edge budget:
  `Ktotal=floor(.1*Nall)`, `Kpatch=Ktotal-Nspecial`. This preserves the user's
  final-10%-edges requirement. It is **not** 10% patch keys plus free specials.
  With 128×(1036+5) tokens: total 133248, 13324 retained keys/query, including
  640 specials and 12684 patch keys; density 9.99940%. A/V percentages refer
  to the **patch selection budget**, with special keys reported separately.
- History is deduplicated by **token index**, not fine label. Multiple distinct
  validated Attention keys in one current fine cluster remain eligible. For
  each query/group, capacity is `fine_count + history_tokens - history_fines`.
  This is the available union of all validated tokens and representatives of
  still-uncovered fine cells. Saturated quota mass is redistributed. Reuse
  preserves prior validation rank; exploration/Value ties use population then
  fine index. This guarantees unique final keys and exact KA+KV without
  arbitrary unvalidated original-token filler. The initial v1 experiment
  over-constrained history to one token per fine cell; it is retained only as
  an ablation, NOT the final implementation.
- For pure Value layer 0, choose uniformly spread fine-cell representatives
  rather than taking only the first frames when equal-population cells tie.
- Special queries use the same final budget; they are not secretly Dense.
- Deterministic cluster mean reductions are used. No `Npatch × Npatch`
  distance matrix is constructed. Small coarse Lloyd distance chunks are allowed.

## Calibration and validation protocol

The literal formula uses **coarse-logit multiplier 1**. It is retained as the
reference arm, not overwritten by calibration. The optional multiplier is a
single global scalar applied to coarse logits before softmax, never to actual
attention logits. It does not contain layer indices, pose labels or a fixed
layer schedule. This extension is explicitly recorded as `logit_scale`.

Only Chess is used for selecting the A/V profile. Six other scenes are held out
from selection. Offline Dense-trajectory sensing is a **diagnostic**, not an
online probe and not the fixed-10% algorithm's accuracy/timing result.

The global multiplier sweep is {1,2,4,8,16,32,64,128,256}; dominance is
{continuous, .8, .9, .95}; sampled fractions are {1%, .5%, .25%, .1%}.
Offline fraction comparisons use nested subsamples of saved 1% query logits;
they approximate, but are not identical to, independently spaced query sets.
Actual sparse validation uses the exact requested fraction sampling rule.

Profile objective: equal-weight mean absolute deviation from target Attention
ratios [0,1,0] over layers [3–9,10–16,17–23]. Layers 0–2 have no target ratio.
The sweep minimum at 0.1% is not automatically chosen: its Layer-14 mode
switches near tau, while .25% retains the same pure/mixed count as 1%.
The profile-only screened candidate was multiplier 32, tau .8, eta .25%; it
was not assumed to guarantee accuracy. Real sparse validation rejected it:
seven-scene AUC30 46.403%, versus reference 33.322% and archived Dense84.293%.
Fine ratio .15 and coarse target128 remained fixed for these two arms.

Eleven subsequent **Chess-only** single-parameter sensitivity checks tested
multiplier16/64; tau.9/.95/continuous; eta1%/.5%/.1%; fine.20; and coarse64/256.
These use one cold forward and do not provide formal warm timing. Changing
tau to .9 alone raised Chess AUC30 to85.829%, while .8 gave3.624% and .95
gave13.049%. The very large sensitivity is a limitation, not something to hide
behind favorable regional budget averages.

The first candidate therefore used multiplier32, tau.9, eta.25%, fine.15,
coarse128. This **second selection does use Chess pose accuracy**, unlike the
initial offline signal-only screen; six other scenes do not choose the final
threshold. A Dense-minus-3pp Chess gate is a post-screen diagnostic criterion,
not a preregistered hypothesis. Among the tested single-parameter alternatives,
only tau.9 passes that gate. Full results from both rejected arms remain saved.

After fixing history capacity and coarse-QK numerical invariance, that v1
threshold recommendation did not transfer: the first correct v3 seven-scene
run at tau.9 obtained AUC30 52.098%, and Chess13.614%. Therefore a **correct-v3**
3×3 joint Chess sweep was performed with multiplier{16,32,64} and tau{.8,.9,.95}.
It selects multiplier32 and tau.95: Chess AUC30 83.972%, with regional Attention
shares11.16%/52.15%/1.39%. Multiplier16/tau.95 had slightly higher Chess84.425%
but only31.67% Attention in the middle, failing the requested middle dominance.
The exploratory rule chooses the highest Chess AUC among candidates with
Value-majority outside and Attention-majority in the middle. This is parameter
selection, **not** a runtime fixed-layer schedule. No other scene chooses these
parameters. All nine outcomes, including failures, are in
`outputs/layer_shared_av/joint_v3_chess_n128/report.md`.

The calibrated config now uses **multiplier32, tau.95, eta.25%, fine.15,
coarse128, memory query chunk1024**. Bare Python API defaults retain the literal
reference multiplier1 rather than silently replacing it with calibration.

The initial chunk1024 v1 trial **failed** numerical equivalence: Chess AUC30
dropped from85.829% at chunk128 to5.443%, despite repeatability within each
configuration. GPU checks established that selection on identical P and the
indexed attention kernel were themselves chunk-invariant. Shape-dependent
coarse FP32 GEMM rounding was therefore isolated and the final v3 separates
canonical coarse-QK arithmetic tiles128 from memory/routing chunks1024. These
tiles do not share query masks. Memory chunk size must be a multiple of128.
Real128-frame trunk fingerprints (outputs, fine states, history) are checked
at every layer0–6 across memory chunks128/1024; results are saved separately.
This real-input check passed with byte-identical outputs and states at all
seven checked layers; `outputs/layer_shared_av/chunk_invariance_v3.json` records
the source hash and fingerprints. This audit used tau.9; it is not a claim of an independently
checked full24-layer two-chunk pose comparison.
The token-history-only v2 run was stopped rather than mixing its partial
results into final evaluation. Final v3 has a fresh output directory.

Formal reference and candidate runs: 7Scenes, uniform128 frames/scene,
checkpoint `model_tracker_fixed_e20.pt`, BF16, TF32 disabled, one warmup and
three measured forwards. All initialization, hierarchy updates, sensing,
routing, actual attention, and validation are included. Per-layer stage events
are diagnostic-inclusive, and their small instrumentation overhead is included.
Stage profiles are from the final measured forward, whereas reported whole
and global latency are medians of three forwards. They need not sum to the
reported median latency. Stage `hierarchy` also includes Q/K norm and RoPE;
`routing_and_sparse_attention` includes mask building, actual sparse attention,
and normalized-history validation rather than isolating each of these costs.
Pose fingerprints and source snapshots are saved. Accuracy metrics are
macro-averaged over scenes. No old cold timing is treated as a paired baseline.

## Reproduction

```bash
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/evaluate_layer_shared_av.py --scene chess --frames 128 \
  --warmup 1 --repeats 3 --logit-scale 32 --dominance .95 --sampling .0025 \
  --query-chunk 1024 --output-dir outputs/layer_shared_av/selected_joint_v3_n128
```

Use multiplier1, tau.9, eta.01 for the reference. Existing completed result
directories cannot be overwritten; choose a fresh path for new experiments.

`scripts/run_layer_shared_av_validation.py` queues the final seven scenes on
idle GPUs only. `report_layer_shared_av.py` compares reference, rejected
profile-only tau.8 candidate, and final tau.9 candidate. The separate
`report_layer_shared_av_ablation.py` preserves all eleven sensitivity checks.
`report_layer_shared_av_joint_sweep.py` preserves the additional correct-policy
joint sweep; the consolidated report separates old v1 ablations, the first
correct v3 tau.9 arm, and the selected v3 tau.95 arm.

Fresh Chess Dense/SAF3R timing is also performed with identical128-frame input,
one warmup and three measured forwards. Lazy DINO Top-K must be reset and
**recomputed inside every timed SAF3R forward**. The earlier cached-TopK timing
is explicitly marked invalid for full SAF3R cost; archived accuracy comparisons
remain accuracy-only. SAF3R uses its original profile/topk128-per-frame, not a
uniform10% budget. Do not interpret this as an operator-level equal-density test.

The full regression suite passed **218 tests, 1 skipped**; this includes CUDA
attention/LSE/normalized-weight checks, exact token budgets and capacities,
token-vs-fine history semantics, cold-start invariance to V, state reset, and
canonical coarse-QK chunk invariance. Final source is frozen during each suite
of real scene evaluations; every scene has source snapshots and SHA256 hashes.
