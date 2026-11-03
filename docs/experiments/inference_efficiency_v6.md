# Inference efficiency v6: frozen experimental snapshot

This branch preserves the exact v6 candidate identified by the recorded
**18.109623 second** warmed median complete forward on7Scenes heads/128frames,
RTX5090 D. The immediate grid-local implementation measured26.100919seconds
in the same interleaved run:1.44127x relative to that implementation. This is
not a Dense speedup. Dense was not rerun, by user request.

**Experimental status: ETH3D quality regresses.** This is an inspectable source
snapshot, not a validated equivalent optimization, and is separate from the
layered no-Drop / Rep-always method. The active optimization worktree/chat may
continue investigating; this branch does not include subsequent experiments.

The frozen config is `configs/evaluation/reuse_efficiency_v6.json`: uniform20%
retention, original Drop/History/Rep/Exact rules, grid-local history semantics,
head-mean history, split/fused sparse Attention, fused mask/state/coarse
operations, probe-index caching and reusable CUDA window graphs. It retains
16 dual iterations. The requested mathematical equivalence has not been
established by the full dataset results. No thresholds or scores are altered
for publication.

See `outputs/efficiency_v6_snapshot_20261103/report.md` for all six metrics,
per-scene/equal-scene dataset summaries, differences versus matched local Dense
and grid-local references, raw timing, peak memory and limitations. The
publication provenance records the frozen source hashes and original paths.

The speed diagnostic uses one warmup and three measured forwards per variant,
interleaving reference/candidate in one process. Model/image loading and GT
metrics are outside the timer; reset, clustering, routing, state selection,
history, reordering, CUDA graph initialization/replay and all output heads are
inside. The exact input list is preserved in `full_forward/input.json`.
`PYTORCH_ALLOC_CONF=expandable_segments:True` was set for the diagnostic.
`strict_contexts=true` monitoring found no other GPU application; the monitor
runs on CPU without its own CUDA context. The separate quality campaign uses
one warmup and one timed forward per scene, so its runtime table must not be
called a three-repeat median or compared as paired Dense timings.

Quality scope: all7 fixed7Scenes sequences,128uniformframes each; all11 ETH3D
entries,14–76frames; pose plus recon_unposed. AUC/F1 percentages in tables,
fractions in JSON; ACC/Comp/CD metres, unsquared CD=(ACC+Comp)/2.7Scenes uses
0.05m threshold,0.0078125m voxel; ETH3D uses0.25m/0.0390625m. Saved SAF3R
references are ETH3D-profile lazy first-correspondence-head Top-4, not paper
DINO Top-K. No published baselines or H100 timings are mixed into local rows.

## Verify and reproduce

Run the stdlib verifier after checkout (no GPU, datasets or weights needed):

```bash
python scripts/verify_efficiency_v6_snapshot.py
```

For model execution, install the repository environment and configure dataset
roots as in `configs/evaluation/default_datasets.yaml`; place the external VGGT
checkpoint at `checkpoints/vggt/model_tracker_fixed_e20.pt`. Original machine
paths in recorded input lists must be mapped to your dataset root. Weights,
datasets, credentials and active-worktree files are not part of this snapshot.

Use the checked-in original strict CPU-only GPU monitor; the selected GPU must
have no other model program, including memory-only CUDA contexts. Example
single-scene quality evaluation, in the configured project environment:

```bash
python scripts/run_gpu_lease.py --strict-contexts --gpu 0 \
  --monitor /tmp/v6-heads-monitor.json --log /tmp/v6-heads.log -- \
  python scripts/evaluate_zdrop_history_states.py \
    --dataset 7scenes --method zdrop_history_states --scene heads --frames 128 \
    --warmup 1 --repeats 3 \
    --config configs/evaluation/reuse_efficiency_v6.json \
    --output-dir /tmp/v6-heads-new
```

For the original performance diagnostic use `scripts/benchmark_efficiency_full.py`
with `--input outputs/efficiency_v6_snapshot_20261103/full_forward/input.json`,
the v6 config, and a fresh output directory, under the same monitor. This
explicitly benchmarks both v6 and its immediate grid-local reference; it does
not run Dense. Historical controls are reused in the shipped report. Re-running
unchanged controls requires explicit user authorization under AGENTS.md.
