# Selected method: Query256 + sequential fixed-source32 unanchored clustering

The user's selected configuration is now the default
`configs/evaluation/reuse_shared_value_loss_q256.json` in the isolated worktree
`/data_NVME1/mmc_lyxiang/3D/SAF3R-source-local-unanchored`
(branch `codex/source-local-unanchored-20261105`).

The immutable named configuration is
`configs/evaluation/reuse_shared_value_loss_q256_source_sequential_w32_t1.json`.
The previous adaptive-local configuration is preserved as
`configs/evaluation/reuse_shared_value_loss_q256_adaptive_reference.json`.
Historical adaptive diagnostic launchers now explicitly use that reference file.
The original Query-group-scaling worktree has not been changed by this adoption.

The selected method uses Query groups of 256, at most 20% retained Keys,
26009 fine clusters on these 128-frame inputs, fixed groups of up to 32 consecutive
fine-cluster IDs, two centroid updates, free migration of former representatives,
source-local temporary empty-cluster repair and nearest-real-member representatives.
Source groups may cross an adjacent frame boundary. History interval remains 256.
Loss, four states, budgets, attention/evidence kernels and Graph policy are unchanged.

Fresh full-forward profiling of heads/chess/stairs (128 frames each) is archived in
`outputs/selected_source32_audit_20261105/`. The normal full-forward three-scene
mean is 11.045888 s. Its separate instrumented closed ledger totals 11.202133 s.
Those scopes must not be mixed. Current Value clustering is 0.322818 s in the
selected profiled passes. Every included strict GPU monitor reports no foreign
GPU overlap; exact frame lists and resolved checkpoints were checked against
historical controls and prior quality runs.

Unchanged SAF3R and Dense were not rerun per AGENTS.md. The report labels old
SAF3R/adaptive figures as historical, not contemporaneous speed measurements.
Current Dense speedup is N/A without paired timings. This batch is an explicit
timing diagnostic: all six quality fields are N/A with reasons. Prior complete
same-config pose/recon_unposed quality is reproduced separately and labeled with
its batch, not treated as newly measured quality.

The full additive ledger and tree are in `report.md`, `leaf_ledger_mean.md/csv`
and `summary.json`. Normal timing covers warmup plus three complete forwards;
each detailed scene ledger uses one whole-wall median pass among three, with
explicit remainders and integer-microsecond closure. The three-scene equal mean
is not a seven-scene dataset result.

## Attention tile update

The selected default now uses tile64/four warps/one stage. Query group remains256 and clustering remains sequential source32. The timings above describe the earlier tile32 batch, not the new tile64 default. See `docs/experiments/query256_source32_bq64_20261105.md` and `outputs/source32_query_tiles_20261105/report.md` for the matched tile comparison, all six quality scores and the numerical/quality limitations.
