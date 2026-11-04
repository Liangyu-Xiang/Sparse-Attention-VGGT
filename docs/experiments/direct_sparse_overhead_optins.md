# Direct Sparse: opt-in overhead experiments (2026-11-04)

**Base:** codex/v8-direct-sparse-20261104. **Branch:** perf/direct-sparse-overhead-optin-20261104. Default config unchanged.

**No new GPU speed or complete geometry quality result is claimed.** The new options need GPU/CUDA and dataset validation.

## Measured starting point (historical)

The 128-frame, 24-layer Direct profile at outputs/direct_sparse_quality_20261104/report.md measured:
- Main Direct AV: 2.764 s (profile top-level).
- Legacy 32-query selected-support Grid observation: 3.611 s.
- Legacy Grid-conditional cache production: 1.360 s.
- Prior / state loss / budget / packing / compensation / diagnostics: 4.007 s.
- Separate uninstrumented whole forward: 18.999 s median.

The attention core is no longer the dominant bottleneck.

## A. Fast diagnostics (intended numerically equivalent)

Set **minimal_diagnostics: true**. The CUDA Graph and eager selector skip only research-oriented state counts, gain/overlap statistics and their copies. They retain the usual loss, eligibility, 16-round budget solver, support, mass compensation, plus the existing essential validation flag (pack error OR mass error). This removes diagnostic work from the hot path; no history or attention mathematics is changed.

The per-layer records schema is smaller; leave the option off when reporting full per-state diagnostics. Config: configs/evaluation/reuse_v8_direct_sparse_less_diagnostics.json.

## B. Four-Probe selected-support Grid summary (NOT equivalent)

Set **direct_probe_grid_summary: true**.

Currently the independent legacy observer computes selected-support Grid mass by re-evaluating selected QK for all 32 queries per Group. The four Probe QK values are already needed for the original D2+M5c per-Grid conditional denominator.

The new opt-in computes Grid summary from those same four probes, weighted according to the existing tail-duplicate probe weights. It uses a Tensor-Core-compatible 16-row Query tile with only four active rows. The main Grid-free AV remains unchanged. The existing four-probe conditional denominator, cache formula, Special handling and beta calibration remain unchanged (subject to FP rounding), but the *previous real Grid mass summary* is now sampled.

Original:
~~~
GridMass_full(g,r) = mean_{h,q in group} sum_{e in selected∩Grid(r)}
  exp(raw_qhe + bias_ge - final_LSE_qh)
~~~

Opt-in:
~~~
GridMass_probe(g,r) = mean_h sum_p probe_weight(g,p) sum_{e in selected∩Grid(r)}
  exp(raw_phe + bias_ge - final_LSE_ph)
~~~

The second, uncompensated Grid mass channel also retains exp(-bias_ge).

**This is a modeling approximation.** It can affect next-layer residual demand, candidates, states, supports and geometry quality. Never describe it as lossless or as the default replacement. Config: configs/evaluation/reuse_v8_direct_sparse_sampled_grid.json.

An independent combined config enables both: configs/evaluation/reuse_v8_direct_sparse_sampled_grid_less_diagnostics.json.

## Validation and measurements

1. Run syntax checks on all modified Python modules.
2. On a CUDA+Triton server, first run:
~~~
python -m pytest -q tests/test_direct_probe_grid_summary.py
~~~
The oracle checks each summary against independent PyTorch selected-softmax probabilities and compares D2+M5c denominators and history counts between full-query and sampled variants.
3. Run identical heads128 checkpoints and inputs with one warmup and three complete forwards on one physical GPU. Do not overlap CUDA processes and do not mix profiler wall time with an uninstrumented median.
~~~
CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore --scene heads --frames 128 \
  --config configs/evaluation/reuse_v8_direct_sparse_less_diagnostics.json \
  --output-dir outputs/low_overhead/diagnostics_off_heads128 \
  --warmup 1 --repeats 3

CUDA_VISIBLE_DEVICES=7 python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore --scene heads --frames 128 \
  --config configs/evaluation/reuse_v8_direct_sparse_sampled_grid.json \
  --output-dir outputs/low_overhead/probe_grid_heads128 \
  --warmup 1 --repeats 3
~~~
4. For minimal_diagnostics, assert all model outputs, selected IDs, biases and history prior match the original. Only record fields may differ.
5. For probe summary, measure Grid mass differences and first selected-support divergence. Run all 7Scenes and ETH3D scenes with AUC@3/AUC@30, ACC/Comp/CD/F1. Reject quality regressions; never tune on evaluation ground truth.
6. Report independent GPU-event timings for observation, cache, state loss, solver, packing, compensation and complete model, along with peak memory and launch count.

## Limits

- A four-row observation padded to 16 rows is not guaranteed to be eight times faster than the 32-row method.
- The original per-Key conditional cache still recomputes Probe QK. Its measured ~1.36 s is **not** eliminated by these changes.
- State loss/solver/packing retains its original algorithm and remains a major target for future equivalent optimization. Prior P0 reuse did not pass all-window consistency and must not be silently reintroduced.
- CPU syntax/AST success is not Triton compilation or GPU speed/quality evidence.
- No substitution of actual *compensated* Sparse Softmax probabilities for the original Grid-conditional single-Key Dense estimate; that previous beta_gridmass variant severely regressed quality.
