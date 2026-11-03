# Experimental: direct sparse History probability transport

This opt-in branch is a **controlled ablation**, not a replacement of the
latest GPU-server worktree or of the D2 dense-proxy probability approach.

## Formula

For each query group and Canonical Grid r with n_r real patch keys:
- H_r: distinct patch edges actually executed in the previous layer.
- p_sp(e): last layer measured sparse-softmax single-key probability, with
  population-corrected representatives divided by their representation count.
- P^0_prev(r), P^0_curr(r): the *coarse* Grid masses on a consistent scale.
- R_r=P^0_curr(r)/P^0_prev(r).
- M5c beta=R_r^(eta_down) when R_r<1, else R_r^(-eta_up).
  When P^0_prev is numerically zero, beta=0.

Define:
```
score_history(e) = beta_r * p_sp(e) + (1-beta_r)/n_r
score_unseen(j) = 1/n_r
Z_r = sum_{e in H_r} score_history(e) + (n_r-|H_r|)/n_r
p_hat(e) = P^0_curr(r) * score_history(e)/Z_r
p_hat(j) = P^0_curr(r) * (1/n_r)/Z_r
```

All observed + unobserved patch probabilities in r sum to P^0_curr(r).
If coarse mass is patch-conditional, results are **patch-conditional**, not a
global all-key distribution. Special tokens need separate consistent mass.

**The normalized output is a dense-proxy distribution, not measured Dense
Attention.** Sparse-support normalization can inflate p_sp(e), and this
experiment must quantify that inflation against a Dense oracle.

## How to enable / control

Use `configs/evaluation/reuse_residual_exploration_sparse_transport.json`.
Default config remains legacy. Three fields are opt-in:
```json
"history_probability_mode": "sparse_transport",
"transport_eta_down": 1.0,
"transport_eta_up": 0.0
```

Other candidate M5c pairs can be swept after a stable baseline. Keep all
other state definitions, budgets, and Triton kernels identical when comparing.

## Suggested checks on a live GPU server

1. `pytest -q tests/test_reuse_history_transport_cpu.py tests/test_reuse_mask_cache_cpu.py`
2. Evaluate legacy and sparse_transport with identical 7Scenes inputs,
   seed, budgets, model weights, and warmed-up single-process GPU timing.
3. Compare history probability against small-scale Dense-softmax oracle:
   MAE, KL, ranking recall, per-grid mass, and History coverage mass.
4. Compare selection state histograms, unique mask edges, budget feasibility,
   AUC@3/30, ACC, Comp, CD, F1, end-to-end seconds and mask builder ms.
5. Profile extra scatter/gather costs (history transport must not erase sparse
   kernel speedup). Do not count attention execution as mask construction.

Warning: the current public `refactor/unified-reuse-exploration` branch may
predate uncommitted remote Codex improvements. DO NOT overwrite those files
with this experimental branch without reviewing `git status` and diffs.
