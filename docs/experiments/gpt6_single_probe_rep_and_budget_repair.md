# GPT-6 isolated fixes: historical Rep eligibility and underfilled budgets

Base: `codex/fastest-single-probe-snapshot-20261103` (`b0d4c46767555d837c5e790614d7599c037f9a20`).

## Verified code findings

1. The lightweight state kernel explicitly made the Rep state infinite-loss whenever its representative appeared in the previous support (`rep_seen`). This is not implied by the Rep-population contribution model: a historical representative can still represent its entire cluster. The unnecessary exclusion has been removed and the numerical-state oracle updated.
2. Recorded 7Scenes/heads layer 12 selected only ~4,022 patch keys against a 26,009-key cap, whereas v6 used ~26,002. This is a major underfill of an upper-bound budget, **not proof of one particular root cause**. The current 16-step Lagrange solution has no budget-filling repair.
3. The preceding analysis hypothesized a 1-probe/4-probe mismatch. This was **already corrected in the source snapshot**: `reuse_probe_cache.prepare` chooses one center probe with unit weight when `probe_count=1`. Accordingly, this branch does **not** alter probe positions or weights. It adds a runtime guard to prevent future accidental mixed-probe shapes.

## Changes and experimental controls

- `saf3r/triton/reuse_light_states.py`: Rep is eligible whether or not its key was present in historical support. The Rep loss and population-based execution weight remain unchanged.
- `saf3r/patch/reuse_global_history.py`: refuse incompatible evidence, observed Grid, special, LSE, and weight probe dimensions before entering the producer kernel.
- `saf3r/patch/reuse_budget_repair.py`: optional greedy feasible improvement of an underfilled discrete budget. Each accepted change strictly decreases the *existing* selected-state loss, consumes extra capacity, and keeps the total within the key budget. It is not an exact knapsack solution and does not guarantee the cap will be saturated.
- `saf3r/patch/reuse_lightweight.py`: default `budget_repair=false`; opt in with `budget_repair=true` and eager (non-selection-graph) selection.
- Isolated configs: `configs/evaluation/reuse_light_single_probe_repaired.json` and `configs/evaluation/reuse_light_full_repaired.json`. These change the budget-repair flag relative to their respective original configs. **The Rep eligibility correction also applies to both**, as it is a source-level correction.
- `scripts/validate_lightweight_reuse.py`: update the numerical-state oracle for the Rep eligibility correction.
- `tests/test_lightweight_budget_repair.py`: CPU tests verifying feasibility, monotonic objective, and no-op when the budget is full.

## Validation status

- CPU: a matching copy of the budget-repair implementation passed 150 randomized tests of budget feasibility and nonincreasing summed losses; static Python compilation passed. The checked-in unit test is included but must be executed against the actual repository file.
- GPU: **not available in this environment**. The Triton state kernel, full-forward speed, probability quality, and reconstruction quality have **not been remeasured**.
- AUC@3: N/A (no model/dataset GPU evaluation).
- AUC@30: N/A (same reason).
- ACC: N/A (same reason).
- Comp: N/A (same reason).
- CD: N/A (same reason).
- F1: N/A (same reason).
- Dense speedup: N/A (no matched Dense full-forward measurement).

## Next verification on the authorized GPU

```bash
python -m unittest discover -s tests -p 'test_lightweight_budget_repair.py'
python scripts/validate_lightweight_reuse.py --output-dir /tmp/light-gpt6-validation
python scripts/evaluate_zdrop_history_states.py --dataset 7scenes --method zdrop_history_states --scene heads --frames 128 --warmup 1 --repeats 3 --config configs/evaluation/reuse_light_single_probe_repaired.json --output-dir /tmp/light-gpt6-heads
```

Also evaluate the 4-probe repaired configuration with the same exact frame list for causal comparison, and inspect `counts_mean`, `state_0_clusters_mean`, and the six geometric metrics before treating the fix as quality-preserving. Use the repository's strict GPU lease wrapper where appropriate. Benchmark original and repaired methods with *matched* GPU inputs; do not label historical timings as paired speedups. Do not silently rerun unchanged Dense or SAF3R controls.

These patches diagnose and mitigate the specific observed faults. The severe regression must be considered **unresolved** until the complete GPU validation and controlled ablations pass.
