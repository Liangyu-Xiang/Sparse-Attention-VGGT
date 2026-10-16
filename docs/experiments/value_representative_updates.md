# Cheap GPU representative updates for recursive Value clustering

The implementation follows the latest representative-update suggestions in the
referenced “几何注意力加速策略” conversation. Exact local medoids remain a
diagnostic/control, not the fast runtime default configuration.

## Runtime configuration

Use `configs/evaluation/eval_layer_value_local_vggt_fast.yaml` with the normal
Hydra evaluator. It inherits the original VGGT checkpoint and L3--L9 scope,
selects GPU `representative_update: centroid`, and retains 3.5% patch tokens
(approximately 4.02% attention connections including special keys).
The original configuration and `representative_update: exact` remain unchanged
for reproducibility. All methods still need a dense L2 initialization; this
change does not make cold-start PAM free.

Python API:

```python
install_layer_value_attention(
    model, retention=.035, neighbors=8, iterations=2,
    local_backend="gpu", representative_update="centroid",
)
```

Supported representative policies:

| Policy | Behavior | Distance work |
|---|---|---|
| `exact` | Original exhaustive real-member Euclidean medoid | sum of cluster-size squares times d |
| `previous` | Keep previous representative token indices | zero representative-search distances |
| `centroid` | Scatter cluster mean, choose nearest actual member | O(Nd) |
| `sampled` | Include valid previous rep plus fixed member candidates | O(cNd), plus grouping/sorting |

For sampled medoids set `sampled_candidates: 4` or `8`. Remaining candidates
are deterministic uniformly spaced positions in token-index-ordered cluster
membership; small clusters may repeat candidates. Candidate distances are
evaluated against **all** current members using the original sum-of-L2
objective, not distances to the mean. Previous reps are retained on numerical
ties. Representatives always remain actual current-cluster token indices,
and K/V gather uses the same indices for every head.

Centroid-nearest is exact (within numerical tolerance) for the fixed-cluster
**squared** Euclidean member-representative objective, but approximate for the
original unsquared Euclidean objective. Evaluation retains the original mean
L2 distortion; the objective has not silently been switched to squared L2.
Unchanged synchronous assignment cannot guarantee unsquared-distortion
monotonicity when followed by a centroid-nearest update.

## What is deliberately unchanged

- One partition over mean raw V, not per-head matching.
- Global dense L2 seed; sparse natural global layers 3--9, all other attention unchanged.
- Previous-layer Euclidean cluster-neighbor graph, q=8.
- Two synchronous assignment/update iterations.
- A token's candidate labels are fixed by its original previous-layer label for both iterations.
- Previous-layer representative tokens are fixed membership anchors, preserving nonempty clusters and K.
- FP32 head mean / FP64 local Euclidean arithmetic, for an isolated updater comparison.

The reassignment **algorithm and constraints** are identical. Actual labels
and subsequent online features can diverge once the representative changes.
The hierarchy/sibling-neighborhood and attention-dependent update-budget
proposals require a larger coarse/fine attention redesign and are not included
in this representative-only experiment.

## Four-group experiment (sampled tested at both 4 and 8)

```bash
PYTHONPATH=. python scripts/benchmark_value_representatives.py \
  --gpus 0 1 2 3 4 5 6
```

All 11 ETH3D scenes use the same input order/resolution/checkpoint and cached
converged 3.5% L2 seed. Each scene runs `exact`, `previous`, `centroid`,
`sampled4`, `sampled8` in the same process/GPU, with two warmups and five
measured full-model forwards. Record whole inference, L3--L9 attention-module
time including selection/QKV/projection, local selection time and pose AUC.
Initialization, metrics, saving and oracle diagnostics are outside timing.
CUDA-event intervals may include host dispatch gaps.

After each measured method, a separate untimed diagnostic forward captures
its actual current-layer V, final labels and representatives. On that **same
partition**, exhaustively compute exact member medoids and report

    100 * (D_method - D_exact_same_partition) / D_exact_same_partition.

This isolates representative-choice error. It does not measure the additional
effect of a different partition or online feature trajectory, and it is not
a global k-medoids optimality gap. Final pose AUC measures the complete
recursive method. The diagnostic checks that its extra forward reproduces the
timed pose output. No expensive exact update is hidden inside fast timings.

Results: `outputs/layer_value_local/eth3d_representative_updates_35pct/`.
The initial pipes smoke test is preserved under
`outputs/layer_value_local/eth3d_representative_updates_35pct_smoke/`.

## Completed results

All 11 scenes completed. Exact/previous/centroid/sampled4/sampled8 L3--L9
module times are 168.325/97.444/102.878/108.974/110.457 ms in the paired batch.
Centroid reduces local selection from 101.036 to 35.480 ms, approximately 2.85x,
and selected-layer time by approximately 1.64x. Its same-partition unsquared
Euclidean distortion gap is 0.035469%; final AUC30/5 are 82.428440/43.273967%,
versus Exact's 82.171313/43.050708%. Overall inference excluding initialization
is 2.235338 vs 2.303775 seconds. This does not imply a similarly large full-model
speedup or universal accuracy improvement. Exact reproduces the previous pose
baseline for every scene within 1e-12. 64 related tests passed, 1 optional test
skipped. Detailed methodology/table:
`outputs/layer_value_local/eth3d_representative_updates_35pct/report.md`.
