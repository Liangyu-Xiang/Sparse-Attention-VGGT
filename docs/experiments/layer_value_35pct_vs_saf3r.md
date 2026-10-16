# ETH3D: approximately matched 4% attention-connection budgets

The Value target is **3.5% patch retention**, not 4% patch retention. With 925
patches and 5 retained special tokens per image, this gives approximately
(0.035*925+5)/930 = 4.0188% retained attention connections. The original SAF3R
L3--L9 profile averages 4.0159% over these scenes and layers. Rounding K causes
small scene-dependent differences. This matches the average budget, not each
SAF3R head/layer or connection topology.

## Matched experiment

```bash
PYTHONPATH=. python scripts/benchmark_layer_value_gpu.py \
  --retention 0.035 --compare-original-value --compare-saf3r \
  --gpus 1 2 3 4 5 6 \
  --output-dir outputs/layer_value_local/eth3d_gpu_value_35pct_vs10pct_saf3r_l3_9
```

- All 11 ETH3D scenes, 377 valid calibrated images, unchanged input order and preprocessing.
- Same original VGGT weights; only natural global layers 3--9 become sparse.
- Each scene measures dense, Value 3.5%, Value 10%, then original-profile SAF3R on the same GPU/process.
- Two warmups and five measured synchronized full forwards per method; summarize scene medians.
- Value is layer-shared mean raw V, q=8, two synchronous assignments/exact real-member medoid updates on GPU, FP64 clustering arithmetic.
- Fresh converged two-start global eager PAM initialization at the requested K is computed from the same dense L2 features **before timing**. It is not a truncation of the original 10% medoid set. Actual current L2 V is checked during first warmup.
- The original converged 10% seed is reused for the Value 10% reference. Both seed costs are excluded; dense L2 attention and every local update remain timed.
- Measure both whole inference and the sum of L3--L9 attention-module intervals including selection/QKV/projection. No in-forward file dumps.
- SAF3R uses the original profile; configured topk=128 is inactive in this layer range. Original attention modules and hooks are restored between methods.
- Initialization files and excluded fitting times are in each scene's `initialization/layer_02/` directory. Per-scene results contain all four methods' accuracy and timing data.

## Why lower retention need not be faster

Changing patch retention from 10% to 3.5% reduces neighbor-graph distance
arithmetic to approximately (0.035/0.1)^2 = 12.25%. Restricted assignment
still compares 9 candidate clusters per token. For balanced clusters, exact
medoid member-pair work grows by 0.1/0.035 = 2.857x. Actual per-iteration sums
of squared cluster sizes are recorded, since imbalance can change this ratio.
Selected-K/V attention arithmetic falls to about 4.02/10.48 = 38.3% of its
previous amount, but these components have very different GPU throughput and
fixed launch/indexing costs. End-to-end behavior and accuracy must be measured.

Output root: `outputs/layer_value_local/eth3d_gpu_value_35pct_vs10pct_saf3r_l3_9/`.

## Completed results

All 11 scenes complete. Same-batch mean scene-median L3--L9 module times:
Value 10% 198.471024 ms, Value 3.5% 198.240622 ms, SAF3R 81.598830 ms.
Overall inference excluding initialization: 2.368399 vs 2.370151 seconds for
Value 10% vs 3.5%, i.e. effectively unchanged. Gather+SDPA falls from 25.101981
to 11.159183 ms, but local updates rise from 115.998432 to 129.364018 ms.
Value 3.5% pose AUC30/15/5/3 are 82.171313/70.412460/43.050708/28.431150 percent,
slightly above Value 10% in this scene-macro average, not uniformly by scene.
Detailed report: `outputs/layer_value_local/eth3d_gpu_value_35pct_vs10pct_saf3r_l3_9/report.md`.
