# GPU local Value medoids, excluding the layer-2 initialization

## Scope and timing protocol

- Same ETH3D 11 scenes / 377 calibrated images as the original online Value experiment.
- Original VGGT checkpoint; dense layers 1--2 and 10--24, sparse global layers 3--9.
- Mean raw V over 16 heads; 10% patch medoids shared across heads, q=8, two synchronous updates.
- FP64 Euclidean distances and exhaustive real-member medoid updates on GPU. Fixed source-label neighborhoods and previous-medoid anchors remain unchanged.
- Same-scene converged layer-2 PAM labels/medoid token IDs are preloaded before timing. Actual current layer-2 mean V is checked against the cached input on first warmup. Only initialization is excluded: layer-2 attention, mean-V calculation and the GPU recursive updates remain timed.
- No in-forward file dumps. Two warmups followed by five synchronized wall-clock forwards for both dense and sparse on the same GPU. Report per-scene medians and ratio of sums of medians, not average per-scene speedups.
- CUDA events measure L3--L9 attention modules (including QKV/projection/local selection), local selection including head-mean V, and representative K/V gathering plus SDPA. Event intervals can include host dispatch gaps; they are not pure instruction-throughput measurements.
- This is conditional, initialization-excluded performance, **not** cold-start end-to-end inference. It makes no claim that the PAM seed is available for free in deployment.

The implementation uses FP64 GEMM for cluster-neighbor Euclidean distances,
Top-K with exact boundary-tie repair instead of sorting every cluster row, and
chunked ragged member-pair evaluation for exact medoids. The sole per-iteration
host scalar determines the number of pair chunks; no feature arrays or cluster
distance computations return to CPU. GPU/CPU numerical operation ordering can
differ; the mathematical metric and medoid objective do not.

## Reproduce

```bash
PYTHONPATH=. python scripts/benchmark_layer_value_gpu.py --gpus 1 2 3 4 5 6 \
  --output-dir outputs/layer_value_local/eth3d_gpu_local_excluding_seed_gemm
```

Requires the existing matched scene seeds under
`outputs/layer_value_local/eth3d_max100_uniform_q8/*/selection/forward_0001/layer_02`.
The earlier `eth3d_gpu_local_excluding_seed` output is a diagnostic version with
direct FP64 CUDA cdist and full-row neighbor sorting, not the optimized result.

## Leading arithmetic comparison

Let N be the patch count across all frames, S the special-token count, n=N+S,
H=16 the number of heads, d=64 the per-head feature dimension, K=round(0.1N),
q=8, T=2. Let s[t,k] be cluster sizes after assignment at local iteration t.
One multiply-add counts as two FLOPs. These are leading arithmetic estimates,
excluding square roots, softmax, sorting/Top-K, gather/scatter and memory traffic.

| Operation | Leading FLOPs per global layer |
|---|---:|
| Dense QK^T and AV, all H heads | 4 H n² d |
| Representative-K/V QK^T and AV | 4 H n (K+S) d |
| Neighbor distances using FP64 GEMM | approximately 2 K² d |
| Two restricted assignments, including explicit old-label comparison | approximately 3 T N (q+2) d |
| Exact within-cluster medoid evaluation | approximately 3 d sum[t,k] s[t,k]² |
| Final distortion distances, a diagnostic retained in measured code | approximately 3 N d |
| Head-mean V reduction | O(H N d) |

The recorded `approx_local_distance_flops` includes the neighbor, assignment,
medoid and final-distortion estimates, but not the head-mean reduction or
non-distance operations. Treat it as an arithmetic comparison, not an exact
hardware-operation count or predicted speedup. Tensor-core BF16 dense attention
and FP64 distances have very different achieved throughput.

For balanced clusters and retention r=K/N, sum[t,k]s[t,k]² approximately T N/r.
Ignoring special tokens and lower-order terms, the distance-work ratio is

    [2 r² + 3 T (q+2)/N + 3 T/(r N) + 3/N] / (4 H).

At r=0.1, T=2, q=8, H=16 this is

    0.0003125 + 1.921875/N,

or roughly 0.03--0.05% of dense attention-core arithmetic over these input sizes.
Actual measured cluster-size squared sums are retained in each scene result,
so the empirical estimate does not rely on the balance assumption.

This is **not globally linear complexity**: the rebuilt exact neighbor graph
still has K² distances, and cluster imbalance can make exact medoid work large.
The sparse attention retains (K+S)/(N+S) of dense attention-core arithmetic,
slightly more than 10% because all special tokens are kept.

Only 7 of 24 global layers are changed. Even if all global layers had equal
cost and all other model costs vanished, the arithmetic reduction of all 24
global attention cores would be about 24/[17+7*0.105], or 1.35x, not 10x.
Unchanged frame attention, QKV/projection, MLPs and prediction heads further
limit end-to-end gains. This equal-cost arithmetic bound is not a guarantee
of a runtime speedup or a strict hardware-runtime upper bound.

## Completed ETH3D results

All 11 scenes complete: warmed mean scene-median inference 2.429843 seconds
dense vs 2.358096 seconds GPU Value excluding seed, a 1.030426x aggregate ratio.
L3--L9 attention modules including local updates: 273.738845 vs 175.477792 ms,
a 1.559963x ratio. Seven local updates average 94.062580 ms.
All pose AUC metrics match the CPU Value method; maximum local-layer distortion
difference is 1.11e-16. Actual cluster-size-based local distance arithmetic is
0.033965--0.047533% of dense attention-core arithmetic (77-layer mean 0.039224%).
Full report and scene table: `outputs/layer_value_local/eth3d_gpu_local_excluding_seed_gemm/report.md`.
