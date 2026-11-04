# Shared Value Loss: Cluster-count and frame-scaling GPU evaluation

**Date:** 2026-11-04. **Server:** `14.152.92.142:5948`. **GPU:** RTX 5090 D 32GB. **Key Retention:** 20% in every variant. Flat Value Cluster fractions: 20% / 10% / 5%. Shared Value Loss, sampled256 historical probability, four state types, no Rep/History population compensation, and identical 24-layer VGGT weights.

## Forward speed on the same 7Scenes heads sequence

| Frames | 20% clusters | 10% clusters | 5% clusters | 10% speedup | 5% speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 12.0658s | 11.2026s | 11.873s | 1.077× | 1.016× |
| 160 | 17.9998s | 15.9795s | 15.6210s | 1.126× | 1.152× |
| 192 | OOM | OOM on next forward | 20.5799s | N/A | N/A |

128 frames: one warmup and three measured forwards per variant (original six-scene campaign).
160 frames: one warmup and two measured forwards each, three independent exclusive physical GPUs, same GPU SKU; medians shown.
192 frames: 20% OOM on full forward, 10% warmup succeeds but measured forward OOM in GELU/MLP, 5% warmup + one measured forward + CUDA Event profiling succeeds. 5% at 192 has **one** warmed runtime sample and is not directly comparable to OOM variants.

The 128-frame three-scene mean forward times were 12.211s / 11.379s / 11.712s for 20/10/5%. Short ETH3D (31–38 frames) means were 2.524s / 2.720s / 3.168s, so fewer clusters were slower on short sequences.

## CUDA Event profiled stages (seconds, all 24 global layers)

### 128 frames

| Stage | 20% | 10% | 5% |
|---|---:|---:|---:|
| Value Clustering | 1.080 | 0.635 | 0.547 |
| State Loss / Budget / Mask | 2.384 | 1.961 | 2.484 |
| Sparse Attention / evidence | 2.885 | 2.959 | 2.946 |
| Other named history/probability | 0.934 | 0.798 | 0.717 |
| Other model and unmarked intervals | 4.145 | 5.136 | 4.950 |
| **CUDA forward interval** | **11.428** | **11.488** | **11.644** |

### 160 frames

| Stage | 20% | 10% | 5% |
|---|---:|---:|---:|
| Value Clustering | 1.551 | 0.915 | 0.720 |
| State Loss / Budget / Mask | 4.163 | 2.922 | 3.484 |
| Sparse Attention / evidence | 4.566 | 4.648 | 4.631 |
| Other named history/probability | 1.248 | 1.126 | 0.984 |
| Other model and unmarked intervals | 6.697 | 6.502 | 6.408 |
| **CUDA forward interval** | **18.225** | **16.113** | **16.227** |

These CUDA Event intervals include scheduling/queue gaps and are **not kernel self times**. They come from additional instrumented forwards, not the benchmark median. Across frame counts, 10%/5% yields modestly higher speedups, but no several-fold improvement.

## Explanation

For 128 frames, `Cluster count × padded cluster width` is nearly constant:

- 20%: 26,009 × 64 = 1,664,576
- 10%: 13,260 × 128 = 1,697,280
- 5%: 6,630 × 256 = 1,697,280

As a result, reducing the number of clusters does **not** proportionally reduce per-Group shared-Loss reduction work. The Value clustering stage speeds up, whereas the state selection stage has a minimum at 10% and gets slower again at 5%; Sparse Attention remains almost constant because the Key Retention budget is unchanged.

At 192 frames the 5% variant's geometry width grows to **512**; it still executes with 20.414 GiB peak Torch allocated memory and a 20.5799s warmed forward. The 20/10% variants OOM in full model MLP, **not** directly in the State Loss kernel.

## Accuracy caution

On three 7Scenes 128-frame sequences, F1 averaged **46.721 / 45.484 / 45.126%** for 20/10/5%.
On three ETH3D short sequences, F1 averaged **54.147 / 50.182 / 35.617%**.
The 160/192-frame tests are **speed and capacity diagnostics only**, not fresh geometric accuracy evaluations. The 5% reduction is currently **not** accuracy-equivalent.

## Artifacts

All raw GPU measurements, strict GPU lease reports, full 24-layer statistics, original OOM tracebacks and source snapshots are persisted **on the remote server** at:

`/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/shared_value_cluster_scaling_20261104/`

Subfolders: `quality/` (six-sequence cluster sweep), `scaling/` (128/160/192 stage profiling), `source/` (changed implementation and configs). A longer Chinese report is saved as `report.md` in the remote folder.

The source changes are on this **isolated experiment branch**, not `main`.
