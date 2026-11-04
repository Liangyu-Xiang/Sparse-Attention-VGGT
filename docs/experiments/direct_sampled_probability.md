# Sampled selected-Key probabilities from Direct Sparse Triton

Independent branch `codex/direct-inline-probability-20261104`,base `d10fd3e6`.
Scope: kernel extension and fixed-input numerical/performance diagnostics.
Current model history,retention,selector and default execution are unchanged.
The user authorized choosing an efficient output direction with fixed-interval
Query sampling;this is not a request to silently reduce the history samples.

## API

```python
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled

output,evidence=direct_sparse_attention_sampled(
    q,k,v,sparse_key_ids,special_key_ids,scale,
    logit_bias=bias,
    probe_count=1,
    probe_group_stride=8,  # Query interval256;16 means512
)
probability=evidence['patch_probability']       # [ceil(G/8),1,A]
query_indices=evidence['probe_positions']       # [ceil(G/8),1]
selected_ids=sparse_key_ids[::8]                # matching compact supports
special_probability=evidence['special_probability']
```

Q=[H,R,D],K/V=[H,N,D],IDs=[G,A],G=ceil(R/32),-1 padding. Supports are
shared across Heads. Output AV covers ALL R Queries. The sampled probabilities
are each sampled Query's actual compensated sparse Softmax probabilities on its
selected Keys,averaged AFTER each Head's own final normalization. Specials stay
in that denominator and have a separate output. These are not Dense probabilities,
not a Query average,and not Softmax of Head-mean logits.

For one Probe,default local-window Query indices are4,4+32*stride,...,with the
usual clamp for very short tails. Current2048-query windows align with strides8
and16,so the complete heads128 trajectory has fixed global intervals256/512.
Arbitrary window boundaries reset the local sampling phase. The four-Probe form
also works,but dense four-Probe output is not a near-free performance claim.

## Implementation

A single AV launch uses a uniform branch per CTA,OUTSIDE its Key loop. Sampled
Groups call the AV specialization with masked probability writes;ordinary Groups
call the usual AV specialization. Every Query Group computes main QK exactly once.

For sampled rows,store the ALREADY computed online-Softmax exponent in FP16,plus
its FP32 running maximum for that Tile. A second,small kernel waits for the FINAL
per-Head LSE,computes `stored_exponent*exp(tile_max-final_LSE)`,then averages Heads.
The stored intermediate alone is not a final probability;running denominators
are never treated as final. Both the main AV and final probability kernel are
included in reported timings. No probability QK recomputation,Grid routing,sort,
full QxK mask or FP32 raw-logit edge bank is added.

For a64-Group,16-Head,26009-patch+640-special-Key window,one Probe and stride8
use6929152bytes of probability/shift scratch(~6.61MiB),versus221732864bytes
(~211.46MiB)for the original dense four-Probe inline export. The final output is
FP32,but its FP16 auxiliary storage introduces a small measured probability error.
AV remains unchanged in matched tests. There is no claim of exact probability
bit patterns or geometric quality equivalence after changing history sampling.

`debiased_log=True` additionally exports log head-mean scores with assignment
bias removed,using centered FP16 log intermediates. Those are scores,not normalized
actual probabilities,and have their own quantization errors. Main measured
near-free results use actual probabilities (`debiased_log=False`).

## Experiments and limitations

Experiments preserve failed and slower alternatives:original Tile-normalized
inline,exponent reuse with Gather,masked stores,two separate AV launches,and the
final single-launch branch. Two AV launches suffer a fixed sparse-launch cost;
the final branch shares a launch with the ordinary Groups. Compiler metadata
records registers,spills and shared memory;it is not measured hardware occupancy.

Numerical tests cover FP16/BF16,pure/compensated attention,bias60,zero assignment,
empty patch support WITH Specials,tail/duplicate Probes,one/four Probes,strides
1/2/4/8/16,non-contiguous Head windows and fresh-input CUDA Graph replay.
A preflight tl.store pointer-broadcast compilation failure and a benchmark
instrumentation/import-order failure are retained with six quality scores N/A.

The final real-data benchmark uses exact Q/K/V,IDs,bias from the frozen current
Direct-probability trajectory,heads128,ALL1584 windows across ALL24 global layers,
all16Heads,three CUDA-event samples per variant/window. Alternate outputs never
feed the model. Ordinary Dense and SAF3R were not rerun. GPU7 benchmark and GPU6
numerical validation each use a CPU-only exclusive lease;one CUDA program per GPU.

All timing tables are kernel totals across24 layers,not per-layer or complete
model forward times. Native full-forward Dense speedup is N/A;no paired full-model
Dense timing was run. All six geometry fields are N/A with a concrete kernel-only
reason. Existing18-scene geometry scores cannot be reused as quality validation
for feeding these sparser Query samples into history.

See `outputs/direct_inline_probability_summary_20261104/report.md` for final
measurements,per-layer/per-window tables,source hashes and monitor verification.

## Completed real-data results

All1584 windows/all24layers:AV-only2.864744289s;separate one-Probe QK
probability export3.710718651s(+29.5305%);single-launch inline every32Queries
3.689132036s(+28.7770%);every256Queries2.947034430s(+2.8725%,extra82.290ms);
every512Queries2.929577986s(+2.2632%,extra64.834ms). These are kernel totals,
including probability finalization,not complete model forwards.

Every actual window has AV relative L2=0 for all variants. Maximum absolute
probability error against the FP32 probe reference is7.927418e-6 at interval256
and6.616116e-6 at interval512;normalization errors are at most2.092124e-5.
The160 numerical cases and6 fresh-input CUDA Graph cases pass. New geometry
metrics and full-model Dense speedup remain N/A.
