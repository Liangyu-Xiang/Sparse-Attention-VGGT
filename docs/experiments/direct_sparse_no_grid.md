# Direct Sparse Attention without Grid routing

Base: published6285125b,followed by the separately archived PyTorch Flash compatibility prototype c965ff76. Working branch:codex/v8-direct-sparse-20261104. Main/shared checkout and other optimization worktrees were not edited. Initial remote state and staged commits are recorded in the campaign provenance. Default backend remains serial_tuned.

The [complete campaign report](../../outputs/direct_sparse_quality_20261104/report.md) contains18-scene quality,complete-forward timing,all1584-window same-QKV benchmark,24-layer tables,tile tuning,launch/resource observations and unavailable occupancy reason.

## API and scope

```python
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention

output, evidence = direct_sparse_attention(
    q, k, v, sparse_key_ids, special_key_ids, scale,
    logit_bias=already_computed_bias,  # None selects pure_sparse
    probe_positions=positions,
    return_history_probability=True,
    history_export="lse_probe",      # B; or "inline" for candidate A
)
```

Q is[Head,Query,Dim],K/V[Head,FullKey,Dim],IDs[Group,SelectedKey],one Group per32Queries,shared across Heads. IDs are actual original K/V indices; -1 is padding. Specials are unique supplied original indices. Optional bias is already computed by the unchanged selector; the kernel does not perform mass transfer. Query head windows may have non-contiguous head stride,with contiguous token/channel rows. Probe positions are indices in the current query window and must belong to their query Group. One/four Probes,tail clipping and CUDA Graph capture/replay are tested. Outputs are full-query AV and final Head/query LSE; optional probabilities are normalized per Head first,then averaged across Heads. Special and patch probability arrays are separate; no forced patch-only normalization.

The main AV kernel has no key_grid/regions/offset/count/order input,no Grid output,no sort,no Grid loop,and no Query×Key global mask. It traverses Specials and selected IDs in fixed Key tiles with online softmax and one normal AV QK pass. Both pure_sparse and compensated_sparse are tested; compensation remains enabled for complete-model evaluation. It is a Triton Flash-style kernel,not a call to PyTorch's native Flash backend. The latter remains available through reuse_v8_torch_flash.json,but measured43.434s because of packing overhead and is not the chosen Direct path.

## History A/B

A reuses main QK and stores tile-normalized FP16 probe probabilities plus per-tile LSE. Only AFTER the final full-support LSE is available are tile values rescaled and reduced across Heads. It never reports running denominators as final probability. It avoids FP32 raw logits but still uses a large per-Head normalized FP16 temporary (~211MiB/full window). B stores final LSE and recomputes only a few Probe QKs,then produces compact normalized Head-mean probabilities without a per-Head edge bank. Same-input tests show B faster and more accurate;all-window A/B costs and temporary sizes are recorded.

The user chose ORIGINAL v8 D2+M5c for the full model. Actual sparse probability A/B outputs are therefore NOT fed into the failed beta estimator. An independent module reconstructs original Grid-conditionals,EPS compatibility and FP16-u14 cache. Original legacy full-query Grid summaries still need an auxiliary QK pass outside AV;its cost is explicitly reported. Keeping those definitions means this is not a claim to remove all Grid computation from the whole model.

The independent coarse observer pools Keys once/layer and shares ONE FP32 Probe×pooled-Key QK between full-Key(with Specials) M5c and patch-only residual normalizations. Original residual arithmetic used BF16-rounded pooled Keys/matmul;sharing FP32 is a precision change. It preserves the formulas but cannot promise identical complete trajectories. A matched-input diagnostic quantifies candidate sensitivity;geometric causal contributions are not isolated by rerunning/tuning quality.

Legacy sums use a conservative Cauchy bound <=32 to keep exponent differences in[-64,0]. A per-Grid stable max/probe fallback handles higher-norm inputs;no probability floor or quality threshold was tuned. Four-probe cache production shares K loads/Tensor Cores. Ordinary and high-norm cases and all24 first-window original AV/summary/denominator/cache oracles pass.

## Results and limitations

All1584 real windows of heads128 were replayed with original v8 Q/K/V,IDs and bias:windowed Dense Flash10.222401s,old Grid Sparse7.456610s,Direct compensated2.864181s,pure2.806066s,Direct+A7.863682s,Direct+B3.752775s. These SUM three-sample window medians;they are not full-model forward times. Dense uses the same query windows and full Keys,so it is not a substitute for native full-sequence Dense timing. Tile128/4warps/2stages wins the tested fixed-input sample aggregate;no GT was involved.

Final complete forward:18.999162s median,one warmup+three complete forwards,GPU7,peak allocated18.739395GiB. Includes all clustering,history,observation,routing,reordering,initialization inside forward and prediction heads. Dense full-forward speedup N/A:no unchanged Dense rerun in this campaign. Historical Dense/v8 timing belongs to separate batches.

All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames have pose+recon_unposed and all six metrics. Compared with historical matched v8,7Scenes F1 decreases0.47058percentage points;ETH3D F1 increases1.39953points but ACC distance worsens3.9881%. Thus kernel numerical correctness does not establish quality equivalence. Opt-in only;default unchanged. The report preserves every per-scene metric and macro rather than selecting better repeats.

Hardware occupancy is unavailable:actual Nsight Compute failed ERR_NVGPUCTRPERM. Compiler resource metadata is recorded separately and is NOT occupancy. Remaining full-model bottlenecks include legacy observation (~3.6s),conditional cache (~1.36s),and prior/state/budget/mask (~4s),not just the now-small AV core.

## Staged commits

- 801b66f0:Grid-free normal AV and final LSE.
- 9b59254b:actual probability A/B and fixed-input kernel tests/benchmark.
- 20656a68:independent legacy observation and compact conditional cache.
- d05e9ac5:shared probe K loads and stable bounded/fallback sums.
- d948d36c:opt-in model integration,shared coarse observation and accurate QK/memory bookkeeping.

Large redundant per-scene source tar files stay local;one complete frozen source archive and per-run hashes are published. Failed capture/preflight traces remain available. Every diagnostic reports all six geometry fields unavailable with concrete reasons. Full quality distances are metres,CD=(ACC+Comp)/2 unsquared,AUC/F1 percentages in human tables and fractions in JSON. Same reconstruction thresholds,alignment/fusion and exact frames are audited;SAF3R controls use local lazy Top4/ETH3D profile,not paper DINO Top-K.
