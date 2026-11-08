# Group-mean logit residual transfer

This is the current research reference, explicitly installed through `saf3r.patch.group_mean_logit_forward.install()`. It does not change the repository default and has not passed the quality/speed acceptance gates against the fast version. It improves all six ETH3D macro metrics over the previously published centered050 implementation, but only F1 exceeds the historical fast control.

## Complete inference flow

The 24 global-attention layers use the following procedure. Local attention and the model's prediction heads retain their existing behavior. A new forward resets history; inference requires sequential layers and batch size one.

1. Compute the layer's Q, K, V and RoPE using the existing model. Split the token sequence into Query groups of 256. Exclude special tokens and padding from Query averaging, ordinary Key estimation, and history. Formal attention still includes the existing special-Key and special-Query treatment.
2. Project current Value vectors to the existing shared head-sketch features, update the existing Value clustering, and obtain labels, representatives and cluster populations. Normalize the sketch to obtain z_j and compute ||z_j||² and ||z_j-z_rep||². These are geometry features for the state loss, not observations of representative attention probabilities. The cluster fraction remains 0.1.
3. Average every ordinary Query in each group and every ordinary Key in each aligned coarse block. The target Key block size is 1024, with balanced frame-local blocks; in the evaluated ETH3D preprocessing each frame supplies 925 ordinary Keys, hence one 925-Key block per frame. For each head h, group g and block b compute µ_l(h,g,b) = scale * mean(Q_l) dot mean(K_l,b). No current fine per-Key QK is performed before state selection.
4. Read only the preceding global layer's actually executed ordinary sparse-mask Key IDs S_(l-1)(g) and raw residuals δ_(l-1)(h,g,j). A representative belongs to this set only if formal attention actually computed it in that preceding layer. Unknown members never inherit its response. On layer zero there is no history and every estimated score is its current coarse logit.
5. For a block's observed support T = S_(l-1)(g) intersect block_b, form x_j = 0.5 * (δ_(l-1)(h,g,j) - mean_(k in T) δ_(l-1)(h,g,k)). Let σ² be mean_(j in T) x_j². For known j set estimated_logit_j = µ_l(h,g,b) + x_j - 0.25σ². For unknown j set estimated_logit_j = µ_l(h,g,b). The variance correction is a heuristic computed only on observed support, not a claim to conserve block mass exactly. No new current fine QK is used. The current default does not multiply this residual by an adaptive beta.
6. Softmax these estimated logits over all ordinary Keys separately for each head, then average all 16 heads to obtain p_hat(g,j). Head averaging happens after softmax; no current representative probability is queried. This estimate is used to score states and construct compensation. It is distinct from the final sparse attention probability.
7. For current Value cluster C with representative r, first filter previous actual members using the existing eligibility rule: H = {j in C intersect S_(l-1)(g): p_hat_j > (sum_C p_hat)/|C| or j=r}. Thus all previous actual edges can inform logit transfer, but only these qualified members enter the History state. Use the existing four states and losses:
   - Drop (0): emit no ordinary edge; loss sum_(j in C) p_hat_j ||z_j||²; cost 0.
   - Rep (1): execute r; loss sum_(j in C) p_hat_j ||z_j-z_r||²; cost 1.
   - History (5): available only if H is nonempty; execute H union {r}; loss sum_(j in C minus H) p_hat_j ||z_j-z_r||²; cost |H union {r}|.
   - Exact (10): execute all members; modeled loss 0; cost |C|.
   Losses are representation proxies, not exact Dense output errors. Adding r to the History execution support does not grant preselection knowledge of its current logit.
8. Run the existing 16-iteration budget allocator to minimize the modeled loss. Keep retention at 10% on layers 0–5, 18% on 6–17, and 10% on 18–23. Pack and deduplicate the chosen edges and validate support/budget legality.
9. Apply the existing geometric-mean compensation. Nonrepresentative history edges and Exact edges have multiplier 1. For Rep, count compensation is |C| and predicted compensation is max(1, sum_C p_hat / p_hat_r). For History, count compensation at r is |C|-|H union {r}|+1; predicted compensation is max(1, (sum_C p_hat - sum_H p_hat + 1_(r in H)*p_hat_r) / p_hat_r). The final representative multiplier is sqrt(count_compensation * predicted_compensation). This is an execution multiplier, not an estimated probability.
10. Execute the formal sparse attention for every Query row on selected Keys plus the existing special Keys: its logits are the current true QK logits plus log(multiplier). Sparse softmax and AV produce the actual output; special-Query correction and the output projection retain the existing implementation.
11. While that same formal kernel computes QK, it sums the already computed raw logits over all valid ordinary Query rows in each group. For each actually executed ordinary Key export δ_l(h,g,j) = mean_ordinary_queries(raw_logit_l(h,q,j)) - µ_l(h,g,b(j)). Replace all previous cache values with these actual current values and IDs for the next layer. Estimated scores, compensated logits, and unknown Keys never enter the cache. Layer 23 skips the unused raw-logit export. There is no second selected-edge QK pass.

The raw residual cache has one Query summary per head/group/key rather than four probe summaries. This reduces that component by 4x, not total peak GPU memory. The legacy `history_cache_bytes` record omits some auxiliary buffers; use measured full-forward peak memory for total-memory comparisons.

## Measured quality and speed

Equal-scene ETH3D 11-scene macros; AUC/F1 are percentages and reconstruction distances are unsquared metres. CD=(ACC+Comp)/2. Historical fast and old-logit controls are reused unchanged.

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical fast | 25.8581 | 80.1708 | 0.3192 | 0.8815 | 0.6004 | 57.3814 |
| Historical centered050 | 20.9442 | 77.3498 | 0.3818 | 0.9567 | 0.6693 | 50.8996 |
| Current group mean | 23.2609 | 77.9969 | 0.3390 | 0.8947 | 0.6168 | 59.2034 |

RTX 5090 D, PyTorch 2.9.1+cu128, CUDA 12.8, physical GPUs 4/5/6, one application per GPU. Each scene has one warmup and three synchronized full-forward measurements. The table reports the median of those three samples. Scope includes initialization/reset, QKV, coarse pooling, clustering, routing, history export/update, reordering, attention, special correction and prediction heads. Dataset loading, CPU geometry evaluation and JIT cold-start time are outside this warmed-forward scope. It is not complete disk-to-reconstruction latency.

| Scene | Frames | Warmed full-forward seconds | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| courtyard | 38 | 1.3988 | 9.420 |
| delivery_area | 40 | 1.4659 | 9.496 |
| electro | 39 | 1.4299 | 9.454 |
| facade | 76 | 3.2201 | 11.388 |
| kicker | 31 | 1.1262 | 9.170 |
| office | 26 | 0.9545 | 9.012 |
| pipes | 14 | 0.5467 | 8.586 |
| playground | 32 | 1.1449 | 9.221 |
| relief | 19 | 0.7461 | 8.767 |
| relief_2 | 20 | 0.7727 | 8.805 |
| terrains | 42 | 1.5630 | 9.566 |

The equal-scene mean of these medians is 1.306247 seconds; frame counts differ, so this is not a fixed-length benchmark. Speedup against original Dense is N/A: no contemporary paired Dense timing; Dense's reference definition is 1.00x. Speedup against fast is also N/A: historical fast timing is not a contemporary matched control. Quality has not passed, so no 7Scenes all-seven 128-frame speed acceptance has been performed. No speed-gate success is claimed.

Quality uses the same checkpoint SHA, frozen frame lists, preprocessing and evaluation implementation as the saved controls. Threshold 0.25m, downsampling/TSDF voxel 0.0390625m, sdf_trunc 0.2m, mesh sampling 1,000,000; unchanged unposed Umeyama/RANSAC alignment. SAF3R Top-K/profile is N/A. Exact settings, inputs, raw six metrics, per-layer state statistics and timing samples are under `outputs/logit_state_loss_20261108/eth3d_group_mean/`. Summary tables are alongside that directory. The full local research archive also contains the six rejected alternatives and RGB diagnostics; see `REMOTE_CONTENTS.md` for what is published in this selected-method commit.

## Reproduce

Use an environment with PyTorch, Triton, addict and Open3D. Dataset and checkpoint paths are recorded in the existing frozen input manifests. They can be relocated without changing checkpoint SHA or the actual input frames; archive those relocation changes. Dataset and checkpoint files are not bundled.

```bash
python experiments/logit_group_mean/run_campaign.py --output outputs/group_mean_rerun/eth3d
```

The output directory must be new. The campaign runs only the changed method on GPU4/5/6 through the bundled exclusive monitor, checks checkpoint/frame manifests and the no-fine-QK/previous-only invariants, and evaluates both pose and recon_unposed. The legacy `--variant centered050` selector retained by the entry point installs group_mean_logit_cumulant050, verified in each layer record; it does not select the old four-probe forward.

CPU checks:

```bash
python -m pytest -q tests/test_logit_state_loss_models.py tests/test_previous_logit_delta.py tests/test_coarse_spatial_layout.py
```

The standalone numerical kernel check is `experiments/logit_group_mean/test_kernel.py`; run it only on an idle allocated GPU. Existing results show identical AV outputs to the previous formal sparse kernel for token lengths 256/289/512/735, and group-mean raw-logit discrepancy about 1e-7. Those numerical checks are diagnostics, with all six quality fields N/A; they are not separate complete quality evaluations.
