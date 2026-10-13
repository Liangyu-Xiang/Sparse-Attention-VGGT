# Sparse-support transfer across VGGT layers

## Protocol

This diagnostic evaluates all 16x16 source/target head pairs on the seven
7Scenes scenes.  Each scene uses frames 0 and 20 and one GPU.  For a source
head, the experiment retains its per-query Top-25% patch-key indices.  A later
target head receives only these indices and recomputes its own QK logits,
restricted softmax, and V aggregation.  Special tokens are excluded from both
the query and key domains.

The target dense patch-only output is the reference.  Comparisons include the
target's own Top-25% QK support and a uniform 25% support.  Cross-layer
distances 1, 2, 4, and 8 are evaluated.  A fixed-mapping test selects the source
head on six scenes and evaluates it on the held-out seventh scene.

Normalized regret is

`(E_transfer - E_self) / (E_uniform - E_self)`.

It is reported only when the target-self baseline is better than uniform.
This restriction matters: target-self Top-QK is better than uniform for only
27.4%, 28.7%, 31.0%, and 33.6% of targets at distances 1, 2, 4, and 8.

## Broad transferability result

| Layer distance | LOSO median normalized regret | LOSO fraction <= 0.25 | LOSO beats uniform | Fixed-source scene agreement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.884 | 27.4% | 14.7% | 46.5% |
| 2 | 1.091 | 21.6% | 13.6% | 46.8% |
| 4 | 1.650 | 11.7% | 10.3% | 49.2% |
| 8 | 1.974 | 3.0% | 6.0% | 51.0% |

Thus a globally reusable head mapping is not supported.  Even at adjacent
layers, a source chosen on six scenes has median normalized regret 0.884 on the
held-out scene.  Transfer degrades as layer distance grows.

## Previously identified L10-L16 chain

The continuity-derived chain is more promising than arbitrary head pairs.

| Source -> target | Mean transfer error | Mean normalized regret | Beats uniform |
| --- | ---: | ---: | ---: |
| L10H3 -> L11H0 | 0.2319 | 0.7118 | 85.7% |
| L11H0 -> L12H8 | 0.0353 | 0.1857 | 100% |
| L12H8 -> L13H10 | 0.0110 | 0.0379 | 100% |
| L13H10 -> L14H10 | 0.0026 | 0.0053 | 100% |
| L14H10 -> L15H8 | 0.0035 | 0.0064 | 100% |
| L15H8 -> L16H2 | 0.0929 | 0.1330 | 100% |

Direct reuse without refreshing the source mask also works over part of this
chain:

| Source -> target | Distance | Mean transfer error | Mean normalized regret | Beats uniform |
| --- | ---: | ---: | ---: | ---: |
| L10H3 -> L12H8 | 2 | 0.0396 | 0.2092 | 100% |
| L10H3 -> L14H10 | 4 | 0.0088 | 0.0497 | 100% |
| L12H8 -> L16H2 | 4 | 0.1920 | 0.3736 | 100% |

## Conclusion

The experiment supports **selective local continuity**, not a universal
cross-layer reuse rule.  The middle of the identified L10-L16 chain is a good
candidate for an implementation-level sparse reuse experiment; arbitrary
heads or long-distance reuse should remain disabled or gated.  The frequent
failure of target-self Top-QK against uniform support also shows that QK rank
alone is not a reliable sparsification oracle for every head.

Raw per-scene arrays, pair tables, the cross-scene summary, and fixed mappings
are stored in `outputs/sparse_support_transfer/7scenes_top25/`.
