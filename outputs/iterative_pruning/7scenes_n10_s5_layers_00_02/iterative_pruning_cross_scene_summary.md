# Iterative output-preserving pruning: cross-scene fidelity

Per-query patch keys are greedily removed in batches at 100% → 75% → 50% → 37.5% → 25.68%. All special keys remain dense; each reported NMSE compares the sparse output with the exact dense output at the same layer and head.

| Layer | Iterative-pruning mean NMSE at 25.68% | Static 2×2 reference NMSE |
| ---: | ---: | ---: |
| 0 | 0.065833 | 0.001576 |
| 1 | 1.758796 | 0.035269 |
| 2 | 0.270257 | 0.103813 |

## Mean pruning curves

- Layer 0: 100.00%: 0.000000; 75.00%: 0.002776; 50.00%: 0.018493; 37.50%: 0.031168; 25.68%: 0.065833
- Layer 1: 100.00%: 0.000000; 75.00%: 0.074968; 50.00%: 0.468667; 37.50%: 0.643717; 25.68%: 1.758796
- Layer 2: 100.00%: 0.000000; 75.00%: 0.016510; 50.00%: 0.087416; 37.50%: 0.149128; 25.68%: 0.270257
