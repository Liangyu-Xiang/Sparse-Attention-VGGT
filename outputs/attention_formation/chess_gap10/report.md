# Attention structure formation: Dense Chess frames0 and10

No sparse model changes. Head-mean **probabilities**, not softmax of mean QK. Main table excludes special Queries/Keys and conditionally normalizes patch attention; all-token results are separately recorded in all_token_metrics.csv. All2072 patch Queries, no Query sampling. K=207; candidate sizes207/414/828.

| Pair | C previous | C next | Top10 overlap % | Top20 recall % | Top40 recall % | Coarse20 recall % | Coarse20 Key coverage % | Actual QK coarse20 recall % | Actual QK Key coverage % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6→7 | 0.0625 | 0.0661 | 75.07 | 95.77 | 99.91 | 65.82 | 39.96 | 18.92 | 11.24 |
| 7→8 | 0.0661 | 0.0187 | 56.27 | 84.00 | 99.19 | 46.94 | 35.57 | 51.55 | 22.68 |
| 8→9 | 0.0187 | 0.0642 | 68.31 | 90.45 | 98.92 | 43.01 | 33.88 | 46.46 | 21.08 |
| 9→10 | 0.0642 | 0.0708 | 54.11 | 75.62 | 93.91 | 44.90 | 28.40 | 34.36 | 14.84 |
| 10→11 | 0.0708 | 0.1768 | 57.24 | 80.47 | 96.08 | 63.31 | 34.46 | 33.14 | 28.99 |
| 11→12 | 0.1768 | 0.1705 | 59.14 | 81.19 | 95.40 | 76.34 | 44.60 | 43.06 | 30.91 |
| 12→13 | 0.1705 | 0.1597 | 60.23 | 80.67 | 92.97 | 64.61 | 27.19 | 36.25 | 21.23 |
| 13→14 | 0.1597 | 0.2760 | 59.36 | 83.06 | 96.95 | 69.13 | 26.89 | 32.57 | 24.28 |
| 14→15 | 0.2760 | 0.1464 | 70.76 | 90.01 | 97.88 | 75.80 | 30.93 | 57.95 | 21.43 |
| 15→16 | 0.1464 | 0.0512 | 61.12 | 81.19 | 93.78 | 59.21 | 27.95 | 45.57 | 22.30 |

Coarse groups are computed independently at each source layer, but BOTH the source candidate and next-layer target are mapped into that SAME source grouping. No future grouping or future attention is used to rank candidates. Dense mass is a diagnostic reference, not the cheap sensing operator. High coarse recall is not evidence of useful localization without inspecting its actual Key coverage and the random baseline.

Chance fine recall is9.99/19.98/39.96% for10/20/40% candidates. Query-independent Key-saliency recall, query specificity D, coarse10/40%, and chance-corrected coarse recall are included in patch_metrics.csv.

This two-frame single-scene diagnostic cannot establish a universal bootstrap layer, or guarantee that a cheap sparse sensing/tracking mechanism attains Dense-oracle predictability.
