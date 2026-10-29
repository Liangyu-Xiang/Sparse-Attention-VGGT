# Original reuse: history counts

7Scenes all seven scenes,128frames. Counting replays only; no new quality or speed comparison. Each cluster/query-group has0..4 reused historical Key IDs; full exact-expansion members are separate. Layer IDs0..23. Percentages are equal-scene macro of cluster/query-group frequencies.

| Layer | 0 history % | 1 % | 2 % | 3 % | 4 % | Mean history / cluster / query group |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 100.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 100.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 99.996248 | 0.001432 | 0.000716 | 0.000468 | 0.001136 | 0.000088 |
| 3 | 98.833155 | 0.487181 | 0.145851 | 0.288504 | 0.245308 | 0.026256 |
| 4 | 97.827321 | 1.199438 | 0.369731 | 0.428582 | 0.174929 | 0.039244 |
| 5 | 96.865515 | 1.792409 | 0.614620 | 0.556820 | 0.170636 | 0.053747 |
| 6 | 95.846318 | 2.552804 | 0.851280 | 0.596763 | 0.152835 | 0.066570 |
| 7 | 95.288540 | 3.085751 | 1.021158 | 0.501026 | 0.103524 | 0.070452 |
| 8 | 98.091456 | 1.026393 | 0.604569 | 0.243779 | 0.033805 | 0.031021 |
| 9 | 98.004965 | 1.211196 | 0.445145 | 0.248874 | 0.089820 | 0.032074 |
| 10 | 97.811454 | 1.397529 | 0.501458 | 0.249566 | 0.039993 | 0.033091 |
| 11 | 96.690438 | 1.871477 | 0.766924 | 0.422707 | 0.248454 | 0.056673 |
| 12 | 82.923928 | 7.884838 | 4.173155 | 2.895296 | 2.122783 | 0.334082 |
| 13 | 78.389345 | 11.703820 | 5.392192 | 2.828750 | 1.685893 | 0.377180 |
| 14 | 78.262066 | 12.054981 | 4.921238 | 2.554944 | 2.206771 | 0.383894 |
| 15 | 80.036639 | 11.054399 | 4.625802 | 2.381701 | 1.901459 | 0.350569 |
| 16 | 86.983345 | 10.970923 | 1.642587 | 0.293972 | 0.109172 | 0.155747 |
| 17 | 97.264023 | 2.159098 | 0.390082 | 0.128155 | 0.058642 | 0.035583 |
| 18 | 99.862538 | 0.123988 | 0.008093 | 0.003194 | 0.002187 | 0.001585 |
| 19 | 99.986425 | 0.006067 | 0.002922 | 0.002497 | 0.002090 | 0.000278 |
| 20 | 99.959328 | 0.026768 | 0.001143 | 0.005232 | 0.007530 | 0.000749 |
| 21 | 99.910158 | 0.032016 | 0.016409 | 0.016552 | 0.024866 | 0.002140 |
| 22 | 99.952161 | 0.042500 | 0.001681 | 0.001083 | 0.002576 | 0.000594 |
| 23 | 99.999300 | 0.000643 | 0.000050 | 0.000004 | 0.000003 | 0.000008 |

## Whole-network macro

| Selected historical Keys | Fraction of cluster/query-group/layer combinations |
|---|---:|
| 0 | 94.949361% |
| 1 | 2.945235% |
| 2 | 1.104034% |
| 3 | 0.610353% |
| 4 | 0.391017% |

Mean per cluster/query-group: 0.085484295; conditional on nonzero reuse: 1.692544.

Per-cluster arrays are in scene/per_cluster_counts.npz. layerXX_histogram has shape [fine_clusters,5]; row c is cluster ID c, columns0..4 count how many of4164 query groups selected that number of historical Keys. Sum of each row=4164. Layer-specific population and representative_patch_id arrays are also stored (layers1..23). Full exact-expansion Keys are a separate state, not counted as reused-history-state edges.

These are fresh same-configuration replays. Original historical per-cluster histograms were not saved; do not claim these are recovered from the old run. Saved-run means are included beside replay means in each counts.json. L0 and L1 have zero selected history.
