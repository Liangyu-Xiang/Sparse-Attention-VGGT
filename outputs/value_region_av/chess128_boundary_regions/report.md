# Value-based whole-region Attention: Chess128 comparison

Same128 input frames, checkpoint,392×518 resolution, BF16, one warmup and three measured forwards. No primary per-layer hooks/synchronizations. Includes all hierarchy/selection/packing/history costs. Local/frame attention unchanged; all24 global layers use the manual mixed schedule.

| Method | Full seconds ↓ | Speedup ↑ | Peak allocated GiB ↓ | AUC5 % ↑ | AUC15 % ↑ | AUC30 % ↑ |
|---|---:|---:|---:|---:|---:|---:|
| dense | 16.4117 | 1.000× | 14.3586 | 39.6949 | 77.3671 | 88.4662 |
| fused | 14.2988 | 1.148× | 14.9271 | 37.6427 | 75.5668 | 87.3659 |
| region | 12.5754 | 1.305× | 14.9269 | 36.7692 | 75.0074 | 87.0571 |

New vs old latency reduction: 12.05%.
Actual unique edge density: 19.999549%–20.001413%. Largest per-Query-group budget overshoot: 20 edges; every layer satisfies overshoot < maximum fine cluster population.

## Independent profile (seconds, not the primary timing)

| Stage | Old fused | Whole regions |
|---|---:|---:|
| Global modules | 9.2909 | 7.9155 |
| Value | 0.0480 | 0.0475 |
| Hierarchy / pooling | 0.0349 | 0.0410 |
| Selection incl. terminal / expansion | 4.0142 | 2.5244 |
| Terminal real-Key QK (included in old selection) | 1.7761 | 0.0000 |
| Parent scoring / ranking | 0.0000 | 0.7659 |
| Crossing-parent child refinement | 0.0000 | 0.4633 |
| Crossing-child fine refinement | 0.0000 | 0.4204 |
| Whole-region expansion | 0.0000 | 0.3074 |
| Final unified attention | 3.6468 | 3.7007 |
| History | 0.5553 | 0.5525 |

Profile rows have containment relationships; do NOT sum selection and its sub-stages.

## Per-layer edge budgets and selected complete regions

| Layer | Target A keys | Value keys | Unique edge % | Max overshoot | Whole parent regions | Whole child regions | Whole fine regions |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 26009 | 19.999549 | 0 | 0 | 0 | 0 |
| 1 | 5202 | 20807 | 20.001319 | 14 | 112445 | 5051 | 13212 |
| 2 | 5202 | 20807 | 20.001124 | 13 | 85352 | 7332 | 14706 |
| 3 | 5202 | 20807 | 20.001273 | 14 | 94807 | 5616 | 15197 |
| 4 | 5202 | 20807 | 20.001134 | 14 | 121138 | 5885 | 16060 |
| 5 | 5202 | 20807 | 20.001118 | 11 | 116648 | 6147 | 15622 |
| 6 | 5202 | 20807 | 20.001121 | 12 | 118029 | 6386 | 14976 |
| 7 | 5202 | 20807 | 20.001045 | 14 | 126292 | 6167 | 15053 |
| 8 | 5202 | 20807 | 20.001097 | 12 | 111163 | 5894 | 14569 |
| 9 | 5202 | 20807 | 20.001127 | 12 | 103172 | 6055 | 15308 |
| 10 | 20807 | 5202 | 20.001212 | 14 | 350277 | 5920 | 15837 |
| 11 | 20807 | 5202 | 20.001236 | 11 | 400432 | 6023 | 16076 |
| 12 | 20807 | 5202 | 20.001261 | 17 | 430674 | 6116 | 16237 |
| 13 | 20807 | 5202 | 20.001264 | 14 | 410596 | 6071 | 15852 |
| 14 | 20807 | 5202 | 20.001251 | 12 | 414821 | 6115 | 16733 |
| 15 | 20807 | 5202 | 20.001356 | 14 | 452376 | 5882 | 15715 |
| 16 | 20807 | 5202 | 20.001411 | 15 | 420701 | 5786 | 15097 |
| 17 | 5202 | 20807 | 20.001413 | 20 | 117209 | 5672 | 14088 |
| 18 | 5202 | 20807 | 20.001245 | 12 | 121474 | 5617 | 13144 |
| 19 | 5202 | 20807 | 20.001195 | 11 | 123610 | 5566 | 14389 |
| 20 | 5202 | 20807 | 20.001197 | 15 | 107025 | 5750 | 13481 |
| 21 | 5202 | 20807 | 20.001143 | 12 | 123409 | 5904 | 14521 |
| 22 | 5202 | 20807 | 20.001213 | 11 | 101532 | 5700 | 13250 |
| 23 | 5202 | 20807 | 20.001039 | 12 | 107357 | 5763 | 13699 |

## Scope / limitations

The old fused selector remains available as mode=fused. mode=region uses the same Value maintenance and same manual20% target schedule, but selects complete Value-based regions, with explicitly allowed fine-cluster overshoot. These are different masks, not a bitwise-equivalent kernel optimization.

The first parent boundary alone is refined to its children; the first child boundary alone is refined to fine clusters. Other chosen parents/children are final complete regions. Fine clusters are never cut. Parent scoring remains Query-group specific. A Value representative/history Key excludes only that Key, not the whole partially covered cluster.

No terminal real-Key QK, no full selected logits, and no Token×Token mask. Expansion uses a bounded window-of-Query-groups × patch-count uint8 history lookup and compact support indices; this workspace and its clearing/packing are included in time and memory.

History remains a same-pass tile-mass approximation. All heads retain their own Q/K/V and unified softmax. This single-scene pose experiment is not a full seven-scene quality acceptance.

Acceptance checks: {"speed_2x": false, "memory_not_above_dense": false, "chess_auc30_loss_at_most_1pp": false, "full_seven_scene_three_task_assessment": "not_run"}
