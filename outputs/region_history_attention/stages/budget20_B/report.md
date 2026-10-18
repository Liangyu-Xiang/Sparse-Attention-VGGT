# Actual intermediate Attention branch masks

Chess frames0/10, Dense0–9/sparse10–16/Dense17–23; B;50%history. All final masks and pose are bitwise identical to the previous unobserved run. No decisions or scores were recomputed by a substitute selector: a Python return observer captured the actual selector locals.

Blue marks membership in original2082x2082matrix coordinates. Region panels mean a Key belongs to a region under consideration; they do NOT imply that its real QK was computed. Only final_with_specials is the actual original-token QK/AV execution support. Pooled virtual-Key probes have no single original-token position.

The coarse top50%gate filters HISTORY ONLY. Unresolved exploration is the complement of history-covered fine cells over the whole patch space. It is NOT a sequential subset of the initial coarse gate. Level2panel is the union of groups containing quota-selected fine cells, including the partially used boundary group.

Stages1–6 show history/filtering and level1selection; stages7–12 show level2selection, fine quota, representative collapse, final support and outgoing history.

## Layer10

![filter](L10_filtering.png)

![explore](L10_exploration.png)

![one query](L10_one_query.png)

## Layer11

![filter](L11_filtering.png)

![explore](L11_exploration.png)

![one query](L11_one_query.png)

## Layer12

![filter](L12_filtering.png)

![explore](L12_exploration.png)

![one query](L12_one_query.png)

## Layer13

![filter](L13_filtering.png)

![explore](L13_exploration.png)

![one query](L13_one_query.png)

## Layer14

![filter](L14_filtering.png)

![explore](L14_exploration.png)

![one query](L14_one_query.png)

## Layer15

![filter](L15_filtering.png)

![explore](L15_exploration.png)

![one query](L15_one_query.png)

## Layer16

![filter](L16_filtering.png)

![explore](L16_exploration.png)

![one query](L16_one_query.png)
