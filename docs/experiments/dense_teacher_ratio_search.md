# Dense-teacher diagnostic search of sequence-specific A/V ratios

User-authorized experiment: improve the poor sequences while preserving sparse
inference speed, using Dense as a teacher to find better ratios. This is an
**offline, sequence-specific oracle study**, not an online adaptive policy or
held-out paper comparison. Selecting the target scenes by observed degradation
does not make them held-out. No7Scenes GT metric is used to rank ratios.

Targets fixed before search: heads, office, stairs. Heads/office lose >10pp F1;
stairs loses >5pp AUC30 under the existing manual History80 scheme. Keep128
uniform frames, seed0, same checkpoint/preprocessing, all24 global layers
sparse,20% target budget,80% next-layer Attention history, whole-Parent selection,
32-Query shared supports, original execution kernel. Layer0 stays pure Value.

Search75 schedules: Attention proportions for layers1–9 and17–23 each take
0/20/50/80/100%; layers10–16 take50/80/100%. The current20/80/20 schedule is
included. Ratios allocate patch edge counts; they are not output mixture weights.
Every candidate keeps the same base fine count because layer0 remains pureValue.

For one unmodified Dense trajectory, replay the actual sparse selector and
kernel on16 uniformly spaced complete32-Query groups at each layer. All real
Keys remain available to the exact Dense reference. Each candidate has its own
causal history, with outgoing counts based on its known next-layer ratio.
Value fine clustering is shared across candidates only because their inputs
are identical **Dense** hidden states. Representative merging, Parent scores,
history deduplication, full-Parent expansion, unified softmax and tile history
use the production code. CPU and BF16 GPU tests compare sampled replay against
the full production forward under identical Dense-forced inputs/history.

Rank by mean per-layer relative squared pre-projection Attention AV error
(layers1–23), using even sample groups. Report odd-group error separately;
held-out probe error and final GT scores do not select candidates. This is a
finite-grid sampled surrogate, not globally optimal sequence-level pose/F1.
Teacher forcing does not capture accumulated sparse hidden-state drift.

Run the top2 schedules and the manual schedule end-to-end, without Dense hooks.
Measure each on the same GPU as a fresh Dense baseline:1warmup+3full forwards,
medians; output AUC3,AUC30,ACC,Comp,CD,F1,speedup and peak memory. A speed-feasible
recommendation must be no slower than both the manual sparse baseline and Dense
by measured medians. Rank eligible candidates by Dense-teacher error, never GT.
Report all quality changes; a timing pass does not imply a quality pass. No
acceptance threshold is loosened and no Dense fallback is inserted.

Dense/search cost is recorded separately. Sparse-only latency describes
inference with a frozen known schedule. If this search is run for every new
sequence, its full cost must be included and no online acceleration is implied.
The next stage, only if this diagnostic succeeds, is to approximate its ratios
using deployable low-cost signals and independent calibration data.

```bash
/data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/run_dense_ratio_search.py \
  --output-dir outputs/parent_region_av/dense_teacher_ratios_128 \
  --scenes heads office stairs --frames 128 --gpus 5 4 6
```

GPU4–6 are assigned dynamically when idle. GPU4 first waits for the prior
128-frame queue, GPU6 for the prior300-frame comparison.
Workers dynamically claim remaining scenes only when idle. Original results
are preserved; source/config snapshots and status files track the new study.
