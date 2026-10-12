# Progressive grid sparse-attention: Uniform-25% evaluation

## Evaluated configuration

The evaluated baseline keeps one key from each 2x2 spatial patch grid
(`grid_keep_ratio: 1.0`), which is approximately 25% of patch keys regardless
of the number of frames.  Patch queries attend only to selected patch keys;
special-token queries remain dense.  Evidence-based exploitation is disabled
for this baseline (`exploitation_ratio: 0.0`).  The implementation supports a
gradual 0%, 25%, 50%, 75% exploitation schedule when enabled, without a
maximum consecutive-layer constraint.

The evaluation used the VGGT checkpoint in
`configs/evaluation/eval_progressive_grid_vggt.yaml`, 128 uniformly sampled
frames per scene, and four GPU shards.  The post-processing summary below
deduplicates scenes before averaging.

## Results

| Dataset | Scenes | AUC@3 | AUC@5 | AUC@15 | AUC@30 | Accuracy | Completeness | Chamfer | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7Scenes | 7 | 22.09 | 36.06 | 70.63 | 84.33 | 0.1555 | 0.1304 | 0.1429 | 0.4313 |
| ETH3D | 11 | 6.51 | 14.37 | 43.15 | 61.36 | 0.6732 | 1.2669 | 0.9701 | 0.3562 |

`stairs` was accidentally evaluated in two shards during this run.  The table
uses it once only; the committed four-way configuration assigns it solely to
GPU 3, so future runs contain 18 unique scenes.

## Reproducibility

Run the four independent GPU shards with:

```bash
SAF3R_KDTREE_WORKERS=16 bash scripts/evaluate_progressive_grid_vggt.sh \
  outputs/progressive_grid_uniform_25pct
```

The evaluator uses `scipy.spatial.cKDTree` with the bounded worker count above
for reconstruction metrics.  This parallel metric path was added after the
recorded run, so it improves future evaluation time without changing its
numeric definition.
