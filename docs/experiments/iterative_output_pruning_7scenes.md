# Iterative output-preserving pruning: 7Scenes diagnostic

## Protocol

For global attention layers 0--2, each patch-query/head starts with all patch
keys.  Special keys are always retained.  At each current support, the
profiler recomputes attention/output and scores a patch key `j` by its exact
single-key deletion effect:

```text
Delta_j = a_j / (1 - a_j) * ||v_j - o||_2
```

It then removes the lowest-cost batch and repeats at patch-key retention
levels 100%, 75%, 50%, 37.5%, and 25.68%.  The final budget is 2,660 of
10,360 patch keys per query; inputs are ten frames (0, 5, ..., 45) from each
of the seven 7Scenes calibration sequences.  NMSE compares sparse attention
output with the exact dense output at the same layer/head.

This is batched greedy local pruning, not an exhaustive globally optimal
subset search.  The analytic deletion cost was independently checked against
directly recomputing softmax after deleting one key (maximum absolute error
`1.8e-7`).

## Result

| Layer | Iterative-pruning mean NMSE at 25.68% | Static 2x2 reference NMSE |
| ---: | ---: | ---: |
| 0 | 0.065833 | 0.001576 |
| 1 | 1.758796 | 0.035269 |
| 2 | 0.270257 | 0.103813 |

The coarse batch-greedy rule should not be used as the deployment sparse
policy: local single-key costs do not compose into a low-error final set, with
the failure most acute in layer 1.  The result is useful as a diagnostic and
motivates a set-level/look-ahead objective or much finer pruning batches.

## Reproduction

```bash
GPU_IDS=0,1,2,3,5,6 bash scripts/run_vggt_iterative_pruning_7scenes.sh \
  outputs/iterative_pruning/7scenes_n10_s5_layers_00_02

PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/summarize_vggt_iterative_pruning.py \
  --input-root outputs/iterative_pruning/7scenes_n10_s5_layers_00_02 \
  --static-reference outputs/dense_sparse_patterns/7scenes_n10_s5_layers_00_02/selected_cross_scene_static_2x2_patterns.json
```
