# ScanNet-50 DINO64: partial matched evaluation

300-frame cap, 42 common completed scenes. This is NOT a full 50-scene macro.
Non-lazy DINO per-frame Top64 with max2048 cap; ETH3D profile. Same checkpoint/input per pair. Uniform valid-pose sampling. No lowered image resolution, frame cap, or K.
Unposed predicted-camera depth unprojection; FastVGGT gauge and bounding-box scale/center alignment. Confidence>=1; 100k point cap; 5cm voxel and F1 threshold; distances clipped at0.5m. CD=(ACC+Comp)/2, unsquared metres. Not the official sum-CD.

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| dense | 12.77721 | 80.47573 | 0.21545 | 0.24792 | 0.23168 | 15.87184 |
| saf3r | 9.64000 | 77.28279 | 0.21235 | 0.24028 | 0.22632 | 15.35060 |

Timing uses only 36 uncontended matched pairs: Dense=72.8585s, SAF3R=27.9239s, speedup=2.6092x. One warmup and three complete forwards per run. Includes DINO similarity/caching. Other timings must not be included in this speedup.

Completion: 300 Dense48/50, SAF3R44/50; 500 Dense49/50, SAF3R0/50; 700 both0/50. Preflight OOM prevented remaining 500-SAF3R/700 tasks from starting. Six fields for failed/not-started runs are N/A with reasons in failure.json/quality.json/blocked_jobs.json.

## Exact matched scenes
```json
[
  "scene0000_00",
  "scene0013_02",
  "scene0042_02",
  "scene0056_00",
  "scene0071_00",
  "scene0084_01",
  "scene0096_00",
  "scene0109_00",
  "scene0121_01",
  "scene0136_01",
  "scene0150_00",
  "scene0164_01",
  "scene0177_01",
  "scene0194_00",
  "scene0207_01",
  "scene0238_00",
  "scene0267_00",
  "scene0280_00",
  "scene0294_02",
  "scene0309_00",
  "scene0325_01",
  "scene0340_01",
  "scene0353_02",
  "scene0367_01",
  "scene0380_02",
  "scene0395_00",
  "scene0435_03",
  "scene0451_01",
  "scene0466_01",
  "scene0477_00",
  "scene0493_01",
  "scene0509_01",
  "scene0525_00",
  "scene0555_00",
  "scene0571_00",
  "scene0582_02",
  "scene0606_01",
  "scene0619_00",
  "scene0631_01",
  "scene0648_00",
  "scene0663_01",
  "scene0691_00"
]
```
