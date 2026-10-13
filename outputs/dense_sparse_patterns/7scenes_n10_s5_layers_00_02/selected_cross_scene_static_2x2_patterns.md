# Cross-scene dense-calibrated static 2x2 sparse patterns

The pattern was selected by averaging exact sparse-vs-dense output NMSE over 7 7Scenes sequences.  Each sequence uses ten frames (0, 5, …, 45), retains one patch key per 2×2 cell (~25.7%), and keeps all special keys dense.

Corner encoding: 0=even/even, 1=even/odd, 2=odd/even, 3=odd/odd patch row/column offsets.

| Layer | Best one-mask corner | One-mask mean NMSE | Head-rank-mod-4 baseline | Best per-head corners | Per-head mean NMSE | Dense-only winner heads |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 0 | 0.001844 | 0.001916 | c0: 0,2,7,10,12,13,15; c1: 4,5,9,14; c2: 1,11; c3: 3,6,8 | 0.001576 | none |
| 1 | 0 | 0.037811 | 0.060661 | c0: 1,3,4,5,6,7,8,9,10,11,15; c1: 12; c2: 0,2,13,14 | 0.035269 | none |
| 2 | 2 | 0.108763 | 0.108183 | c0: 5,7,8,11,12,13; c1: 3,9; c2: 1,2,4,6,10,14,15; c3: 0 | 0.103813 | 0, 1, 10, 14 |

The matching JSON is the executable per-head configuration.  It selects only fixed 2×2 masks; dense-only oracle candidates are recorded solely as diagnostics.
