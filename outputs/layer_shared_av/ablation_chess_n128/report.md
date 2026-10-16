# Chess128 sensitivity checks

One cold forward per trial, not formal warm timing. Profile targets are Attention shares [0,1,0] for zero-based layers [3–9,10–16,17–23].
Default trial configuration: multiplier32, tau.8, eta.25%, fine.15, coarse128, query chunk128.

| Trial | AUC30 (%) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) | Profile MAE | Cold whole (s) |
|---|---:|---:|---:|---:|---:|---:|
| scale16 | 30.315 | 0.00 | 26.19 | 0.00 | 0.246 | 58.32 |
| scale64 | 35.741 | 16.08 | 54.99 | 3.63 | 0.216 | 167.22 |
| tau09 | 85.829 | 17.00 | 50.90 | 7.89 | 0.247 | 170.82 |
| tau095 | 13.049 | 10.16 | 48.23 | 0.85 | 0.209 | 167.90 |
| continuous | 56.551 | 10.24 | 49.97 | 3.16 | 0.211 | 230.33 |
| eta01 | 14.418 | 6.22 | 45.03 | 0.00 | 0.204 | 98.41 |
| eta005 | 15.142 | 6.18 | 49.66 | 0.00 | 0.188 | 107.52 |
| eta001 | 2.012 | 7.03 | 42.76 | 0.00 | 0.214 | 94.98 |
| fine020 | 51.953 | 3.02 | 44.67 | 0.00 | 0.194 | 84.51 |
| coarse064 | 28.842 | 6.69 | 44.73 | 0.00 | 0.207 | 87.91 |
| coarse256 | 32.119 | 3.87 | 46.58 | 0.00 | 0.191 | 92.79 |

The profile-only screen selected tau.8 but real sparse accuracy rejected it. Tau.9 was the only single-parameter trial above a Chess AUC30 floor of Dense-minus-3pp (85.466%).
That accuracy gate is a post-screen selection using Chess only, not a preregistered threshold. Other scenes do not select the final threshold. Revalidate the selected policy on all seven scenes, with larger query chunks tested separately for numerical equality.
Large accuracy swings across tau and sampling fractions indicate genuine fragility despite bitwise repeatability within a fixed configuration. Matching the desired regional means does not establish a robust method.
