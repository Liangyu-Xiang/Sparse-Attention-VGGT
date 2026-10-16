# Correct v3 policy: joint Chess128 calibration

All trials use token-edge history capacity and canonical coarse-QK tiles128. Eight trials are single cold forwards; (32,.9) reuses the completed three-repeat validation. Do not compare cold timings to warm medians.
Exploratory selection: require Value-majority in3–9/17–23 and Attention-majority in10–16, then maximize Chess AUC30. This is a qualitative profile constraint, NOT a hardcoded runtime layer schedule. All other scenes remain out of this selection.

| Scale | Tau | AUC30 (%) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) | Profile MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.8 | 38.275 | 0.00 | 24.58 | 0.00 | 0.251 |
| 16 | 0.9 | 10.136 | 5.50 | 27.68 | 0.00 | 0.259 |
| 16 | 0.95 | 84.425 | 8.33 | 31.67 | 3.27 | 0.266 |
| 32 | 0.8 | 22.175 | 13.59 | 49.25 | 0.00 | 0.214 |
| 32 | 0.9 | 13.614 | 16.85 | 47.92 | 0.00 | 0.230 |
| 32 | 0.95 | 83.972 | 11.16 | 52.15 | 1.39 | 0.201 |
| 64 | 0.8 | 9.815 | 8.65 | 58.39 | 4.14 | 0.181 |
| 64 | 0.9 | 17.895 | 12.66 | 59.20 | 1.74 | 0.184 |
| 64 | 0.95 | 8.050 | 14.01 | 57.66 | 0.00 | 0.188 |

Selected calibration candidate: {"logit_scale": 32, "dominance": 0.95, "sampling": 0.0025, "fine_ratio": 0.15, "coarse_size": 128, "query_chunk": 1024, "auc30": 83.97186679790028, "source": "outputs/layer_shared_av/joint_v3_chess_n128/scale32_tau0.95/chess/result.json", "repeats": 1, "early_attention": 0.11158039374690273, "middle_attention": 0.5214781276749111, "late_attention": 0.013864486191827724, "profile_mae": 0.2013222507546064, "dominance_profile": true}

Single-scene calibration is not evidence of generalization. Run the selected configuration on all seven scenes, report the independent six-scene macro mean separately, and retain all rejected configurations.
