# Reuse variants and SAF3R: consolidated local measurements

7Scenes all seven scenes, 128 uniform frames. Same exact input lists/checkpoint/preprocessing/hardware/pose and recon_unposed settings; pairing audit for new cohort passed for all21 runs. All jobs terminal: 56 successful +7 OOM out of63 primary method-scene combinations, plus7 fresh paired Dense timing controls. Diagnostics and superseded noisy timing runs excluded.

Quality is equal-scene macro. AUC/F1 percentages; distances metres. F1 threshold0.05m, downsampling0.0078125m; CD=(ACC+Comp)/2 unsquared distances. Runtime includes all complete model forward costs, with1 warmup +3 measurements. Peak GiB is equal-scene average of per-run median measured peak allocated GiB. Speedup is ratio of equal-scene mean median full-forward seconds for paired Dense/method; below1x means slower.

Timing cohorts: reuse methods retain their original matched Dense (12.282440s). SAF3R methods use freshly paired Dense (12.294450s); no historical timing rebasing. Fresh Dense quality equals original Dense quality exactly. Raw samples and hardware/protocol checks are retained.

SAF3R DINO64 uses final DINO feature cosine similarity, 64keys per frame, max_topk8192 (128*64; no cap beyond requested per-frame support), no guaranteed dense first frame. Its declared fixed K is a paper-mechanism reproduction setting; unpublished per-scene paper K is not claimed recovered. Lazy4 uses the first correspondence head. Both use configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json. These are local measured rows, with no published-reference rows or H100 timing baseline.

| Method | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean median s | Mean peak GiB | Speedup vs paired Dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGT Dense | 7/7 | 23.668916 | 84.638873 | 0.153741 | 0.125616 | 0.139678 | 48.012766 | 12.282440 | 14.359506 | 1.000000x |
| SAF3R DINO per-frame Top-64 | 7/7 | 24.062031 | 84.684395 | 0.142626 | 0.126598 | 0.134612 | 46.589959 | 7.042883 | 18.912240 | 1.745656x |
| SAF3R lazy first-correspondence Top-4 | 7/7 | 23.934899 | 84.713102 | 0.143263 | 0.127883 | 0.135573 | 47.315421 | 6.440425 | 14.615167 | 1.908950x |
| Reuse original 2x2 | 7/7 | 23.114689 | 84.188812 | 0.128410 | 0.115737 | 0.122074 | 49.743379 | 82.355772 | 15.496621 | 0.149139x |
| Query shift8 (OOM) | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Maximize state error | 7/7 | 0.002343 | 4.202814 | 0.596885 | 0.907816 | 0.752351 | 8.910193 | 81.916786 | 15.497871 | 0.149938x |
| Single QK saved logits | 7/7 | 23.629663 | 84.314480 | 0.128313 | 0.116898 | 0.122606 | 49.867651 | 85.644784 | 15.497455 | 0.143411x |
| Canonical grid 4x4 | 7/7 | 23.795463 | 84.559606 | 0.136435 | 0.117374 | 0.126904 | 48.748197 | 81.811923 | 15.524978 | 0.150130x |
| Canonical grid 8x8 | 7/7 | 23.343176 | 84.477526 | 0.141876 | 0.114727 | 0.128301 | 50.306009 | 83.404581 | 15.616847 | 0.147263x |

shift8: all7 scenes OOM in all-group historical support merging/sorting/coalescing. All six metrics and paired speedup remain N/A with per-run reasons. It is not a completed quality evaluation.

Per-scene and raw source reports:

- reuse_single_change_7scenes128_20261029/report.md
- reuse_canonical_grid_7scenes128_20261029/report.md
- saf3r_matched_reuse_comparison_20261029/report.md
- saf3r_matched_reuse_comparison_20261029/pairing_audit.json

## Cosine objective fresh cohort

Fresh matched Dense, SAF3R DINO64, original reuse and cosine-distortion+mass-transfer runs. Do not mix these refreshed original-control scores/timings with earlier-cohort scores. 28/28 valid runs complete; invalid initial GPU-layout attempts excluded. Same seven scenes/128frames/thresholds; no test-truth parameter tuning.

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean median s | Mean peak GiB | Speedup vs fresh Dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 23.668916 | 84.638873 | 0.153741 | 0.125616 | 0.139678 | 48.012766 | 12.280962 | 14.359506 | 1.000000x |
| saf3r_dino64 | 24.062031 | 84.684395 | 0.142626 | 0.126598 | 0.134612 | 46.589959 | 7.049674 | 18.912240 | 1.742061x |
| original | 23.248266 | 84.275989 | 0.126817 | 0.115065 | 0.120941 | 50.258228 | 81.654497 | 15.497243 | 0.150402x |
| cosine_distortion | 8.025755 | 75.555282 | 0.332110 | 0.124529 | 0.228320 | 18.996635 | 80.347683 | 15.498739 | 0.152848x |

Full raw and per-scene results: reuse_cosine_7scenes128_20261029/report.md, summary.json, pairing_audit.json and state_behavior.json.
