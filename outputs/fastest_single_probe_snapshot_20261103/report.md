# Fastest completed sparse candidate: experimental snapshot

**Status: severe quality regression; not a validated quality-preserving optimization.**

`single_probe`: warmed median full-forward **12.832572641 s**, heads/128 frames, RTX 5090 D GPU5, one warmup and three timed forwards. Includes clustering, routing, history, reordering and output heads; excludes model/image loading and GT evaluation. Single CUDA process; CPU launcher creates no CUDA context. Peak allocated memory (maximum of three samples): **15.77969 GiB**.

Fastest among the completed recorded sparse full-forward candidates checked at publication. Ongoing experiments are not included. Different batches do not establish paired speedups. Dense speedup: **N/A**, no contemporaneous Dense measurement; unchanged controls were not rerun. Dense identity is 1.00x. This is not the layered no-Drop / Rep-always method.

Frozen configuration: retention cap20%, adaptive-local clustering, one probe, 16 budget steps, eight attention parts. Configuration is configs/evaluation/reuse_light_single_probe.json. Method formulas differ from the previous v6/v7/v8 engineering candidates.

Quality: complete pose + recon_unposed for 7/7 7Scenes scenes (128 frames) and 11/11 ETH3D scenes (14–76 frames), equal-scene macro. AUC/F1 percentages below, ACC/Comp/CD metres; CD=(ACC+Comp)/2, unsquared. Raw JSON and per_scene.csv use AUC/F1 fractions. Reconstruction threshold/voxel: 7Scenes0.05m/0.0078125m; ETH3D0.25m/0.0390625m. Inputs/checkpoint/preprocessing/protocol audits pass for all18 runs. Quality-run timings are separate single samples, not the benchmark medians.

| Dataset | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7scenes | 7 | 0.03750 | 11.36588 | 3.79284 | 0.53262 | 2.16273 | 1.95361 |
| eth3d | 11 | 0.21268 | 12.78051 | 1.42488 | 5.40327 | 3.41408 | 12.65176 |

The severe accuracy collapse prevents using this candidate as a replacement for the original method. Earlier lightweight report completeness failures concerned other variants; this selected candidate has all18 complete raw quality results. No missing score is filled or shifted.

All six serial single-process variant timings and monitors are retained under timing/ for selection provenance. The timing-only result has all six quality fields unavailable with its diagnostic reason; the separate quality/ results supply the actual quality evaluations. Source files were copied from the hashed immutable quality source, not the actively edited worktree. External data/checkpoints are not uploaded; configure checkpoints/ and workspace/ locally.

Run `python scripts/verify_fastest_single_probe_snapshot.py` for CPU-only artifact/hash/metric validation. The recorded timing launcher is under timing_launcher/; use its --single-process --strict-contexts options for reproductions. No new GPU run was performed for publication.
