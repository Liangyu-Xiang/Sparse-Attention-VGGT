# efficiency v6 frozen experimental snapshot

**Status: experimental; ETH3D quality regression. Do not treat this as a validated equivalent optimization.**

Identified18-second result: heads/128frames, RTX5090 D; complete-forward warmed median **18.109623s** versus immediate grid-local reference **26.100919s**, **1.4413x** relative to that reference. This is not a Dense speedup or a dataset mean. Three timed repeats per variant, one warmup; full reset/clustering/routing/history/reordering and output heads included. Image/model loading and GT evaluation excluded. See full_forward/result.json and monitor.json.

The separate quality campaign completes pose and recon_unposed for7/7 fixed7Scenes sequences(128frames each) and11/11 ETH3D scenes(14–76frames). Each quality run uses1warmup+1timed forward; those seconds are single samples, not warmed3-repeat medians. Controls are historical and labeled by batch in per_scene.csv. No unchanged controls were rerun for publication.

Local measured quality, scene-equally weighted; AUC/F1 percent, distances metres, CD=(ACC+Comp)/2 unsquared:

|Dataset|Method|Scenes|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Mean scene seconds*|Peak GiB|Dense speedup|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|7scenes|dense|7|23.66892|84.63887|0.15374|0.12562|0.13968|48.01277|12.28930|14.360|1.00x|
|7scenes|saf3r|7|23.93490|84.71310|0.14326|0.12788|0.13557|47.31542|6.44042|14.615|N/A|
|7scenes|grid_local|7|22.12341|84.41707|0.14673|0.12431|0.13552|45.21774|27.31123|17.207|N/A|
|7scenes|efficiency_v6|7|22.14215|83.96454|0.14769|0.12186|0.13477|46.98241|18.99606|18.816|N/A|
|eth3d|dense|11|29.18346|81.81621|0.31319|0.84885|0.58102|62.57314|1.86725|10.606|1.00x|
|eth3d|saf3r|11|23.33220|80.17474|0.30311|0.85772|0.58041|62.83039|1.57212|10.686|N/A|
|eth3d|grid_local|11|23.95338|79.26978|0.32686|0.87532|0.60109|56.86820|4.43206|11.228|N/A|
|eth3d|efficiency_v6|11|19.85806|76.70152|0.38170|0.96712|0.67441|49.61912|3.27552|12.901|N/A|

*Candidate dataset seconds are single measured forwards after warmup; historical controls have their saved repetition protocol. They are not paired speed comparisons. Dense identity is1.00x; all candidate Dense speedups are N/A because no contemporaneous Dense measurement exists.

## Quality differences versus matched local references

|Dataset|Reference|AUC@3 pp|AUC@30 pp|ACC relative %|Comp relative %|CD relative %|F1 pp|
|---|---|---:|---:|---:|---:|---:|---:|
|7scenes|dense|-1.52676|-0.67433|-3.93823|-2.99181|-3.51266|-1.03035|
|7scenes|grid_local|+0.01875|-0.45252|+0.65346|-1.97409|-0.55167|+1.76467|
|eth3d|dense|-9.32541|-5.11469|+21.87267|+13.93228|+16.07236|-12.95401|
|eth3d|grid_local|-4.09532|-2.56826|+16.77725|+10.48789|+12.19791|-7.24908|

ETH3D regression against grid_local: AUC@3 about−17.1% relative, CD about+12.2% relative. The optimizing chat is still investigating. No scores are shifted/scaled, budgets changed, or gates relaxed to conceal this drift. This snapshot is separate from the layered no-Drop / Rep-always method in the requesting chat. It retains uniform20% retention and its original Drop/History state rules.

7Scenes threshold0.05m / voxel0.0078125m; ETH3D threshold0.25m / voxel0.0390625m. SAF3R references are lazy first-correspondence-head Top-4, ETH3D profile, not paper DINO Top-K. Inputs/checkpoint/preprocessing/quality protocols match each saved control and audit records are in summary.json. No published-reference scores or H100 timings are mixed in.

Source provenance: the immutable source/ in efficiency_quality_v6_20261103, not the actively changing optimization worktree. All packaged code/config/scripts match the recorded source hashes. Package manifest records original absolute local paths for provenance; data/checkpoints are external and are not uploaded.
