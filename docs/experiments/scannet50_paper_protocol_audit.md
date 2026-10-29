# ScanNet50 paper protocol audit

Sources checked: https://arxiv.org/html/2607.03612 (v2), Sections 4.1, 4.3, 5.1, Table 4; https://github.com/jndeng/SAF3R/blob/main/configs/evaluation/eval_saf3r_vggt.yaml.

The paper selects per-frame Top-K using cosine similarity of final DINO features, caching indices within the forward for subsequent correspondence heads. The current official repository defaults to lazy first-correspondence-head Top-4, and retains a commented DINO Top-64 alternative. These are distinct configurations.

Table 4 ScanNet50 uses 300, 500 and 700 uniformly selected images, the FastVGGT evaluation protocol and H100 80 GiB. The previous local 128-frame run used lazy Top-4 and camera-centre Sim(3) / incremental voxel fusion, so it is a local matched experiment, not a Table 4 reproduction.

The paper gives K=16..128 depending on scene but no scene-to-K mapping. Fixed DINO K=64 is a documented candidate, not an exact recovered paper configuration. Config: configs/evaluation/scannet50_paper_dino64.yaml.

The repository DINO candidate has max_topk=2048. Its selection utility first selects K/frame and then globally caps to max_topk; this cannot preserve 64/frame over 300..700 images. The candidate config sets max_topk=44800 to avoid this hidden change. This may exceed local 32 GiB GPU capacity. OOM must be reported with all six quality fields unavailable; do not lower K, truncate inputs or silently enable a global cap.

Required next evaluation: match Dense and DINO settings at each frame count, preserve the paper FastVGGT trajectory/CD metrics separately from the required six metrics (CD=(ACC+Comp)/2 unsquared), report per-scene raw metrics and equal-scene macro, and include all DINO similarity/top-K construction in warmed full-forward timing. Prior 128-frame timing is not a paired Dense timing baseline for these runs. Per-scene K and any alternative global-cap configuration must be declared before evaluating test data.
