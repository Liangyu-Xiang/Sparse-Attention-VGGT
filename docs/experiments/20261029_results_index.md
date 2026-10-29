# 2026-10-29 实验结果索引

本页索引本轮已完成的本地实测、原始指标、输入列表、源码快照及验证记录。全部结果保留原始尺度；不同计时批次在各报告中注明。

## 最新：新的 cosine-distortion 目标及对应质量转移

[完整报告](../../outputs/reuse_cosine_7scenes128_20261029/report.md) · [逐场景与宏平均](../../outputs/reuse_cosine_7scenes128_20261029/summary.json) · [状态分布](../../outputs/reuse_cosine_7scenes128_20261029/state_behavior.json) · [配对核对](../../outputs/reuse_cosine_7scenes128_20261029/pairing_audit.json)

7Scenes 全七场景、128 帧，28/28 个有效运行完成。固定原版 2×2 grid、32-query 分组和20%边预算；新方案包括逐token注意力加权cosine distortion及状态质量转移。Drop=2π，Exact=0；使用固定1e-6数值先验平滑。无测试真值调参。29项CPU/CUDA测试与真实128帧中间量数值检查通过。

| 方法 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | 平均预热中位秒 | 平均峰值GiB | Dense加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 23.668916 | 84.638873 | 0.153741 | 0.125616 | 0.139678 | 48.012766 | 12.280962 | 14.359506 | 1.000000x |
| saf3r_dino64 | 24.062031 | 84.684395 | 0.142626 | 0.126598 | 0.134612 | 46.589959 | 7.049674 | 18.912240 | 1.742061x |
| original | 23.248266 | 84.275989 | 0.126817 | 0.115065 | 0.120941 | 50.258228 | 81.654497 | 15.497243 | 0.150402x |
| cosine_distortion | 8.025755 | 75.555282 | 0.332110 | 0.124529 | 0.228320 | 18.996635 | 80.347683 | 15.498739 | 0.152848x |

新目标及质量转移联合变体的质量显著低于本轮原版对照，且几乎退化为纯Rep状态。原版对照在本轮重新测量，与前轮有小幅数值/离散状态复现波动；本轮比较使用本轮对照。初始GPU历史张量步长假设错误的尝试已作废并归档，不计入有效结果。

[实现及工程边界说明](reuse_cosine_distortion.md) · [引用目标函数定义](reuse_cosine_distortion_reference.md) · [有效全规模数值检查](../../outputs/reuse_cosine_corrected_fullscale_20261029/numerical_checks.json)

## 其他 Reuse 变体与 SAF3R

- [统一比较](../../outputs/reuse_variants_consolidated_20261029/report.md)：保留不同批次，包含 Dense、原版、shift8、最大误差、单次QK、4×4/8×8以及两种SAF3R。
- [三个独立改动](../../outputs/reuse_single_change_7scenes128_20261029/report.md)：shift8七场景OOM，六项质量与速度均不可用；其余变体完成。
- [canonical grid细化](../../outputs/reuse_canonical_grid_7scenes128_20261029/report.md)：4×4、8×8均完成。Dense/original对照通过仓库内相对符号链接复用。
- [SAF3R匹配比较](../../outputs/saf3r_matched_reuse_comparison_20261029/report.md)：DINO每帧Top-64和lazy first-correspondence Top-4分别标注，21/21运行完成，使用新配对Dense计时。
- [历史选择数量](../../outputs/reuse_history_counts_20261029/report.md)：同配置计数重放；每场景per_cluster_counts.npz保存逐层逐cluster的0..4条历史数量分布。诊断不含新的质量分数或速度比较。
- [heads128耗时分配](../../outputs/reuse_heads128_detailed_profile_20261029/allocation_breakdown.md)：含逐层细分；CUDA-event诊断包含插桩和启动间隙，与未插桩时间分别记录。

## ScanNet50

- [128帧lazy Top-4本地匹配实验](../../outputs/scannet50_matched_128_20261029/report.md)：50场景全部完成；使用声明的本地重建协议，不能称为论文Table4复现。
- [300帧DINO Top-64尝试](../../outputs/scannet50_dino64_300_20261029/report.md)：Dense50场景成功，SAF3R50场景OOM。
- [500帧尝试](../../outputs/scannet50_dino64_500_20261029/report.md)、[700帧尝试](../../outputs/scannet50_dino64_700_20261029/report.md)：Dense和SAF3R均全部OOM，六项质量均不可用，不是完整质量评估。
- [论文协议核对](scannet50_paper_protocol_audit.md)：论文DINO特征Top-K与仓库lazy默认分别记录；未公开的逐场景K映射未声称恢复。

## 报告约定和复现入口

所有默认质量运行包含pose和recon_unposed；AUC/F1标注百分比，7Scenes距离为米，CD=(ACC+Comp)/2且使用未平方距离。7Scenes重建阈值0.05m、下采样0.0078125m。速度使用相同输入/硬件/协议的配对Dense预热中位完整前向时间比，Dense=1.00×，低于1×为更慢；包含初始化、聚类、路由、历史和重排。模型/图像加载位于完整前向计时外。

主要入口：`scripts/run_reuse_cosine_eval.py`、`scripts/run_reuse_ablation_eval.py`、`scripts/run_reuse_grid_ablation.py`、`scripts/run_saf3r_comparison.py`。输入、checkpoint路径和源码SHA256保存在各run目录；运行脚本的数据和Python环境路径对应本地实验主机，异机复现需配置相应路径。

无效/中断/OOM诊断保留具体原因及六个N/A字段；`reuse_cosine_invalid_stride_20261029`、早期已标记invalid的cosine诊断及`timing_excluded`均不参与最终汇总。
