# Rep-based History Loss：ETH3D 3场景评测

固定布局修复后的原始四状态基线，仅验证用户给出的A/B。未引入新状态、Fine QK、每Query Group全量Value距离、M5c改动、聚类改动、预算改动或Attention kernel改动。特殊Query保留文档基线的camera-only设置，不混入上轮all-special修正。

A：max(0,L_Rep−sum_H p_h||V_h−V_rep||²)，实际support=H∪{rep}，cost去重；历史非rep权重1，rep权重=population−(support_count−1)。这是Value表示失真代理，不是softmax输出误差。

B：仅|H|=1时max(0,L_Rep+(pi_C−2p_h)||V_h−V_rep||²)；多历史Key保持原损失；support和cluster_population权重不变。

本批Baseline/A/B均重新推理与pose+recon_unposed评分。相同checkpoint、精确帧表、预处理与协议均审计通过。AUC/F1为百分数，ACC/Comp/CD为米，CD=(ACC+Comp)/2非平方距离；宏平均为场景等权平均。

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14.764243 | 78.158209 | 0.308703 | 0.508498 | 0.408601 | 55.402579 |
| A | 19.398096 | 77.820655 | 0.292947 | 0.454225 | 0.373586 | 58.641635 |
| B | 16.929832 | 79.601345 | 0.319433 | 0.478276 | 0.398855 | 57.617242 |

这是三个校准场景的小规模验证，不是完整11场景结果。

## 逐场景六指标

| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | pipes | 14 | 19.047619 | 84.615385 | 0.190343 | 0.302177 | 0.246260 | 78.470359 |
| baseline | courtyard | 38 | 16.642959 | 83.380749 | 0.328753 | 0.198338 | 0.263545 | 56.690394 |
| baseline | playground | 32 | 8.602151 | 66.478495 | 0.407014 | 1.024979 | 0.715997 | 31.046985 |
| A | pipes | 14 | 20.146520 | 84.908425 | 0.187351 | 0.200236 | 0.193793 | 80.675249 |
| A | courtyard | 38 | 28.639165 | 83.304884 | 0.273376 | 0.193396 | 0.233386 | 64.982052 |
| A | playground | 32 | 9.408602 | 65.248656 | 0.418114 | 0.969044 | 0.693579 | 30.267604 |
| B | pipes | 14 | 22.710623 | 87.326007 | 0.254978 | 0.219045 | 0.237011 | 77.310553 |
| B | courtyard | 38 | 18.065434 | 85.154101 | 0.311830 | 0.190889 | 0.251360 | 58.608144 |
| B | playground | 32 | 10.013441 | 66.323925 | 0.391492 | 1.024894 | 0.708193 | 36.933029 |

重建阈值0.25m，下采样/voxel 0.0390625m，TSDF trunc 0.2m，mesh采样1000000点；沿用原unposed融合/对齐、mask和深度处理。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；fine cluster fraction0.12，Query group256。方法不使用SAF3R Top-K/profile。

## 独立运行记录

以下为0暖机、1次完整冷前向，包含JIT/缓存建立；不作为速度比较。所有ETH3D相对Dense暖机加速为N/A，因为用户仅要求7Scenes速度。

| Method | Scene | GPU | Cold forward s | Peak allocated GiB | Actual Key density |
|---|---|---:|---:|---:|---:|
| baseline | pipes | 5 | 4.142365 | 8.6107 | 0.160438 |
| baseline | courtyard | 5 | 6.139636 | 9.5356 | 0.160375 |
| baseline | playground | 5 | 5.978561 | 9.3591 | 0.160403 |
| A | pipes | 6 | 4.956051 | 8.6133 | 0.160498 |
| A | courtyard | 4 | 6.762627 | 9.5349 | 0.160448 |
| A | playground | 4 | 6.475995 | 9.3577 | 0.160471 |
| B | pipes | 4 | 4.721221 | 8.6107 | 0.160453 |
| B | courtyard | 6 | 6.825899 | 9.5360 | 0.160376 |
| B | playground | 6 | 6.237414 | 9.3591 | 0.160410 |
