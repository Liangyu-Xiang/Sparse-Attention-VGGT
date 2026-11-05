# 扩展状态＋缓存优化：完整 ETH3D 评测

11/11 个既有评测场景，共377帧；所有候选场景均在本批重新推理和评测，包括此前的三个校准场景。沿用既有ETH3D过滤协议：terrace/meadow按原设置排除，不新增过滤。

固定此前 states_only_fast 的代码、配置与逐层预算，不再调参。新增 Alternate/Pair 是在整体预算下供各簇选择的可选状态，不要求每簇保留一个Key。沿用原状态损失；本次不启用 normalized_moment 或概率质量权重校准。

与当前布局修复版对比，复用其完整历史质量结果，不重跑未变化的对照。全部输入帧表、checkpoint SHA、预处理、Pose协议、unposed融合/对齐、阈值以及评测源码均通过审计。

AUC/F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，使用非平方距离；宏平均为11场景等权平均。

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| layout_fixed | 18.510638 | 76.749655 | 0.323066 | 0.871509 | 0.597288 | 56.077724 |
| partial_states | 19.996072 | 77.635307 | 0.364432 | 0.908905 | 0.636669 | 51.589169 |

相对布局修复版：AUC@3 +1.4854个百分点，AUC@30 +0.8857个百分点，F1 -4.4886个百分点；CD +0.039381m。

## 逐场景六指标

| Scene | Method | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| facade | layout_fixed | 76 | 24.421053 | 87.207018 | 0.266299 | 0.260766 | 0.263533 | 62.902088 |
| facade | partial_states | 76 | 27.988304 | 88.039766 | 0.301889 | 0.282790 | 0.292340 | 59.589714 |
| terrains | layout_fixed | 42 | 7.394503 | 57.417731 | 0.306701 | 0.272322 | 0.289511 | 61.442925 |
| terrains | partial_states | 42 | 7.588076 | 56.310492 | 0.386313 | 0.356029 | 0.371171 | 52.945220 |
| delivery_area | layout_fixed | 40 | 13.888889 | 79.816239 | 0.556683 | 1.431668 | 0.994175 | 18.517464 |
| delivery_area | partial_states | 40 | 13.803419 | 79.982906 | 0.766656 | 1.669718 | 1.218187 | 10.891116 |
| electro | layout_fixed | 39 | 32.568601 | 91.421502 | 0.161282 | 4.637558 | 2.399420 | 67.550000 |
| electro | partial_states | 39 | 31.893837 | 91.470985 | 0.207418 | 4.730795 | 2.469106 | 64.066270 |
| courtyard | layout_fixed | 38 | 16.642959 | 83.380749 | 0.328753 | 0.198338 | 0.263545 | 56.690394 |
| courtyard | partial_states | 38 | 27.358938 | 86.761498 | 0.443436 | 0.264173 | 0.353804 | 37.313515 |
| playground | layout_fixed | 32 | 8.602151 | 66.478495 | 0.407014 | 1.024979 | 0.715997 | 31.046985 |
| playground | partial_states | 32 | 6.586022 | 69.448925 | 0.353314 | 1.051343 | 0.702329 | 37.815170 |
| kicker | layout_fixed | 31 | 43.082437 | 92.960573 | 0.076072 | 0.072824 | 0.074448 | 99.042027 |
| kicker | partial_states | 31 | 41.146953 | 93.089606 | 0.067987 | 0.063963 | 0.065975 | 99.342156 |
| office | layout_fixed | 26 | 22.666667 | 77.630769 | 0.090124 | 0.087050 | 0.088587 | 98.818262 |
| office | partial_states | 26 | 24.102564 | 77.374359 | 0.091304 | 0.089944 | 0.090624 | 98.580540 |
| relief_2 | layout_fixed | 20 | 4.385965 | 48.561404 | 0.461796 | 0.474723 | 0.468259 | 40.978045 |
| relief_2 | partial_states | 20 | 4.385965 | 48.877193 | 0.414076 | 0.545535 | 0.479805 | 31.632754 |
| relief | layout_fixed | 19 | 10.916179 | 74.756335 | 0.708663 | 0.824198 | 0.766431 | 1.396420 |
| relief | partial_states | 19 | 17.153996 | 76.588694 | 0.670710 | 0.700914 | 0.685812 | 3.424220 |
| pipes | layout_fixed | 14 | 19.047619 | 84.615385 | 0.190343 | 0.302177 | 0.246260 | 78.470359 |
| pipes | partial_states | 14 | 17.948718 | 86.043956 | 0.305654 | 0.242755 | 0.274205 | 71.880185 |

## 协议、预算与运行记录

重建阈值0.25m，下采样/voxel 0.0390625m，sdf_trunc 0.2m，mesh采样1000000点。沿用仓库unposed Umeyama/RANSAC scale+pose alignment，random_state42、既有mask与深度处理。

逐层保留率上限为[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]，均值0.1600；fine cluster fraction=0.12。这是预算上限，实际密度另存layers.json和下表。该方法不使用SAF3R Top-K/profile。

仅使用GPU4/5/6；动态场景队列与CPU重建评分并行。ETH3D每场景0暖机、1次完整前向；以下为冷前向记录，包含JIT/缓存建立，不用于速度比较。ETH3D相对Dense暖机加速比全部为N/A，原因是没有该数据集的成对暖机计时。

| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |
|---|---:|---:|---:|---:|
| facade | 6 | 13.084978 | 11.1064 | 0.160168 |
| terrains | 4 | 8.945694 | 9.6799 | 0.160442 |
| delivery_area | 5 | 9.055277 | 9.6046 | 0.160389 |
| electro | 5 | 9.559228 | 9.5490 | 0.160310 |
| courtyard | 4 | 6.490963 | 9.5064 | 0.160393 |
| playground | 6 | 7.015770 | 9.3343 | 0.160450 |
| kicker | 4 | 9.324564 | 9.2742 | 0.160386 |
| office | 5 | 9.615443 | 9.0783 | 0.160428 |
| relief_2 | 6 | 8.713913 | 8.8477 | 0.160497 |
| relief | 4 | 8.226848 | 8.8171 | 0.160494 |
| pipes | 5 | 4.299389 | 8.6116 | 0.160463 |

## 既有7Scenes速度记录（本轮未重测）

Previously measured phase2 batch, same frozen candidate source/config;7Scenes heads128 only, GPU6 RTX4090,1 warmup+3 repeats, warmed complete-forward median. This ETH3D batch does not rerun timings or supply an all7Scenes timing macro.

| Method | Warmed median s | Dense speedup | Time / layout-fixed |
|---|---:|---:|---:|
| states_only_fast | 9.705677 | 1.7537x | 1.0422 |
| layout_fixed | 9.312878 | 1.8277x | 1.0000 |
| dense | 17.021049 | 1.0000x | 1.8277 |
| saf3r_top4_lazy_correspondence | 8.581854 | 1.9834x | 0.9215 |

SAF3R对照为原版Top-4、lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO特征Top-K或Top-128。速度对照只有计时，六项几何质量为N/A；未混用论文公开分数或H100计时。

本候选此前比布局修复版慢约4.22%，不满足“不牺牲速度”；完整质量评测不会自动替换当前方案。重复前向存在数值波动，单次质量差异不等于统计显著增益。所有原始六项指标、输入清单、源码快照、各场景配置和GPU监控均保存。
