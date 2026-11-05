# 全部特殊token双向保护：完整ETH3D质量与7Scenes速度

11/11个既有评测场景，共377帧；本批全部重新推理和评分。每场景上限128帧，uniform seed0；既有协议排除terrace/meadow，本轮不新增过滤。对照质量复用完整布局修复版结果。

唯一方法变更：dense_special_queries从1改为5。每帧1个camera和4个register token始终作为Key保留，且现在作为Query均对全部Key计算。普通Query仍对全部特殊Key与原方法选出的patch Key计算。保留率、聚类、损失及权重均未调整；未重新启用已淘汰的partial states。

此实现先运行稀疏observer，再以完整SDPA覆盖所有特殊Query输出；历史流仍来自稀疏observer。它保证最终输出的双向支持关系，但不是底层kernel物理剔除特殊Query；重复计算完整计入速度。

AUC/F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离。以下为11场景等权平均。

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| layout_fixed_camera_dense | 18.510638 | 76.749655 | 0.323066 | 0.871509 | 0.597288 | 56.077724 |
| all_specials_dense | 19.916624 | 77.705039 | 0.346437 | 0.895727 | 0.621082 | 58.399991 |

相对布局修复版：AUC@3 +1.4060个百分点，AUC@30 +0.9554个百分点，F1 +2.3223个百分点；CD +0.023794m。

## 同批完整前向计时

7Scenes heads128 only, not all7-scene macro; GPU6 RTX4090,1 warmup+3 repeats, median complete forward including reset/clustering/routing/history/special-query dense correction and SAF3R Top-K reset

| Method | Median s | Dense speedup | Time / layout-fixed | Time / SAF3R | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| all_specials_dense | 9.386872 | 1.8186x | 1.0110 | 1.0925 | 15.3735 |
| layout_fixed | 9.285144 | 1.8385x | 1.0000 | 1.0806 | 15.3730 |
| dense | 17.070703 | 1.0000x | 1.8385 | 1.9867 | 14.3586 |
| saf3r | 8.592450 | 1.9867x | 0.9254 | 1.0000 | 14.6143 |

满足不牺牲布局修复版速度：False；满足SAF3R耗时上浮不超过10%：True。

首组计时保存在speed_heads128及report_initial_timing.md；当时CPU质量评分并行，SAF3R样本8.388–9.460秒，不能据其中位数直接宣称稳定速度优势。补测在全部CPU质量任务结束后执行，顺序为布局修复版→新版本→Dense→SAF3R，仍为同GPU、1暖机+3完整前向，未修改任何方法参数。若补测已完成，本报告主表使用该组；首组结果完整保留，没有删除或混合样本。

补测未观察到CPU质量评分进程。计时仍只有每种方法3个暖机后样本，表中速度结论仅代表该输入与本批测量；不将首组的较快中位数当作已经证实的稳定提速。两组结果均保留，未据结果重新调参。

SAF3R为原版Top-4、lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO Top-K或Top-128。四组计时只有RGB完整前向，六项几何质量为N/A；未混用论文分数和H100时间。控制方法仅计时重跑，依据用户此前明确授权。

## 逐场景六项质量

| Scene | Method | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| facade | layout_fixed_camera_dense | 76 | 24.421053 | 87.207018 | 0.266299 | 0.260766 | 0.263533 | 62.902088 |
| facade | all_specials_dense | 76 | 24.888889 | 87.995322 | 0.286881 | 0.291333 | 0.289107 | 64.855309 |
| terrains | layout_fixed_camera_dense | 42 | 7.394503 | 57.417731 | 0.306701 | 0.272322 | 0.289511 | 61.442925 |
| terrains | all_specials_dense | 42 | 8.207511 | 54.595432 | 0.348048 | 0.213920 | 0.280984 | 71.869034 |
| delivery_area | layout_fixed_camera_dense | 40 | 13.888889 | 79.816239 | 0.556683 | 1.431668 | 0.994175 | 18.517464 |
| delivery_area | all_specials_dense | 40 | 15.000000 | 80.192308 | 0.752512 | 1.647107 | 1.199810 | 12.877058 |
| electro | layout_fixed_camera_dense | 39 | 32.568601 | 91.421502 | 0.161282 | 4.637558 | 2.399420 | 67.550000 |
| electro | all_specials_dense | 39 | 36.212326 | 92.118758 | 0.201345 | 4.681717 | 2.441531 | 65.018391 |
| courtyard | layout_fixed_camera_dense | 38 | 16.642959 | 83.380749 | 0.328753 | 0.198338 | 0.263545 | 56.690394 |
| courtyard | all_specials_dense | 38 | 22.332859 | 85.860597 | 0.166794 | 0.148620 | 0.157707 | 82.860933 |
| playground | layout_fixed_camera_dense | 32 | 8.602151 | 66.478495 | 0.407014 | 1.024979 | 0.715997 | 31.046985 |
| playground | all_specials_dense | 32 | 8.467742 | 67.762097 | 0.418799 | 0.973745 | 0.696272 | 36.035107 |
| kicker | layout_fixed_camera_dense | 31 | 43.082437 | 92.960573 | 0.076072 | 0.072824 | 0.074448 | 99.042027 |
| kicker | all_specials_dense | 31 | 47.383513 | 93.634409 | 0.067240 | 0.062055 | 0.064648 | 98.866890 |
| office | layout_fixed_camera_dense | 26 | 22.666667 | 77.630769 | 0.090124 | 0.087050 | 0.088587 | 98.818262 |
| office | all_specials_dense | 26 | 19.384615 | 75.764103 | 0.078769 | 0.074540 | 0.076655 | 98.819248 |
| relief_2 | layout_fixed_camera_dense | 20 | 4.385965 | 48.561404 | 0.461796 | 0.474723 | 0.468259 | 40.978045 |
| relief_2 | all_specials_dense | 20 | 4.210526 | 53.035088 | 0.449880 | 0.604408 | 0.527144 | 33.169852 |
| relief | layout_fixed_camera_dense | 19 | 10.916179 | 74.756335 | 0.708663 | 0.824198 | 0.766431 | 1.396420 |
| relief | all_specials_dense | 19 | 9.551657 | 75.555556 | 0.820907 | 0.936935 | 0.878921 | 0.488493 |
| pipes | layout_fixed_camera_dense | 14 | 19.047619 | 84.615385 | 0.190343 | 0.302177 | 0.246260 | 78.470359 |
| pipes | all_specials_dense | 14 | 23.443223 | 88.241758 | 0.219633 | 0.218619 | 0.219126 | 77.539582 |

## 协议与运行记录

重建阈值0.25m，下采样/voxel 0.0390625m，sdf_trunc 0.2m，mesh采样1000000点。沿用unposed Umeyama/RANSAC scale+pose对齐、random_state42、原mask和深度处理。

逐层patch保留预算为[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]，均值0.1600；fine cluster fraction=0.12。实际密度另存layers.json与下表；全量特殊Query会增加实际计算覆盖率，不伪装为相同等效密度。

仅GPU4/5/6；ETH3D每场景0暖机、1次完整前向。以下为包含JIT/缓存建立的冷前向记录，不用于速度比较；ETH3D Dense暖机加速为N/A，因为只在7Scenes测速度。

| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |
|---|---:|---:|---:|---:|
| facade | 4 | 8.527902 | 11.1641 | 0.163751 |
| terrains | 5 | 6.261964 | 9.7191 | 0.163999 |
| delivery_area | 5 | 6.611095 | 9.6368 | 0.163987 |
| electro | 4 | 6.823340 | 9.5825 | 0.163885 |
| courtyard | 5 | 6.383212 | 9.5356 | 0.164016 |
| playground | 4 | 7.002625 | 9.3591 | 0.164041 |
| kicker | 5 | 6.454321 | 9.2993 | 0.163982 |
| office | 4 | 5.706665 | 9.1042 | 0.164020 |
| relief_2 | 5 | 5.803803 | 8.8559 | 0.164126 |
| relief | 4 | 5.668420 | 8.8247 | 0.164075 |
| pipes | 5 | 4.867638 | 8.6107 | 0.164045 |

所有场景通过相同帧表、权重SHA、预处理、评测协议与源码、冻结配置、GPU范围和隔离审计。每一层均验证5×帧数个特殊Query为全量、全部特殊Key保留，且预算计数无误。

这是单次完整质量评测；既有实现重复前向存在数值波动，不能把小幅差异当成统计显著收益。完整原始指标、源码快照与计时样本均保存；当前报告不自动替换已保留的方法。
