# Logit传递与状态损失优化：ETH3D质量验收

本轮候选把4probe logit残差替换为整个普通Querygroup的均值rawlogit残差，由正式AV对已经算出的QK累加导出，不重复QK。当前块均值由相同普通Querygroup均值Q计算。历史残差居中乘0.5，再减0.25倍修正后残差在已观测块支持上的方差，校正指数偏差；未知边只用当前粗块。仅上一层实际支持，L18..23不使用历史logit，只用当前coarse；因此rawlogit只在L0..16导出，L17消费L16后清缓存。IDs仍只取上一层实际mask。原四状态、全部状态损失、Value聚类、10/18/10retention、sqrt补偿均不变。没有当前fineQK验证、anchors或伪历史。缓存Query维4降1；多一次当前Q均值池化及正式AV已有rawlogit的行归约。配置和源码在首个质量分数前冻结，未用ETH3D几何真值选择参数。

全部11ETH3D场景pose+recon_unposed；新候选完整实测，其余两个为原本地历史结果复用，非本轮重跑。输入帧、checkpoint SHA、预处理350x518、现有评估协议保持一致。Querygroup256，Keytarget1024/actual925每帧一块。阈值.25m，downsample/TSDFvoxel.0390625m，sdf_trunc.2m，mesh1000000，原unposed Umeyama/RANSAC对齐random_state42。非SAF3R，TopK/profile=N/A。

| method | scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fast_historical | macro11 | 25.8581 | 80.1708 | 0.3192 | 0.8815 | 0.6004 | 57.3814 |
| history_before18_new | macro11 | 22.0402 | 77.8902 | 0.3344 | 0.8962 | 0.6153 | 55.5738 |
| old_logit_historical | macro11 | 20.9442 | 77.3498 | 0.3818 | 0.9567 | 0.6693 | 50.8996 |

AUC/F1百分数，距离米，CD=(ACC+Comp)/2 unsquared。逐场景原始六指标见各run/result.json，表见quality_per_scene.csv。11场景等权宏平均，不按帧数加权。

质量门槛：相对冻结快速版六项宏平均均不得退步。结果：False。具体各指标判定保存在quality_decision.json。局部ScanNetRGB输出改善不能替代此完整验收。

每场景1预热+3热完整前向中位数，包含初始化、聚类、粗块估计、路由、缓存更新、重排、特殊Query和预测头；不含CPU重建。11场景中位数时间等权平均1.2845s。GPU4/5/6均有独占monitor。原始Dense端到端加速比N/A，无同期配对暖机完整前向；Dense定义1.00x。快速版历史时间不构成同期速度比较。没有重跑未修改的质量控制，也没有将筛选含Dense诊断的耗时当作正式速度。7Scenes128速度尚未验收。

| method | scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| history_before18_new | courtyard | 22.8544 | 86.2352 | 0.3136 | 0.2204 | 0.2670 | 53.9720 |
| old_logit_historical | courtyard | 25.6994 | 85.3912 | 0.3438 | 0.2324 | 0.2881 | 43.4621 |
| fast_historical | courtyard | 28.3073 | 86.8089 | 0.2839 | 0.2106 | 0.2473 | 53.8088 |
| history_before18_new | delivery_area | 23.0342 | 87.2949 | 0.5793 | 1.3176 | 0.9484 | 22.4522 |
| old_logit_historical | delivery_area | 20.1709 | 84.0598 | 0.7698 | 1.5955 | 1.1827 | 9.4087 |
| fast_historical | delivery_area | 24.1026 | 86.0427 | 0.6984 | 1.5226 | 1.1105 | 10.4534 |
| history_before18_new | electro | 31.0391 | 91.6599 | 0.1939 | 5.0920 | 2.6430 | 64.2019 |
| old_logit_historical | electro | 34.3230 | 92.1907 | 0.1747 | 5.0664 | 2.6205 | 64.9674 |
| fast_historical | electro | 42.6451 | 93.4593 | 0.1706 | 5.0771 | 2.6239 | 65.0239 |
| history_before18_new | facade | 31.9649 | 89.7743 | 0.3228 | 0.2880 | 0.3054 | 55.7818 |
| old_logit_historical | facade | 27.5205 | 89.3860 | 0.3031 | 0.2932 | 0.2981 | 59.0380 |
| fast_historical | facade | 36.2339 | 90.2164 | 0.2496 | 0.2302 | 0.2399 | 65.5395 |
| history_before18_new | kicker | 42.8674 | 93.3190 | 0.0558 | 0.0499 | 0.0528 | 99.5177 |
| old_logit_historical | kicker | 41.5054 | 93.1756 | 0.0585 | 0.0537 | 0.0561 | 99.4572 |
| fast_historical | kicker | 49.2473 | 94.1362 | 0.0607 | 0.0532 | 0.0569 | 99.3076 |
| history_before18_new | office | 23.5897 | 76.3692 | 0.0598 | 0.0584 | 0.0591 | 99.4970 |
| old_logit_historical | office | 21.3333 | 76.7179 | 0.0783 | 0.0785 | 0.0784 | 99.1775 |
| fast_historical | office | 23.1795 | 77.0051 | 0.0810 | 0.0773 | 0.0792 | 98.9855 |
| history_before18_new | pipes | 24.9084 | 86.8498 | 0.1871 | 0.2053 | 0.1962 | 85.1609 |
| old_logit_historical | pipes | 21.2454 | 86.8132 | 0.2437 | 0.2328 | 0.2382 | 78.3426 |
| fast_historical | pipes | 37.3626 | 91.5385 | 0.1967 | 0.2226 | 0.2097 | 82.1059 |
| history_before18_new | playground | 9.1398 | 69.0524 | 0.3854 | 0.7986 | 0.5920 | 60.3766 |
| old_logit_historical | playground | 6.8548 | 69.2339 | 0.4274 | 0.9122 | 0.6698 | 41.7892 |
| fast_historical | playground | 10.7527 | 68.9718 | 0.3685 | 0.8691 | 0.6188 | 43.5573 |
| history_before18_new | relief | 19.8830 | 78.1481 | 0.6981 | 0.7909 | 0.7445 | 5.0877 |
| old_logit_historical | relief | 17.7388 | 78.6940 | 0.6799 | 0.7094 | 0.6947 | 5.0476 |
| fast_historical | relief | 17.3489 | 79.4347 | 0.6927 | 0.7040 | 0.6984 | 10.2664 |
| history_before18_new | relief_2 | 5.2632 | 45.8246 | 0.4888 | 0.6228 | 0.5558 | 18.0366 |
| old_logit_historical | relief_2 | 5.4386 | 44.7018 | 0.7098 | 1.0379 | 0.8739 | 9.7564 |
| fast_historical | relief_2 | 6.3158 | 54.8070 | 0.3088 | 0.3853 | 0.3471 | 51.7054 |
| history_before18_new | terrains | 7.8978 | 52.2648 | 0.3934 | 0.4145 | 0.4039 | 47.2279 |
| old_logit_historical | terrains | 8.5559 | 50.4839 | 0.4109 | 0.3122 | 0.3616 | 49.4495 |
| fast_historical | terrains | 8.9431 | 59.4580 | 0.4006 | 0.3443 | 0.3724 | 50.4418 |
