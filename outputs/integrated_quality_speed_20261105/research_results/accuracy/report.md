# 状态损失修复：两种 History 语义的实测对照

**本轮没有得到可替换原方案的合格修复版。** 状态损失与执行语义已统一，但完整重建质量仍有回退，耗时也有增加。A/B均保留为实验对照，不提升为默认。

分支 `codex/shared-value-state-loss-search-20261105`；原默认配置保持不变。公式和34候选解释见 `docs/shared_value_state_losses.md`，固定特征筛选见 `loss_report/report.md`。

A：普通 History 权重1，剩余成员由显式 Rep 承担。B：History 集合共同代表整簇，使用 π/保留估计质量校准。两者均使用中心化 Drop、输出空间8维投影和簇内均匀未知概率。
保持原 Query-256 历史采样、每32 Query共享支持集、20%总Key上限（含Special）、原聚类与16轮预算求解。没有强制History配额；缺少历史信息的当前成员不计算QK用于选mask。

## 中间层输出诊断（层索引10–16，从0开始）

Kicker 原/旧修复/A/B使用完全相同的帧、预处理和91个查询（31 camera+60普通探针）；JSON checks核验。Playground为32帧、92个查询。指标均经过Wo、在LayerScale/残差之前；局部对比共用当前稀疏QKV，累计对比分别走完整稀疏/Dense轨迹。
下表 RelL2 为百分数，余弦为无量纲数；累计平方误差与Dense能量先求和再开方。

| 方法 | 场景 | 比较 | 查询 | RelL2 % | 余弦 |
|---|---|---|---|---:|---:|
| 原方案 | kicker | local_same_QKV | camera | 32.64 | 0.93742 |
| 原方案 | kicker | local_same_QKV | probe_non_camera | 21.19 | 0.96344 |
| 原方案 | kicker | independent_Dense_trajectory | camera | 35.48 | 0.93497 |
| 原方案 | kicker | independent_Dense_trajectory | probe_non_camera | 28.28 | 0.94486 |
| 上一修复版 | kicker | local_same_QKV | camera | 9.37 | 0.99602 |
| 上一修复版 | kicker | local_same_QKV | probe_non_camera | 14.21 | 0.98800 |
| 上一修复版 | kicker | independent_Dense_trajectory | camera | 16.33 | 0.98860 |
| 上一修复版 | kicker | independent_Dense_trajectory | probe_non_camera | 20.97 | 0.97381 |
| A_history_self_residual_rep | kicker | local_same_QKV | camera | 9.45 | 0.99551 |
| A_history_self_residual_rep | kicker | local_same_QKV | probe_non_camera | 12.98 | 0.99003 |
| A_history_self_residual_rep | kicker | independent_Dense_trajectory | camera | 17.44 | 0.98698 |
| A_history_self_residual_rep | kicker | independent_Dense_trajectory | probe_non_camera | 20.11 | 0.97657 |
| B_history_group | kicker | local_same_QKV | camera | 10.80 | 0.99426 |
| B_history_group | kicker | local_same_QKV | probe_non_camera | 15.04 | 0.98712 |
| B_history_group | kicker | independent_Dense_trajectory | camera | 17.67 | 0.98610 |
| B_history_group | kicker | independent_Dense_trajectory | probe_non_camera | 22.10 | 0.97256 |
| A_history_self_residual_rep | playground | local_same_QKV | camera | 7.37 | 0.99743 |
| A_history_self_residual_rep | playground | local_same_QKV | probe_non_camera | 12.61 | 0.99104 |
| A_history_self_residual_rep | playground | independent_Dense_trajectory | camera | 12.38 | 0.99382 |
| A_history_self_residual_rep | playground | independent_Dense_trajectory | probe_non_camera | 18.57 | 0.98232 |
| B_history_group | playground | local_same_QKV | camera | 7.36 | 0.99733 |
| B_history_group | playground | local_same_QKV | probe_non_camera | 12.63 | 0.99068 |
| B_history_group | playground | independent_Dense_trajectory | camera | 11.65 | 0.99437 |
| B_history_group | playground | independent_Dense_trajectory | probe_non_camera | 18.13 | 0.98297 |

24层完整Markdown表见 [layer_details.md](layer_details.md)，包括各状态比例、实际历史Key复用、普通/camera局部及累计余弦/相对L2。CSV见 `layer_states.csv` 和 `layer_similarities.csv`。
逐层图：[layer_comparison.png](layer_comparison.png)。诊断不等于几何质量评测，六项质量均为N/A，原因记录在各运行quality.md。

## 六项质量验收

AUC/F1列为百分数，JSON/CSV为fraction；ACC/Comp/CD单位米，CD=(ACC+Comp)/2、非平方距离。7Scenes阈值0.05m/下采样0.0078125m；ETH3D阈值0.25m/下采样0.0390625m。
均为相同checkpoint文件名、精确输入帧列表和预处理、pose+recon_unposed、相同对齐/融合/阈值。历史Dense无checkpoint内容摘要，不能宣称已核验旧服务器权重逐字节一致。SAF3R Top-K/profile=N/A，这些运行没有安装SAF3R补丁。
全部候选在新几何质量评测之前按Dense内部输出诊断冻结，见quality/frozen_manifest.json；不按7Scenes测试真值再调损失。七场景未齐时不输出7Scenes宏分数，ETH3D仅3场景子集。
Dense和上一修复版为既有本地测量参考；Pumpkin选择精确匹配本机773个有效RGB/pose配对所采128帧的既有Dense记录，未混用从1000帧采样的旧控制。

| 范围 | 方法 | 完成场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 7Scenes full7 | 历史本地Dense | 7/7 | 23.1100 | 84.2975 | 0.1478 | 0.1239 | 0.1359 | 48.2009 |
| 7Scenes full7 | 上一修复版 | 7/7 | 21.9594 | 83.5954 | 0.1573 | 0.1296 | 0.1434 | 46.7284 |
| 7Scenes full7 | A_history_self_residual_rep | 7/7 | 21.3266 | 83.7553 | 0.1513 | 0.1247 | 0.1380 | 46.2226 |
| 7Scenes full7 | B_history_group | 7/7 | 20.8029 | 83.5868 | 0.1674 | 0.1312 | 0.1493 | 43.7090 |
| ETH3D subset3 | 历史本地Dense | 3/3 | 36.3852 | 87.1533 | 0.2206 | 0.2927 | 0.2567 | 77.1050 |
| ETH3D subset3 | 上一修复版 | 3/3 | 23.2881 | 81.3490 | 0.2616 | 0.4071 | 0.3344 | 67.0714 |
| ETH3D subset3 | A_history_self_residual_rep | 3/3 | 26.8730 | 81.9460 | 0.3026 | 0.4624 | 0.3825 | 57.8252 |
| ETH3D subset3 | B_history_group | 3/3 | 23.9191 | 81.5087 | 0.2582 | 0.4139 | 0.3361 | 66.0417 |

| 数据集 | 场景 | 方法 | 帧 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7scenes | heads | 历史本地Dense | 128 | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 |
| 7scenes | heads | 上一修复版 | 128 | 53.0348 | 94.1478 | 0.0348 | 0.0313 | 0.0330 | 84.1335 |
| 7scenes | heads | A_history_self_residual_rep | 128 | 51.3985 | 93.9173 | 0.0339 | 0.0316 | 0.0328 | 84.4911 |
| 7scenes | heads | B_history_group | 128 | 47.1211 | 93.3342 | 0.0354 | 0.0310 | 0.0332 | 83.9068 |
| 7scenes | chess | 历史本地Dense | 128 | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 |
| 7scenes | chess | 上一修复版 | 128 | 17.9626 | 87.6444 | 0.1473 | 0.0888 | 0.1181 | 34.8750 |
| 7scenes | chess | A_history_self_residual_rep | 128 | 14.6900 | 86.8303 | 0.1483 | 0.0881 | 0.1182 | 35.0502 |
| 7scenes | chess | B_history_group | 128 | 14.0215 | 86.5461 | 0.1533 | 0.0904 | 0.1219 | 34.3361 |
| 7scenes | stairs | 历史本地Dense | 128 | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 |
| 7scenes | stairs | 上一修复版 | 128 | 4.6834 | 80.1780 | 0.2190 | 0.4770 | 0.3480 | 15.2041 |
| 7scenes | stairs | A_history_self_residual_rep | 128 | 7.0169 | 81.7089 | 0.2285 | 0.4826 | 0.3556 | 14.8238 |
| 7scenes | stairs | B_history_group | 128 | 9.3217 | 81.7532 | 0.2263 | 0.4616 | 0.3440 | 15.0042 |
| 7scenes | fire | 历史本地Dense | 128 | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 |
| 7scenes | fire | 上一修复版 | 128 | 52.2187 | 94.1617 | 0.0455 | 0.0488 | 0.0471 | 68.4935 |
| 7scenes | fire | A_history_self_residual_rep | 128 | 50.6111 | 93.9825 | 0.0501 | 0.0514 | 0.0508 | 61.7879 |
| 7scenes | fire | B_history_group | 128 | 50.4101 | 93.9104 | 0.0549 | 0.0528 | 0.0538 | 54.6458 |
| 7scenes | office | 历史本地Dense | 128 | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 |
| 7scenes | office | 上一修复版 | 128 | 12.3524 | 75.6004 | 0.0518 | 0.0557 | 0.0537 | 58.7722 |
| 7scenes | office | A_history_self_residual_rep | 128 | 11.7946 | 75.3982 | 0.0517 | 0.0541 | 0.0529 | 55.0616 |
| 7scenes | office | B_history_group | 128 | 11.2041 | 75.3375 | 0.0535 | 0.0556 | 0.0546 | 53.1699 |
| 7scenes | pumpkin | 历史本地Dense | 128 | 6.7052 | 75.2083 | 0.4639 | 0.0847 | 0.2743 | 26.3906 |
| 7scenes | pumpkin | 上一修复版 | 128 | 5.9916 | 75.0185 | 0.5194 | 0.1184 | 0.3189 | 22.4485 |
| 7scenes | pumpkin | A_history_self_residual_rep | 128 | 6.0983 | 75.0164 | 0.4660 | 0.0838 | 0.2749 | 27.1220 |
| 7scenes | pumpkin | B_history_group | 128 | 6.1065 | 74.8450 | 0.5699 | 0.1473 | 0.3586 | 18.1071 |
| 7scenes | redkitchen | 历史本地Dense | 128 | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 |
| 7scenes | redkitchen | 上一修复版 | 128 | 7.4721 | 78.4174 | 0.0833 | 0.0870 | 0.0851 | 43.1718 |
| 7scenes | redkitchen | A_history_self_residual_rep | 128 | 7.6772 | 79.4332 | 0.0802 | 0.0816 | 0.0809 | 45.2215 |
| 7scenes | redkitchen | B_history_group | 128 | 7.4352 | 79.3812 | 0.0784 | 0.0797 | 0.0791 | 46.7931 |
| eth3d | kicker | 历史本地Dense | 31 | 57.9928 | 95.5412 | 0.0507 | 0.0462 | 0.0484 | 99.6216 |
| eth3d | kicker | 上一修复版 | 31 | 44.5878 | 93.3548 | 0.0636 | 0.0587 | 0.0612 | 99.4419 |
| eth3d | kicker | A_history_self_residual_rep | 31 | 47.3835 | 93.3190 | 0.0632 | 0.0548 | 0.0590 | 99.3606 |
| eth3d | kicker | B_history_group | 31 | 44.9462 | 92.9534 | 0.0625 | 0.0549 | 0.0587 | 99.0873 |
| eth3d | playground | 历史本地Dense | 32 | 18.6828 | 77.2177 | 0.3403 | 0.6461 | 0.4932 | 73.5691 |
| eth3d | playground | 上一修复版 | 32 | 8.0645 | 68.1317 | 0.4631 | 1.0083 | 0.7357 | 31.5467 |
| eth3d | playground | A_history_self_residual_rep | 32 | 10.6183 | 67.0094 | 0.4369 | 1.0832 | 0.7600 | 30.4342 |
| eth3d | playground | B_history_group | 32 | 11.0215 | 67.6277 | 0.4629 | 1.0200 | 0.7414 | 28.7934 |
| eth3d | courtyard | 历史本地Dense | 38 | 32.4798 | 88.7008 | 0.2707 | 0.1859 | 0.2283 | 58.1241 |
| eth3d | courtyard | 上一修复版 | 38 | 17.2119 | 82.5605 | 0.2582 | 0.1543 | 0.2063 | 70.2254 |
| eth3d | courtyard | A_history_self_residual_rep | 38 | 22.6174 | 85.5097 | 0.4076 | 0.2493 | 0.3285 | 43.6808 |
| eth3d | courtyard | B_history_group | 38 | 15.7895 | 83.9450 | 0.2493 | 0.1669 | 0.2081 | 70.2445 |

## 耗时与显存

heads128同一GPU、同一套模型权重，原Shared Value/新方法交替，各预热2次+正式3次，包含完整forward的初始化、聚类、路由、历史和预测头。权重/图像加载及wrapper切换不在forward内。
各配对仅在本配对内计算增幅；两种候选位于不同物理GPU，不能横比绝对秒数。两套Graph缓存共存，独立显存见quality_per_scene.csv。质量运行的单次时间只记录，不用作跨批配对比较。

| 方法 | 原Shared Value s | 新方法 s | 耗时增幅 % | 对原Shared Value加速 | 对Dense加速 |
|---|---:|---:|---:|---:|---|
| A_history_self_residual_rep | 16.8189 | 19.3590 | +15.10 | 0.869x | N/A |
| B_history_group | 17.2217 | 18.8320 | +9.35 | 0.914x | N/A |
| A_history_self_residual_rep / sparse sums experiment | 17.3046 | 19.8226 | +14.55 | 0.873x | N/A |
| B_history_group / sparse sums experiment | 17.3467 | 20.3165 | +17.12 | 0.854x | N/A |

对原始VGGT Dense的端到端加速比N/A：按项目要求复用未变更Dense，不为计时重跑；没有同输入/同硬件/同协议的配对预热时间。Dense自身按定义1.00x。
因其他作业抢占GPU而被监控中止的尝试单独保留，不进入有效质量或计时汇总；见failed_attempts.json，六项均N/A并列原因。

额外尝试了“预计算簇Value和/模长和＋稀疏历史修正”的代数等价统计优化，CPU/CUDA公式核验通过，但配对计时未观察到稳定收益，未默认启用。它改变浮点求和次序，仅完成内核/计时诊断，没有用初版质量分数冒充优化版质量评测。

## 联合输出损失补充实验

针对跨簇误差及归一化问题，已实现直接优化联合输出误差的离线坐标更新，每次最多32个状态变更；以B的可分中心化解为起点，比较γ=0/0.01/0.1的可分能量正则。它能修正统一概率反例，但真实输出仍需核验。
以下只改变联合目标及其正则，保持本组控制的输入特征/历史/概率/语义相同。预计目标在8维输出投影中计算；Dense误差在完整Wo输出中测量。after/before为相同γ的目标值之比，不能跨γ当作同一种MSE比较。
原型是CPU编排的诊断，尚未完成生产内核、独立轨迹及几何质量评估；六项质量及相对Dense速度均N/A，原因见joint_report各候选quality.md。

| 场景 | 候选 | 普通查询RelL2 % | Camera RelL2 % | 预计目标after/before |
|---|---|---:|---:|---:|
| joint_kicker | joint/control_separable | 13.020 | 10.913 | N/A |
| joint_kicker | joint/gamma0 | 13.634 | 11.654 | 0.18249 |
| joint_kicker | joint/gamma001 | 13.383 | 11.116 | 0.87368 |
| joint_kicker | joint/gamma01 | 12.977 | 10.921 | 0.99268 |
| joint_playground | joint/control_separable | 12.170 | 8.837 | N/A |
| joint_playground | joint/gamma0 | 12.574 | 9.147 | 0.13795 |
| joint_playground | joint/gamma001 | 12.423 | 8.955 | 0.89558 |
| joint_playground | joint/gamma01 | 12.177 | 8.803 | 0.99380 |

## 尚未解决的限制

新增的统一概率反例确认：即使簇内均匀概率完全正确，可分簇损失相加仍会选错mask，因为它遗漏簇间误差交叉项及整张mask的归一化分母。相同2个patch Key预算，Rep+Rep的真实误差为0，Exact+Drop的真实误差为0.8，可分能量和可分矩损失却都偏好后者。详见 `docs/shared_value_state_losses.md` 与 `separable_counterexample.json`。这证明结构缺陷存在，但未测出它对数据集退化的贡献比例。
修正的是状态损失与执行语义之间的可避免不一致；未知QK的簇内均匀、head共享mask、8维输出投影和跨层误差累积仍是近似。不能仅凭History比例或局部误差下降宣称重建质量恢复，更不能宣称每层等同Dense。
`rep_m5c_relaxed`配置及实现仍未出现在已同步的可见远程提交树中，本次没有伪造与它的mask重合率。
所有24层原始数据、各场景六项指标、源快照、输入列表、GPU独占监控都保留于本输出目录及quality子目录。
