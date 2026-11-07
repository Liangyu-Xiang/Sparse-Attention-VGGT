# ETH3D：旧 logit 居中收缩公式完整质量评估

固定公式：历史实际执行边的 delta = 上一层真实 logit − 上一层对应块均值 logit。当前历史边 logit = 当前块均值 + 0.5 × (delta − 上一层同块实际观测支持上的 delta 均值)。非历史边只采用当前块均值。逐 head 重建 softmax 后平均全部16头与4个普通Query probe，用于状态损失。历史缓存只含紧邻上一层实际 sparse mask 边，不保存未执行边估计，不滚动继承旧值；没有当前层选边前细粒度QK校验。正式稀疏AV计算当前所选边，并导出已有logit作为下一层证据。特殊token排除于普通估计，正式AV仍保留原特殊token处理。

Query group256；Key配置目标1024，ETH3D实际925/块，每帧一块。保持四状态、Value聚类、10%/18%/10%层段保留率及sqrt(population residual × estimated prior residual)补偿。不增加新操作，不按本次ETH3D分数调整模型。

全部11场景，精确输入帧14–76、同一checkpoint SHA、350×518预处理、pose和recon_unposed，和既有快速版输入及评估代码SHA逐项匹配。快速版是20261106本地历史结果，不是本轮重跑或论文值。两者除历史建模外，还存在快速版当前4probe全Key细粒度概率与本方法粗块估计的区别，因此不是只改变一个系数的消融。

## 等场景宏平均

| method | scene | AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| fast_historical_20261106 | macro11 | 25.8581 | 80.1708 | 0.3192 | 0.8815 | 0.6004 | 57.3814 |
| logit_centered050_new | macro11 | 20.9442 | 77.3498 | 0.3818 | 0.9567 | 0.6693 | 50.8996 |

AUC与F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，未平方。完整逐场景表见six_metrics.md和per_scene_quality.csv，原始任务指标在各run/result.json。

重建阈值0.25m；下采样/TSDF voxel0.0390625m；sdf_trunc0.2m；mesh采样1000000点；原unposed Umeyama/RANSAC尺度及位姿对齐、random_state42，原深度最近邻上采样及GT有效mask。两方法均非SAF3R，Top-K/profile=N/A。

## 时间和显存

每场景1次冷/预热、3次热完整前向中位数，包括初始化、聚类、路由、粗估计、历史导出更新、重排和预测头。CPU质量评估不计入完整前向。11场景热前向中位数等权均值为1.5869s，逐场景时间/峰值见runtime_memory.csv。GPU4/5/6每卡单任务，有全程独占监测。

相对原始VGGT Dense端到端加速比=N/A：没有同期同硬件配对热完整前向，不为计时重跑未修改Dense。Dense基准定义1.00x。历史快速版时间不能用于本轮严格速度比较。没有7Scenes128帧速度验收，也没有混用ScanNet局部注意力输出误差。

## 结论

本公式11场景宏平均六项指标均弱于快速版：AUC@3下降4.9139个百分点，AUC@30下降2.8210个百分点，ACC增加0.0626m，Comp增加0.0752m，CD增加0.0689m，F1下降6.4818个百分点。局部ScanNet中间层输出收益不足以证明ETH3D完整几何质量提升，本轮未达到“不弱于快速版”的要求。不据本次测试真值调整参数，也不把差异单独归因于logit转移。
