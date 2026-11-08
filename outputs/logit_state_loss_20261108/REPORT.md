# Logit传递与状态损失研究汇总

验收要求保持不变：ETH3D全部11场景六指标等场景宏平均均不弱于冻结快速版；质量通过后再验证7Scenes全部7场景128帧速度不慢于快速版。默认只运行改变的方法，既有快速版/Dense/SAF3R质量均不重跑。不使用7Scenes测试几何真值拟合参数，不加当前层细粒度QK验证，不继承两层以上旧边数值。

## 已完成完整质量结果

| method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| center_drop_l6_new | 20.8644 | 77.7999 | 0.3587 | 0.9070 | 0.6328 | 53.4415 |
| fast_historical | 25.8581 | 80.1708 | 0.3192 | 0.8815 | 0.6004 | 57.3814 |
| old_logit_historical | 20.9442 | 77.3498 | 0.3818 | 0.9567 | 0.6693 | 50.8996 |
| group_mean_new | 23.2609 | 77.9969 | 0.3390 | 0.8947 | 0.6168 | 59.2034 |
| value_group_new | 21.0414 | 76.9608 | 0.3515 | 0.9048 | 0.6281 | 56.6731 |
| group_center_drop_new | 21.5662 | 77.0766 | 0.3274 | 0.9137 | 0.6206 | 57.7319 |
| group_gamma_new | 21.7170 | 76.6943 | 0.3432 | 0.9215 | 0.6323 | 54.7370 |
| history_before18_new | 22.0402 | 77.8902 | 0.3344 | 0.8962 | 0.6153 | 55.5738 |
| marginal_logit_calibrated_new | 20.9046 | 76.8744 | 0.3478 | 0.9323 | 0.6400 | 54.5102 |

AUC/F1百分数，距离米，CD=(ACC+Comp)/2 unsquared。fast/oldlogit行是此前本地完整结果复用；其余各自11场景完整pose+recon_unposed。每个候选的独立门槛判定在*_quality_decision.json/quality_decision.json；未通过不得当默认替代或宣称质量达标。

## 建模与运算边界

1. 状态损失有限比较：权重幂次、cluster总质量界、Value向量合并误差及对角不确定项、History收益置信度、Drop相对Value均值以及补偿/归一化修正。多数RGB局部误差变差，全记录见all_screen_scores.csv。centerDrop只复用当前已有Value特征，不加fineQK。
2. 组均值logit：当前coarse用整个普通Querygroup均值Q和Keyblock均值K，上一层正式AV汇总同一mask所有普通Query的已有rawlogit。只导出/消费紧邻前一层实际执行边。4probe缓存改成组均值1份，AV主输出数值测试完全一致；无第二次选中边QK。修正delta=0.5*居中rawdelta−0.25*Var(0.5居中rawdelta)，方差在上一层同块实际观测支持内计算。
3. 仅用状态损失已计算的Value范数与代表点距离作两项线性logit偏置；ScanNetRGB-only离线拟合，当前层未知Key可有估计偏置，但从不将它写为历史观测。上一层cache仅保存实际rawmean−同层coarse−同层bias。此模型虽然同场景新帧局部输出改善，完整ETH3D未追平组均值基础方案，故不采用它。
4. α全局/层段、头共享残差筛选不加观测，未改已运行候选公式。它们只作为RGB诊断，不能冒用任何完整质量分数。

所有估计和统计排除普通集合外的特殊token，正式AV保持原Special处理；16头平均，不输出逐头结果。Query256；Keytarget1024，ETH3Dactual925/每帧一块。四状态、Value聚类和10/18/10保留率保持。sourceScanNet同一个场景，仅RGB，用于局部建模筛选；ETH3D迭代结果用于工程验收与候选取舍，不声称是从未用于研究的盲测。各参数在自己完整质量运行前冻结，没有按7Scenes测试真值调参。

## 质量、速度、显存和失败

原始六指标/逐场景/等场景宏平均、24层状态、每场景1预热+3热完整前向与峰值见各ETH3D目录和quality_*CSV。正式完整前向包含初始化、聚类、路由、历史、重排、Special和预测头。所有GPU4/5/6每卡单应用monitor无重叠。ETH3D阈值.25m、downsample/voxel.0390625m、sdf_trunc.2m、mesh1000000，原Umeyama/RANSACunposed对齐random_state42；非SAF3R，TopK/profile=N/A。

本轮没有同期Dense配对计时，因此所有方法相对原始Dense加速比=N/A，Dense定义1.00x；历史时间不当同期速度对照。质量尚未通过时不把速度列为已验收，也不以减少保留率换取时间。kernel/局部输出/离线拟合和失败run全部六指标N/A，原因逐项在diagnostic_six_metrics.csv/md。initialkernel测试的参考广播错误、Valuebias脚本名字冲突、旧4probe统计不兼容groupmean的失败均保留，不属于候选算法完整质量失败。修正后的数值测试及11项单元测试通过。

后续还筛选了粗块logit对齐斜率/相关性、保留logit均值的未知回退、头间固定线性映射、补偿质量偏差惩罚、基于完整现有Value的头条件损失。头条件修正最好中间层局部误差16.2358%→16.0135%，但浅层略退步，未达到事前设定的充分改善阈值，没有冒充完整质量结果。线性头映射把加权logitMSE从约4.50降到1.87，却没有稳定改善输出，未晋升。补偿gamma的局部改善同样没有在完整质量中保持。仅去后层历史也未通过。

原始公共QKV大文件继续保留在既有logit_delta_scannet_20261107/capture，不复制；原始每次质量source快照和选择文件保留。round1_source.py是从同一数学代码重建的首轮源版本，不是事前冻结快照；该轮protocol中的源码列表漏列新模型文件，不将事后重建假称运行前SHA证据。部分筛选protocol还漏列新模型模块，最终代码归档也不冒充那些筛选的事前冻结证据。后续完整质量都具有事前sourcehashes和源码归档并已校验。

## 本轮最终状态

7个新增候选×11场景=77个完整pose+recon_unposed场景评估均已完成；74次RGB配置诊断包含重复基线，不算74个独立完整质量方法。没有候选满足快速版六项宏平均全部不退步。当前logit研究参考更新为group_mean_new，因为它的AUC@3、AUC@30、Comp、F1为本轮新候选中最好，并且六项均优于旧centered050；它不是每个指标都优于本轮其他候选（例如ACC/CD有其他候选略好），不将跨方法最优列拼接成一个虚构方法。

具体相对快速版：AUC@3低2.5971个百分点，AUC@30低2.1739个百分点，ACC高0.0197m，Comp高0.0132m，CD高0.0165m；F1高1.8220个百分点。目标尚未完成，不能以单项F1取代六指标门槛。质量先验收，尚未进行7Scenes128速度验收或声称速度达标。

本轮含一个独立目录logit_rgb_calibration_20261108：其5场景RGB注意力校准/6场景未校准结果均完整保留，没有几何真值用于拟合；帧校验入口读取相机元数据已明确更正。该方法同样未通过。不能把校准5场景计作完全未见RGB的盲测。

报告的history_cache_bytes是旧实现的简化计数：它不完整覆盖coarse mean等辅助缓存，删去rawlogit缓存后也不代表上一层IDs不再存储。比较总显存必须使用每run真实Torch峰值。所有方法框架、特殊token处理和保留率未放宽，77次输入checkpoint/evalSHA/预算匹配审计均通过。
