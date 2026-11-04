# v8 历史概率取数方案与五类耗时统计

实施状态：v8 五类耗时诊断已完成；下述简化历史缓存方案处于讨论阶段，尚未实现、测速或评估质量。本文没有改变当前分支的模型实现、状态损失、保留率或默认配置。

## 已测结果

使用冻结的 `efficiency_quality_v8_20261103/source`，而不是把当前分支最新模型当作 v8。7Scenes heads128，24个全局 Attention 层，RTX5090 D GPU7。一次预热后一次带插桩完整前向；CPU 文件锁/监控不建立辅助 CUDA context，最大同时 CUDA 应用数1。

| 部分 | 累计 s | 占稀疏链路 % | 占完整前向 % |
|---|---:|---:|---:|
| 每层 Value 特征、Cluster 划分和更新 | 1.075306 | 6.66 | 5.26 |
| 历史概率估计、生成、归一化、缓存和消费 | 2.976062 | 18.43 | 14.56 |
| 候选资格、状态损失和预算选择 | 2.951354 | 18.28 | 14.44 |
| Mask 构建、质量转移和执行准备 | 2.177158 | 13.48 | 10.65 |
| 稀疏 Attention 执行，含 QKV/Norm/RoPE/输出投影 | 6.966448 | 43.15 | 34.09 |
| 五类稀疏链路合计 | 16.146327 | 100.00 | 79.01 |
| 基础模型及全局模块外开销 | 4.289895 | — | 20.99 |
| 完整前向 | 20.436222 | — | 100.00 |

历史无插桩 warmed median19.981s来自20261103的另一批次，不能用作上表百分比的分母。本次插桩结果不替代该正式中位数；没有重跑 Dense，配对 Dense 加速比 N/A。加载模型、图像、H2D 输入搬运和 GT 评估不在前向计时内。

五类覆盖全局稀疏链路100%。局部 Attention、FFN、输出头、前向初始化及全局模块外开销单列，避免错误归类。完整映射、Graph 子阶段、24层实测时间、计时校验与监控见[原始报告](../../outputs/v8_five_bucket_profile_20261104/report.md)。

## 为什么“只保存上一层概率”与当前实现成本不同

用户希望得到上一层已保留 Token 的稀疏注意力概率，以及粗 Grid 上的概率。少量 Query probe 的这两类信号可以复用已有 Attention 计算，但概率导出仍需归一化、归约和显存写入；融合 Attention 不一定物化整张概率矩阵。

必须明确以下不同定义：

1. **实际稀疏概率**：在正式保留 Key 和 Special 集合上归一化，包含正式 Attention 使用的补偿 bias。记为 `softmax(selected_logits + bias)`。
2. **稀疏概率的 Grid 聚合**：把同一次稀疏 Softmax 的概率按 Grid 求和，不增加 QK，但本身不能给出未选 Token 的真实 Dense 概率。
3. **独立粗 Grid 概率**：用 Query 与 pooled Key/区域大小构造覆盖全图的粗估计，需要少量额外 QK，与稀疏 Softmax 的分母不同。
4. **当前 v8 历史条件概率**：使用无补偿的 Probe Logit，在每个 Grid 中加入未选 Token 的粗估计，重新计算分母，再生成条件概率、head-mean 和压缩缓存。这不等于直接取出的实际稀疏 Softmax 概率。

当前调用链可在归档源码中检查：`reuse_grid_fused_history.save_evidence` → Grid 分母/EPS兼容 → `reuse_grid_local_history.store` → head-mean/压缩缓存；下一层通过 `reuse_headmean_history.build_prior` 消费。

同一次诊断中的历史相关明细：

| 操作 | 累计 s |
|---|---:|
| Grid 分母归一化 kernel | 1.447647 |
| head-mean 与压缩缓存写入 kernel | 0.450126 |
| 当前层历史 prior 估计/消费 | 0.329008 |
| 全部历史相关操作 | 2.976062 |

总计还包括 Coarse 观测、Grid 聚合、EPS兼容、uniform缓存、检查和统计，以上三项不是完整分解。历史总成本不能全部称为“缓存写入”。

典型原始 Probe Logit 为 `[64,16,4,26009]` FP32，约406.39MiB/窗口。写出再读取这种大中间量有显著流量；保存全部 Query 的概率比4个 Probe 更大。简化方案的潜在收益应来自减少中间量和校准步骤，而不是假定概率导出免费。

## 讨论中的简化接口

拟保留：上一层实际 Key IDs、有效位、Probe位置/权重、所选 Key 的少量 Probe 概率，以及**定义明确**的粗 Grid 概率。若在当前层内部先对 head 求平均，必须先完成各 head 自己的归一化，不能把平均 Logit 的 Softmax 当作平均概率。

可考虑在正式 Attention 内复用已算的 Probe Logit/最终 Softmax normalizer，直接导出紧凑历史统计；避免为了保存历史而重新计算完整注意力、物化所有 Query×Key 概率，或保存巨大的逐 head 原始中间量。分区 softmax 的最终合并、Special、补偿 bias、Padding 和尾部重复 Probe 权重均需保持正确。

**这是未来实现方向，不是本次已经获得的加速结果。**

如果下一层只把历史作为路由信号，可以探索直接使用稀疏概率和粗 Grid 信号。如果现有状态损失与质量转移仍要求对所有 Token 给出统一的近似 Dense 概率，就必须定义两种分母之间的融合规则及未选 Token 的质量分配。单纯替换缓存内容会改变概率语义，不能宣称与 v8 等价。必要时额外保存 normalizer 或无补偿证据；先平均 head 后通常不能反演各 head 的归一化信息。

## 后续验证约束

- 先冻结概率定义、接口和实验配置，再实现；不依据7Scenes测试真值调参。
- 保留目标配置原有预算、Cluster 和状态设置。v8 是4-probe；单探针修复版是1-probe，两者不能混用缓存形状或权重。
- 小规模验证 Softmax、Grid质量守恒、Special、补偿、分区合并和缓存读写；随后完整评估 pose 与 recon_unposed。
- 每个改动方法保存 AUC@3/AUC@30/ACC/Comp/CD/F1、逐场景行和等场景宏平均；保持 checkpoint、输入帧、预处理、对齐/融合和阈值一致。
- 只跑改动方法，复用未改的 Dense/SAF3R；没有同批配对 Dense 计时则速度比 N/A。所有计时坚持单卡单 CUDA 进程。
- 拟议简化方案的端到端时间、显存、质量和速度收益目前均 N/A；不能从归一化1.448s推断可无代价省掉同样时间。

| 方法/运行 | AUC@3 | AUC@30 | ACC | Comp | CD | F1 | Dense配对加速 |
|---|---|---|---|---|---|---|---|
| 本次 v8 耗时诊断 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 拟议简化历史方案 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

第一行原因：仅做计时诊断，未运行 GT 质量评估，也没有重跑 Dense。第二行原因：尚未实现和评测。距离指标正式报告使用未平方米，CD=(ACC+Comp)/2；AUC/F1注明分数或百分比。原 v8 完整质量为独立历史结果，不从本次诊断生成新质量分数。

## 交付与复核

- [五类耗时报告](../../outputs/v8_five_bucket_profile_20261104/report.md)
- [24层实测 CSV](../../outputs/v8_five_bucket_profile_20261104/per_layer.csv)
- [分类与完整映射](../../outputs/v8_five_bucket_profile_20261104/summary.json)
- 原始事件、1522次 Graph 子阶段记录：`outputs/v8_five_bucket_profile_20261104/result/profile.json`。
- 冻结 v8 运行源码：同目录 `v8_source_snapshot.tar.gz`，逐文件哈希 `original_source_sha256.json`；不含外部数据或 checkpoint。
- 插桩脚本在同目录 `diagnostics/profile_v8_five_buckets.py`，CPU锁/监控为 `run_gpu_exclusive.py`。复现前解压冻结源码，将插桩脚本放进其`scripts/`，配置外部`checkpoints/`与`workspace/`，使用提供的`inputs/heads128.json`。不要把当前分支最新模型源码自动当作被计时的 v8。
- CPU-only 校验：`python scripts/verify_v8_five_bucket_package.py`。本次同步不新增 GPU 运行。
