# Query-256：确定性参考下的等效加速

本批仅评估 7Scenes heads、chess、stairs，各 128 帧 uniform/seed=0。三场景等权均值不是完整 7Scenes 宏平均。用户明确授权“允许建立确定性配对参考”。原始非确定性基线单列，未覆盖或平移其分数。没有重跑 Dense/SAF3R 控制；相对 VGGT Dense 速度比全部为 N/A，原因是没有本批配对 Dense 时间。

## 验收条件与实现

验收同时要求：每场景完整模型输出 SHA-256 相同；AUC@3、AUC@30、ACC、Comp、CD、F1 原始数值完全相同；完整前向有收益。没有根据测试真值调整参数。执行参数选择依据固定 Q/K/V/支持集的算子数值与速度诊断。

算法保持 Query group=256、20% 总 Key retention（包含且保留全部特殊 Key）、原簇数、adaptive_local/32 邻居、两轮更新、25% 候选比例、四状态损失、16 轮预算、实际 history Query 间隔 256。checkpoint、输入帧、预处理、输出头与 pose/recon_unposed 协议保持一致。SAF3R Top-K 在此 Shared Value 变体中不适用；构建模型时已移除原 sparse patch，未把 lazy 对应头 Top-4 或论文 DINO Top-K 当成本方法设置。

共同确定性参考：CUBLAS_WORKSPACE_CONFIG=:4096:8；Torch deterministic algorithms=True；cuDNN deterministic=True、benchmark=False。聚类/池化等 Torch scatter 使用固定次序；历史校准的浮点求和固定次序；历史修正按稳定升序 token 成员、固定树归约，不使用浮点 atomic_add。这是获授权的执行归约参考，不声称与原非确定性分数逐位相同。

优化：保持原 32 行 Q tile、128 Key tile、4 warps、2 stages；将 7/8 普通 CTA 与实际探测 Query 所在 CTA 分开执行，普通 CTA 不携带证据寄存器。仍生成每 head 最终归一化后的证据再按原定义汇总。特殊/patch 最终证据写入同一分配中的两个连续区段，避免 patch 导出后的额外连续化拷贝。没有物化任何完整 [G,H,K,D] K/V 副本。

Graph：保留原人口分桶各自的 C/M 归约布局；几何至少宽 64，必要时扩容至 128/256/512；容量由 F 与 N/(size/2+1) 的结构上界决定，实际数量通过设备参数传入。固定数量 CTA 按运行时计数遍历活跃分桶，避免大量空 CTA。结构签名包含 shape/stride/dtype/device/捕获标量；prior、历史、eligibility 每次更新，labels/representatives/population/几何每层、每新输入更新。每次输入仍完整 reset 算法历史。缓存最多 4 个 Graph、2048 MiB retained live allocation 估算；创建单个 Graph 可短暂超过上限，超限不保留，reserved 私有池由进程峰值另报。

## 正常完整前向（不含统计探针）

| 场景 | 参考 秒 | 优化 秒 | 参考/优化 | 时间缩短 | 参考峰值 allocated GiB | 优化峰值 allocated GiB | 参考峰值 reserved GiB | 优化峰值 reserved GiB | vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| heads | 16.964746 | 16.780186 | 1.0110x | 1.088% | 18.546 | 17.319 | 24.535 | 21.832 | N/A |
| chess | 16.633520 | 16.458071 | 1.0107x | 1.055% | 19.674 | 17.316 | 28.447 | 22.611 | N/A |
| stairs | 16.903964 | 16.787459 | 1.0069x | 0.689% | 19.433 | 17.317 | 27.391 | 24.137 | N/A |
| 三场景等权均值 | 16.834077 | 16.675238 | 1.0095x | 0.944% | — | — | — | — | N/A |

每方法每场景 1 次完整预热 + 3 次同步完整前向，取后三次 wall 中位数。包括 reset、每次输入所需算法初始化、聚类、选择、输入拷贝、必要 Graph 创建、attention、历史及所有输出头。权重加载和图像解码按既定 full-forward 协议在模型计时之前完成，两方法一致。峰值为三次计时运行中的最大值。本机 GPU 上报 total=49140MiB，显存结果按本机实测，不推广为标准24GiB卡的可运行性结论。首次预热时间单独保存在 cold_samples。输出哈希校验在计时结束后执行，质量评估使用最后一次计时预测；没有将必要方法初始化或证据导出移到计时范围外。收益较小，只有三个计时样本，不据此宣称大幅或统计显著加速。

## 六项质量：两方法原始值完全一致

| 场景 | 方法 | AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| heads | reference | 42.5607 | 91.6663 | 0.0339 | 0.0339 | 0.0339 | 82.6017 |
| heads | optimized | 42.5607 | 91.6663 | 0.0339 | 0.0339 | 0.0339 | 82.6017 |
| chess | reference | 17.9216 | 86.5518 | 0.1159 | 0.0771 | 0.0965 | 45.6506 |
| chess | optimized | 17.9216 | 86.5518 | 0.1159 | 0.0771 | 0.0965 | 45.6506 |
| stairs | reference | 3.8427 | 76.5613 | 0.2111 | 0.5096 | 0.3604 | 14.9063 |
| stairs | optimized | 3.8427 | 76.5613 | 0.2111 | 0.5096 | 0.3604 | 14.9063 |
| 三场景等权均值 | reference | 21.4417 | 84.9265 | 0.1203 | 0.2069 | 0.1636 | 47.7195 |
| 三场景等权均值 | optimized | 21.4417 | 84.9265 | 0.1203 | 0.2069 | 0.1636 | 47.7195 |

AUC/F1 表中为百分比，原始 JSON 为 fraction；ACC/Comp/CD 为米；CD=(ACC+Comp)/2，未平方。F1 阈值 0.05m，下采样 0.0078125m。六项逐场景 delta 均严格为 0。原始 pose/recon_unposed 指标、融合/对齐设置、精确输入列表均保存在各 result.json、input.json 与冻结源码中。

## 原始非确定性基线：独立保留

原冻结源码在同一输入两次完整运行中，从 L0 labels/supports 开始即不相同，最终预测也不同；不是可靠的逐位参考。共同确定性参考与基础等效版分别完成两次逐层重复性检查，逐层状态及完整预测一致；最终 split 版另外通过本批三序列完整输出哈希与逐节点对照验收。原非确定性质量如下，仅作旧批次质量参照，不作本批速度配对基线。

| 场景 | 执行版本 | AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| heads | 原始非确定性旧批 | 41.9168 | 91.4628 | 0.0349 | 0.0343 | 0.0346 | 80.8910 |
| heads | 本批确定性参考 | 42.5607 | 91.6663 | 0.0339 | 0.0339 | 0.0339 | 82.6017 |
| chess | 原始非确定性旧批 | 16.0023 | 85.9925 | 0.1331 | 0.0818 | 0.1074 | 40.0973 |
| chess | 本批确定性参考 | 17.9216 | 86.5518 | 0.1159 | 0.0771 | 0.0965 | 45.6506 |
| stairs | 原始非确定性旧批 | 5.8071 | 77.6321 | 0.2083 | 0.5433 | 0.3758 | 15.9803 |
| stairs | 本批确定性参考 | 3.8427 | 76.5613 | 0.2111 | 0.5096 | 0.3604 | 14.9063 |

上述执行参考变化与“优化相对匹配参考的质量变化=0”是两个不同的差异，不能互相替代。旧批时间未用于本批配对比值。未引入 published-reference 分数或 H100 时间。

## 独立带统计前向

每场景每方法从 3 次统计前向中选择 wall 居中的一次，所有阶段、Graph 次数、显存及叶子账本都来自该同一次运行。未将逐项中位数拼接，也未缩放到正常计时。CUDA Event 包含提交间隙；Graph 使用逐次重放设备时间戳；统计探针会扰动运行。整数微秒取整差额放入显式 remainder，负 1–2 微秒仅表示取整。

| 场景 | 方法 | pass | wall 秒 | 叶子和 秒 | 峰值 allocated GiB | 峰值 reserved GiB | Graph 命中 | 创建 | 淘汰 | 常量更新 | tracked live MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | reference | 1 | 17.380065 | 17.380065 | 18.539 | 24.973 | 207 | 0 | 0 | 46 | N/A |
| heads | optimized | 0 | 17.178750 | 17.178750 | 17.324 | 22.596 | 207 | 0 | 0 | 46 | 225.715 |
| chess | reference | 0 | 17.274148 | 17.274148 | 19.674 | 26.715 | 207 | 0 | 0 | 46 | N/A |
| chess | optimized | 0 | 17.129746 | 17.129746 | 17.317 | 23.828 | 207 | 0 | 0 | 46 | 226.021 |
| stairs | reference | 0 | 17.448812 | 17.448812 | 19.431 | 27.365 | 207 | 0 | 0 | 46 | N/A |
| stairs | optimized | 1 | 17.197097 | 17.197097 | 17.320 | 22.738 | 207 | 0 | 0 | 46 | 226.228 |

参考路径的 raw live_tensor_bytes=0 是计数器未更新，故表中标 N/A，不表示 Graph 不占显存。两方法 warmed reference 都可能命中全部缓存；不能把旧非确定性批次的重捕获开销算作本批收益。缓存容量/运行时数量策略的可变输入验证见 synthetic graph 测试。

### 三场景统计均值（根项互斥可加）

| 操作 | 参考 秒 | 优化 秒 |
|---|---:|---:|
| 图像编码 | 0.922279 | 0.927424 |
| Frame blocks | 1.688993 | 1.695441 |
| Global attention | 9.867223 | 9.615587 |
| Global MLP | 0.400053 | 0.402556 |
| Global归一化、残差等 | 0.206673 | 0.206832 |
| 相机头 | 0.016025 | 0.013720 |
| 深度头 | 1.979081 | 1.985251 |
| 点云头 | 1.980632 | 1.986532 |
| 模型其余：初始化、布局、未细分区间 | 0.306586 | 0.335065 |
| Wall计时边界、返回与末尾同步差额 | 0.000131 | 0.000122 |
| 未细分、分配、提交与阶段边界 | -0.000001 | 0.000001 |
| 完整 wall | 17.367675 | 17.168531 |

### Attention/历史/Graph 展开（父子项有包含关系，不能重复相加）

| 操作 | 参考 秒 | 优化 秒 |
|---|---:|---:|
| 支持集上的attention执行 | 3.724440 | 3.472022 |
| 索引式融合QK、softmax、AV及inline采样 | 3.692979 | 3.440872 |
| 独立历史概率导出 | 0.013138 | 0.012174 |
| 历史概率统计与补全 | 0.789101 | 0.782158 |
| 支持集构建 | 4.273993 | 4.271228 |
| Value聚类 | 2.193471 | 2.178063 |
| 池化与共享几何 | 0.194292 | 0.194771 |
| Selector与Graph管理 | 0.640191 | 0.642835 |
| Graph缓存miss时创建与捕获 | 0.000000 | 0.000000 |
| 跨层几何常量拷贝 | 0.008007 | 0.009696 |
| 每窗口prior等输入拷贝 | 0.013230 | 0.013275 |
| CUDA Graph重放 | 0.613577 | 0.612584 |
| 四状态误差、成本和目标 | 0.108359 | 0.106187 |

## 数值、状态与范围验证

24 passed、1 skipped：尾组/无效槽/特殊 Key/重复索引防护、24 层执行/reset、独立 per-head 概率参考、拆分 launch 与原 branch 的逐位比较、确定性修正数学/重复性/CUDA Graph 捕获、历史归一化与 FP16 cache。跳过项是显式不支持的 Query-128 split execution；该开关只允许 Query-256/32 行/单实际探测 Query。

固定大形状 Q/K/V/支持集（16 heads，R=16384，N=131072，Kpatch=26000）下，含全部证据导出的完整 wrapper：原 branch 20.655ms，split/stages2 18.640ms；输出、LSE、特殊/patch 历史概率均 bit equal。stages1 较慢未用于最终配置。算子数据仅作诊断，不冒充完整方法收益。

| 真实 128 帧序列 | attention 检查次数 | Graph 选择检查次数 | 常量拷贝检查次数 | 输出/LSE/证据/四状态/支持集 |
|---|---:|---:|---:|---|
| heads | 216 | 207 | 46 | 全部逐位相同 |
| chess | 216 | 207 | 46 | 全部逐位相同 |
| stairs | 216 | 207 | 46 | 全部逐位相同 |

Graph 诊断更换 prior、历史 ID、eligibility、Value、labels、representatives、population 与分桶人口：固定结构不重捕获；宽度从 64 扩至 128（真实最大人口 105）正确创建新结构，随后值变化继续命中；支持集逐位一致，无截断。完整输入 reset 清空历史；Graph 仅复用拓扑与缓冲区。

所有诊断的六指标为 N/A 并说明原因；所有正常质量运行独立执行 pose/recon_unposed 并保存原始指标。早期 Torch deterministic scatter 无法被 Graph 捕获的失败原型保留并标失败；较慢候选的原始结果保留在 inventory.json，不把它们冒充收益。见旁边 inventory.json。

## 复现与工件

- 优化配置：[reuse_shared_value_loss_q256_deterministic_split.json](/data_NVME1/mmc_lyxiang/3D/SAF3R-query-group-scaling/configs/evaluation/reuse_shared_value_loss_q256_deterministic_split.json)
- 参考配置：[reuse_shared_value_loss_q256_deterministic_reference.json](/data_NVME1/mmc_lyxiang/3D/SAF3R-query-group-scaling/configs/evaluation/reuse_shared_value_loss_q256_deterministic_reference.json)
- 本批双方冻结源码：[source_snapshot.tar.gz](/data_NVME1/mmc_lyxiang/3D/SAF3R-query-group-scaling/outputs/query256_deterministic_split_20261105/source_snapshot.tar.gz)，SHA-256 清单 source_sha256.json；原未提交基线冻结在 outputs/query256_efficiency_20261105/frozen_baseline.tar.gz。
- 每运行 *_command.json 保存 argv/cwd/额外环境变量；input.json 保存精确帧索引/路径/形状。checkpoint 同一文件，哈希见 provenance.json。
- 正常 result.json/forward.json/quality.json/layers.json/output_sha256.json；独立 event_profile_*.json；选中同一次统计前向的 selected_tree.json 与 selected_leaf_ledger.csv；三场景均值叶子 CSV；GPU 进程监控每 0.5s 采样并在外部上下文重叠时中止。短于采样间隔的上下文不能完全排除。

```bash
/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_final.py
/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_final.py --profile
/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/run_query256_live_split.py
/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python scripts/report_query256_final.py
```

已完成结果会跳过，未完成目录会保留；新复测需复制该冻结工件至新实验目录并更新 OUT，避免覆盖原始记录。正常速度与带统计速度分表，不能把两者阶段混用。相对 Dense 速度始终 N/A；没有重新运行未改的 Dense/SAF3R。
