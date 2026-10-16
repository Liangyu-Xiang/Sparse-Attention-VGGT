# 最终方案：共享 Value 层次结构与探针校准的块稀疏注意力

本说明对应 `adaptive_block_value` 的 v4 实现和 7Scenes 128 帧评测。
它不是最初的精确局部 k-medoids，也不是只对代表 token 计算注意力的版本。

## 1. 范围与共享关系

- 处理 VGGT 全部 24 个 global attention 层（零起始编号 0–23）。
- frame/local attention 不变，原模型的 QKV 投影、head normalization、RoPE 和输出投影保留。
- 对所有 heads 的 Value 求平均，逐层维护一套共享聚类。
- Attention 感知、预算、Mask 和历史保留 head 信息；不先平均 Q/K。
- 同一 head 内每 64 个 Query 共用一组 Key blocks，各 Query 的实际权重分别计算。
- 每次新序列推理重置状态，只在同一次前向的相邻层之间传递信息。

## 2. 共享 Value 层次结构

细簇数固定为 patch-token 总数的约 15%。每簇记录成员、代表 token 和成员数。
代表是距离当前簇 Value 均值最近的真实成员，而非精确欧氏 medoid。
均值使用固定 token 顺序的分段归约，不使用非确定性的浮点原子累加。

细簇代表再进行 3 轮轻量 anchored Lloyd 分组，得到粗组。
粗组数为 `min(细簇数, ceil(patch-token 数 / 128))`；128 是原始 token
平均规模目标，不是每组包含 128 个代表，也不保证每个粗组等大。

第 0 层按帧优先的 patch-token 顺序均匀划分细簇，选取代表，再建立粗组。
此层采用均匀分散的 Key-block 覆盖顺序，不使用 Attention 定位或历史分支。
但自适应模式仍通过少量 Dense 输出探针选择总预算，不等于完全不计算探针 QK。

第 1–23 层继承上一层结构，固定做两轮：

```text
所有 token 同步重新分配 → 计算新簇均值 → 选取均值最近的真实代表
```

每个 token 只能在进入本层时继承的粗组内的细簇之间迁移。两轮候选范围不随迁移扩散。
上一层细簇代表作为锚点，留在原簇中以避免空簇。完成细簇更新后，用当前代表
重建粗分组，供下一层继承。没有全局 PAM 初始化或每层全局细簇 kNN。

## 3. Value 聚类与执行块分离

每 64 个连续 patch Keys 构成一个执行块，按帧优先顺序排列；特殊 token 单独保留。
这些是连续执行块，不是 Key 聚类，也不是 Value 粗组。

Value 覆盖优先级：将代表落在同一 Key block 中的细簇成员数相加，按此权重排序。
第 0 层使用 bit-reversal 顺序，使不同预算下的覆盖保持分散，而非取连续前缀。

选中一个 block 时，注意力使用其中全部真实 Keys/Values。代表 token 用于维护
Value 结构和覆盖优先级，不再直接充当每簇唯一保留的注意力 Key。

## 4. 保留 head 信息的 Attention 感知

每个 head 单独对各 Key block 内的 Keys 求均值。对于一个 64-query block，
取约 1/4、3/4 位置的两个真实 Query，分别计算匹配 head 的 coarse QK 和 softmax，
再平均两份概率作为该 Query block 的定位分数。

不加入 `log(group population)`。记录集中度 C 和 Query 特异性 S；
S 仅作诊断，不再将 `C*S` 直接映射为边预算。块平均 Key 仍是近似，
不能保证保留所有局部 QK 峰值。

## 5. 在线校准总预算和 A/V 比例

每层每个 head 依次尝试总预算 10%、20%、35%、50%、75%，最后 Dense。
一个总预算下比较 Attention 比例 `alpha = 0、C、1`，即 Value-only、混合、Attention-only。
Attention block 数为 `round(alpha * patch-block 预算)`，其余由 Value 覆盖填充。
同时比较是否采用通过当前相关性过滤的历史。

每层取 32 个分散 Query（块内约 1/2 位置，与路由 Query 不同），计算当前输入上的
Dense attention 输出。将其概率质量与加权 Value 按 Key block 汇总，候选 Mask
通过组合这些汇总量评估，不为每个候选重新计算 QK。

要求同时满足：

- 总输出 NMSE ≤ 0.02，即平方误差总和 / Dense 输出平方能量总和。
- 逐 Query NMSE 的第 90 百分位 ≤ 0.10。

接受第一个通过要求的总预算，并选择该预算下通过要求且总 NMSE 最低的候选。
全部失败则执行该 head 的 Dense attention。预算是每层每头选择，不是逐 Query 单独选择。
校准只依赖当前 QKV 和上一层索引，不使用真实位姿标签。

固定 10% 对照也比较混合方式，但关闭误差门槛，不因误差提高预算。
所有在线校准开销都计入正式推理时间。探针通过不保证未采样 Query 或最终位姿精度。

## 6. 历史匹配与 Mask 构建

用当前相关性评价上一层各 head 的有效历史 blocks，构建跨层 head 对应分数，
通过 Hungarian 一对一匹配寻找对应 head。匹配后相关性富集不超过均匀基准时禁用复用。
不默认同编号 head 对应，不传递旧注意力权重。

对一个 Query block，依次选择：

1. 历史复用：最多占 Attention slots 的一半，且必须位于当前相关性 top-2A 区域。
2. 当前探索：以当前相关性最高的未选 blocks 填满其余 Attention slots。
3. Value 补齐：按共享 Value 覆盖优先级，选择尚未被覆盖的 blocks。

A 是 Attention block 预算。最终集合不重复，大小等于所选 block 预算。
同一 head 的预算在本层固定，具体 blocks 随 Query block 改变。

## 7. 稀疏执行与有效历史更新

使用 Tensor Core 块稀疏算子计算 QK、在线 softmax 和 AV，不构造完整 token 级 NxN Mask。
Value-only 和混合分支不进入旧的慢速标量 indexed 路径。特殊 Keys 始终参与计算。
Dense 回退采用带显式 batch 维度的 Flash SDPA，避免长序列退化到平方显存路径。

在同一次稀疏 QK 中，对各已选 block 计算块内 logits 的 logsumexp，再对本 Query block
的 Query 求平均，作为历史排序证据。不为历史验证再次完整计算 QK。

第 0 层不保留历史。后续稀疏 head 保留
`min(已选 block 数, max(4, Attention block 数))` 个有效 blocks。
因此暂时采用 Value-only 不会清空全部历史；Value 选中的 block 也能晋升。
Dense 回退不伪造历史，因为当前 Flash SDPA 不导出这种证据。

## 8. 总边预算与默认参数

特殊 token 计入总预算。patch-block 数按
`ceil((round(rho * 总 token 数) - 特殊 token 数) / 64)` 取整，并限制在有效范围内。
实际比例按真实保留 Keys 计算，尾块使用真实长度，Dense 回退计作 100%。

| 参数 | 默认值 |
|---|---|
| global 层 | 0–23 |
| Query / Key block | 64 / 64 |
| 细簇比例 | patch tokens 的 15% |
| 粗组平均规模目标 | 128 个原始 patch tokens |
| 同步局部更新 / coarse Lloyd | 2 轮 / 3 轮 |
| 在线校准 Query | 每层每头 32 个 |
| 总预算候选 | 10%、20%、35%、50%、75%、Dense |
| 误差门槛 | 总 NMSE 0.02，逐 Query P90 NMSE 0.10 |
| 最小有效历史 | 4 blocks，受已选数量限制；第 0 层和 Dense 回退除外 |

## 9. 验证结果与明确限制

7Scenes 七个指定序列（Stairs 为 seq-02，其余为 seq-01），每个序列均匀采样 128 帧。
一个预热、三次正式重复，表中时间取场景内中位数后做场景宏平均。
包含所有初始化、聚类、在线探针、匹配、选边和回退开销。
SAF3R 使用完整 profile，correspondence topk=128 **每帧**，并非全序列仅 128 Keys。

| 方案 | 实际保留边 | AUC@30 | Global24 耗时 | 整模耗时 |
|---|---:|---:|---:|---:|
| Dense | 100% | 84.29% | 11.33s | 18.99s |
| SAF3R | 随 profile | 84.58% | 4.08s | 11.88s |
| 新固定预算 | 10.0384% | 74.20% | 4.45s | 12.77s |
| 新自适应 | 38.4790% | 83.30% | 8.90s | 16.76s |

额外诊断在另一轮不计入正式测速的前向中，使用 64 个独立 Query（块内约 3/8 位置）。
它们不参与预算选择。自适应平均独立 NMSE 为 0.010569，94.90% 的 head/层/场景
检查通过两项要求。参考是相同当前 sparse-prefix 输入的 Dense 算子，不是原始 Dense
模型每层的输入。三次正式预测逐位一致，14 次额外诊断预测与相应正式预测完全一致。
测试为 200 项通过、1 项跳过。

已纠正已知实现错误，不代表精度与速度目标全部实现。自适应 AUC@30 仍比 Dense
低约 0.99 个百分点，整模比 Dense 快 1.13 倍、但比 SAF3R 慢约 41%。Stairs 仍有
更明显的位姿回归。连续块池化、64-query 共享 Mask 及跨层累计误差仍是方法限制。
剩余主要耗时是稀疏执行约 5.27s，其次 Value 层次更新约 1.09s；Mask 构建约 0.24s。

## 10. 入口与结果

- [核心实现](../../saf3r/patch/adaptive_block_value.py)
- [GPU 算子](../../saf3r/triton/adaptive_block_attention.py)
- [默认配置](../../configs/evaluation/eval_adaptive_block_value_vggt.yaml)
- [英文详细说明与复现实验命令](adaptive_block_value.md)
- [正式报告](../../outputs/adaptive_block_value/7scenes_n128_audit02_v4/report.md)
- [独立误差与分项耗时](../../outputs/adaptive_block_value/7scenes_n128_audit02_v4/diagnostics_report.md)

仓库同步方案、代码、测试及 CSV/JSON/Markdown 统计。大型原始 QKV、Mask 重放张量、
模型权重与数据集保留本地，不作为 Git 附件上传。它们可通过捕获脚本重新生成。
