# 分层混合：代表 token Value 与完整 QK Top-K

层编号为 **0 起始**，处理 VGGT 的 24 个 global 层；frame/local attention 不变。

| Global 层 | 计算方式 |
|---|---|
| 0–2 | Dense；第 2 层额外初始化共享 Value 聚类 |
| 3–9 | 只对共享 Value 代表 token 的真实 K/V 计算注意力 |
| 10–16 | 计算完整 QK，再逐 head、逐 Query 选最高分 patch Keys，重新 softmax 后计算 AV |
| 17–23 | 再次只对共享 Value 代表 token 的真实 K/V 计算注意力 |

## Value 分支

所有 heads 的原始 V 平均后聚类，一套成员划分与代表索引供所有 heads 共享。
实际注意力使用各 head 自己的 Q/K/V，不平均实际输出。
代表是簇内距离当前 Value 均值最近的真实 token，**不是 Key block**。
所有 Query（包括特殊 Query）都保留并访问相同的代表性 K/V 集合。

本版本沿用已实现的 GPU 轻量层次聚类：第 2 层用 anchored coarse Lloyd 和细簇局部
细化初始化，不恢复精确全局 PAM。后续每层在上一层粗组的兄弟细簇中同步重分配两轮，
每轮重选均值最近的成员，再重建三轮 coarse Lloyd 分组。候选粗组在本层两轮中固定，
代表锚点避免空簇，稳定分段均值确保可重复。

**10–16 层也维护 Value 聚类**，但其注意力不使用 Value 选边。第 17 层从第 16 层的
当前结构继续更新，不跨过七层直接复用第 9 层旧状态。没有跨层 Attention 权重历史。

## Attention-only 分支

完整 QK 使用 FP32 分数，对所有 Keys 评分。按照 Query 分批（默认每批 64 个）只限制
工作显存，**不共享 Query Mask，也不近似省略候选 Keys**。
softmax 对同一 Query 的分数单调，因此先按 logits 选 Top-K 与按 Dense 概率选 Top-K
具有相同排序；不必保存全 NxN 概率矩阵或先计算无用的 Dense AV。

与之前的评测一致，所有特殊 Keys 强制保留并计入总预算。剩余名额严格按当前 head、
当前 Query 的 patch-Key 分数选择。因此这是“最高分 patch Keys + 必留 specials”，
不是不加例外的全局 Top-K。

对选中 logits 重新 softmax（等价于截取 Dense 概率后重新归一化），用已算好的权重
通过分块 gathered-AV GPU 算子累加真实 V。**不重新计算选中边的 QK**，也不一次展开
巨大的 `[heads, queries, retained_keys, dimensions]` gathered Value 张量。

## 固定 10% 预算

默认 `total_budget = round(0.10 * 所有 token 数)`，
`细簇/代表数 = total_budget - 特殊 token 数`。
没有块取整、在线探针、自适应扩容或误差 Dense 回退。三个前缀 Dense 层是明确的策略，
不是回退。128 帧时 N=133248、specials=640，保留 13325 Keys，其中 patch 代表/TopK
名额为 12685，稀疏层连接比例约 10.00015%。整个 24 层平均保留边约 21.25%。

“保留边”不等于“QK 实际计算量”：中间七层仍计算完整 QK。仅考虑 QK+AV 的名义 FLOPs：

- Value 层约为 Dense 的 10%。
- Attention-only 层约为 `(100% QK + 10% AV) / 2 = 55%`。
- 全部 global 层约为 `(3*1 + 14*.1 + 7*.55)/24 = 34.375%`。

这些不包含聚类、Top-K、索引访问、投影、显存流量和不同精度的执行效率，不是延迟承诺。
完整 QK/Top-K 可能让中间层比 Flash Dense 更慢，必须实测。

## 验证与复现

默认继承固定 10% 预算，采用 7Scenes 七个指定序列、每个 128 帧均匀采样，与 Dense 和
SAF3R 完整 profile（correspondence topk=128 每帧）在同输入、同 GPU/进程内对比。
所有第 2 层初始化、桥接层 Value 更新、完整 QK、Top-K 与 AV 开销均计入。
额外记录 14 个 Value 层、7 个 Top-K 层与 3 个 Dense 前缀的耗时。

```bash
PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/benchmark_adaptive_block_value.py --variants dense saf3r hybrid \
  --gpus 0 1 2 3 4 5 6 --frames 128 --warmup 1 --repeats 3 \
  --output-dir outputs/hybrid_value_topk/7scenes_n128_r10
```

[配置](../../configs/evaluation/eval_hybrid_value_topk_vggt.yaml) ·
[主实现](../../saf3r/patch/hybrid_value_topk.py) ·
[复用已选 logits 的 AV 算子](../../saf3r/triton/indexed_weighted_value.py)

本实验与旧的“全局 PAM 初始化 + q=8 邻簇”的精确版本不等同；它保留代表 token 的
注意力形式，但使用明确记录的轻量 GPU 聚类。也不能直接套用之前 L3–L9 或全层块稀疏
版本的精度数值。最终结论以新的完整前向评测为准。

### 首阶段：七场景精度验证

Chess 128 帧首轮已经显示中间七层十分昂贵，因此先以每场景一次冷启动前向验证完整
精度，而非直接投入多轮正式测速。运行目录 `outputs/hybrid_value_topk/7scenes_n128_r10_accuracy`，
使用 GPU 1–6（GPU 0 由其他用户任务占用），`--variants hybrid --warmup 0 --repeats 1`。
这轮时间包含首次编译，不称作正式预热加速比，也不将单次预测描述成多次重复稳定性验证。
可以与 v4 中完全相同输入、形状与 checkpoint 的 Dense/SAF3R **精度**比较，但不混合旧时间
宣称新方案的配对速度。上方一个预热、三次重复的命令保留为后续正式测速入口。

### 七场景精度阶段结果

| 方案 | AUC@30 | AUC@5 | 来源 |
|---|---:|---:|---|
| Dense | 84.2927% | 36.6465% | 先前 v4，逐场景核对输入路径、形状与 checkpoint 一致 |
| SAF3R | 84.5763% | 38.1370% | 同上，仅复用精度参考，不混合旧时间宣称新配对加速比 |
| 本轮混合方案 | 83.8174% | 35.0777% | 七场景各一次完整 128 帧前向 |

混合方案 AUC@30 比 Dense 低 0.4753 个百分点，比 SAF3R 低 0.7589 个百分点。
不是所有场景都改善，例如 Heads 从 Dense 的 94.01% 降到 88.68%；不能仅用 Chess
的改善代表整体效果。

单次冷启动场景宏平均：整模 127.096s、全部 global 层 118.530s。其中 14 个 Value 层
1.797s，中间七个完整 QK Top-K 层 115.017s，三个 Dense 前缀层（含初始化）1.715s。
峰值显存均值约 14.86GiB。21 个修改层固定约 10% 连接，24 层整体平均保留 21.2501%。
这些时间包含首次编译，不是多轮预热测速，但已清楚表明中间层路径是主要瓶颈。

测试结果：207 项通过、1 项跳过。详见
[完整报告](../../outputs/hybrid_value_topk/7scenes_n128_r10_accuracy/report.md)。
初始化不采用精确 PAM，已在上文明确；不要将此精度结果归因于旧的精确聚类算法。
