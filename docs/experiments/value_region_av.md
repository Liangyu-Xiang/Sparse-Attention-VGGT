# Value区域整体选择：Attention分支修改

这是独立入口`mode=region`，旧版`mode=fused`保留。全部24个global层采用原人工比例与20%目标边预算；
frame/local注意力不变。不引入Dense种子、Dense回退，也不使用稠密Top-K选边。

## 当前预算与入口

入口为`install_value_region_av(model, ValueRegionAVConfig(...))`，评估使用`--mode region`。
旧实现仍为`--mode fused`，两者不是逐位等价选边。

| 零基层号 | Attention占patch预算 | Value占patch预算 | 目标总边比例 |
|---|---:|---:|---:|
|0 |0% |100% |20% |
|1–9 |20% |80% |20% |
|10–16 |80% |20% |20% |
|17–23 |20% |80% |20% |

设总Token数为N，特殊Key数为S，先计算`P=floor(0.2*N)-S`，
然后`KA=round(alpha*P)`、`KV=P-KA`。特殊Key始终保留，计入总预算。
比例是互斥唯一patch边的预算占比，不是两路输出的加权系数。

当前128帧输入：N=133248，patch数132608，特殊Key数640，每Query目标总Key数26649，
patch预算26009。第0层Value代表26009个；浅/深层Attention目标5202、Value20807；
中层Attention目标20807、Value5202。基础fine簇数始终取最大Value预算26009，
Attention使用这套基础fine簇，而不是随本层KV大小切换到合并后的Value粗簇。

## 搜索结构

Fine划分直接采用当前层Value分支维护的基础fine labels，不再构建另一套空间fine grid。
Value更新、均值最近真实代表、较小预算下的全覆盖合并与旧版一致。
基础fine簇仍是受限局部更新得到的近似聚类，不声称保证全局最优或严格的簇内失真界。

默认每32个基础fine簇构成一个parent，每8个fine簇构成一个child。
稳定fine编号的层级沿用Value局部源组；成员随Value迁移更新。它不是另外一套空间格划分，
也不额外执行全局层次聚类。parent/child的Key为所含fine真实代表Key的等权平均，每层构建一次。

更具体地，Value迁移被限制在固定源组内，因此parent的Token并集保持该源区域不变，
fine和child成员可在parent内部变化。初始源区域仍有来自均匀初始化的局部空间结构；
“取消独立空间fine grid”不意味着当前方案已变成跨全部Token的全局语义层次聚类。

每32个相邻Query共享支持集。块内第4/12/20/28位置的真实Query用于感知；各头分别softmax，
再平均head和探针概率。log人口偏置使用区域完整成员数，不因为Query历史变化而重建池化Key。
所有实际Query仍使用自己的Q和各头权重执行最终注意力。

组内位置是从0开始的索引，尾Query组不足32个时将越界探针夹到最后一个实际Query。
Parent评分对全部parent做softmax；Child仅对边界parent的children做softmax；
Fine仅对边界child的fine候选做softmax。空的padding区域log人口为负无穷，概率为零。
先逐头逐探针softmax、后平均概率，绝不是先平均Q/K再计算注意力。

默认128帧下是813个parent，每个最多32个fine；每parent至多4个child，每child至多8个fine。
按稳定fine编号分组是确定性的低成本层级，不是又按K/V距离做了一次parent/child聚类。

## 可停止的区域细化

1. 历史边硬保留，与Value代表相同的边计入Value；剩余Attention预算按唯一边数计算。
2. 按当前Query组的parent概率排序，依次接收能完整放入预算的parent。
3. 仅对跨过剩余预算的一个parent细化，计算其children的概率。其他选中parent直接成为最终支持。
4. 同样仅对跨过预算的一个child细化到fine；其他选中child直接成为最终支持。
5. 最细层选择完整fine簇前缀，直到补足预算，最后一个fine簇允许超出。不做格内真实Key QK或softmax。
6. 所选区域贡献全部真实成员；删除与Value/history的重复Key后，与特殊Key一起统一softmax、AV。

父层恰好满足预算就不继续细化；GPU边界评分内核会跳过不活跃Query组。
不会对全局全部child或fine计算QK，只评分预算边界的4个child或8个fine。
没有逐parent/child的quota优化或token级重要性排序。

更准确地说，接收的是排名前缀：遇到第一个装不下的区域时细化该区域，
不会跳过它、转而用更低排名的小区域填预算。区域可用边数始终扣除已在Value/history中的Key，
但区域池化Key和log人口仍来自完整区域，不按Query历史重新池化。

“已覆盖”指Key已经在支持集中，只有全部成员都已覆盖的区域才能跳过。
不能因为fine簇里已有一个Value代表/历史Key而排除整簇，否则会漏掉大量未计算成员。

新增唯一Attention边数为`A`，目标为`KA`，满足：

`KA <= A < KA + max_current_fine_cluster_population`。

这里A包括保留下来的、不与Value重复的历史边。若历史实际保留H条，探索补边目标为`KA-H`。

完整区域的所有成员都在最终并集中，重复边不重复占预算。实际边比例、最大超出量和padding分别记录，
不把20%目标描述为严格20%最终边数，也不把它描述为20%总计算量。

## GPU实现和边界

成员表在每个Value局部源组内按当前fine label组织，不进行逐Query的全Token排序。
通过区域描述符（成员表起点、长度）流式展开全部成员；CUDA内核分块压缩索引，无Python逐簇循环。
为历史去重使用受窗口限制的`Query组窗口 × patch数` uint8查找表；这不是完整Token×Token Mask，
但其分配/清零/访存仍是真实开销，全部计入时间和显存。支持集使用紧凑Query组索引和-1尾padding。

继续使用原统一Tensor Core执行算子；一次在线softmax同时覆盖Value、Attention、特殊Key。
历史由同次最终QK的tile概率质量选择，变长支持显式排除padding/特殊Key。按下一层目标预算限制历史长度。
新输入清空所有历史和Value状态。旧版终端探针与最终QK重复的问题在此分支消失；区域感知仍有额外QK。

## 历史如何发送到下一层

最终注意力在统一归一化过程中输出每个64-Key支持tile的patch概率质量与每行logsumexp，
而不是保存所有所选Key的完整logits。聚合当前Query组内的Query和head后，按tile质量排序，
依次保留完整/部分tile内的真实Key，直到凑够历史长度。

这是**tile质量近似**：不等价于按每个Key的精确平均注意力排序；最后一个tile会截取前缀，
也不保证传递的是完整Value簇。特殊Key、-1 padding不参与历史选择，特殊Key仍在softmax分母内。
历史可来自Value或Attention支持集，跨层只传索引，不传旧权重；下一层会重新计算这些Key的当前权重。

当前层发送长度为`min(本层目标patch预算, floor(0.5*下一层KA))`。
下一层收到后，与新Value代表重叠的历史Key归入Value，不占Attention预算；
因此“历史约占Attention的一半”是发送上限，不是保证每层去重后仍恰好一半。
第0层虽不运行Attention搜索，仍从Value支持的最终注意力提供下一层历史；第23层不再发送。

## 验证与复现

正确性测试覆盖整区域成员、去重、硬历史、预算超出界、纯分支/预算缩小、统一softmax、
变长padding、尾Query组、动态Value成员表、逐头逐探针评分及跨多个块的GPU展开与CPU对照。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_fused_shared_av.py \
  --output-dir outputs/value_region_av/new_trial --mode region --frames 128 --profile
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_fused_shared_av.py \
  --output-dir outputs/value_region_av/new_trial --mode fused --frames 128 --profile
PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python scripts/evaluate_fused_shared_av.py \
  --output-dir outputs/value_region_av/new_trial --mode dense --frames 128 --profile
python scripts/report_value_region_av.py outputs/value_region_av/new_trial
```

预热1次、正式3次取中位数，主计时无逐层同步；独立profile另跑一次，不加入主计时。
Chess seq-01均匀128帧，保存具体输入清单、配置、源码快照；只评估位姿，不外推为七场景三任务结果。
首次实现中途调整JIT参数特化的未完成输出会保留为`region_interrupted_compile_v1`，不用于最终比较。

## 首轮128帧Chess结果

同次配对测试，参数未按Chess真值搜索；相同20%目标和人工分支占比。

| 方法 | 完整前向 | 相对Dense加速 | allocated峰值 | AUC5 | AUC15 | AUC30 |
|---|---:|---:|---:|---:|---:|---:|
|Dense |16.4117s |1.000× |14.3586GiB |39.6949% |77.3671% |88.4662% |
|旧融合 |14.2988s |1.148× |14.9271GiB |37.6427% |75.5668% |87.3659% |
|Value完整区域 |12.5754s |1.305× |14.9269GiB |36.7692% |75.0074% |87.0571% |

新版比旧版耗时降低12.05%，但AUC30再降0.3088个百分点；相对Dense下降1.4091个百分点。
**不满足近无损、2倍加速和显存不高于Dense的目标。** 这里没有深度/重建结果，不能报告总体质量通过。
实际唯一边比例约19.99955%–20.00141%（目标取整使第0层略低于20%），单Query组最多多20条边；
各层均验证了超出量严格小于本层最大fine簇人口。

独立profile中选边从4.0142s降到2.5244s；旧终端真实Key选择1.7761s被删除，
新整区域展开0.3074s。最终注意力执行仍约3.7007s，历史0.5525s，Value维护0.0475s。
parent评分/排序0.7659s，边界child0.4633s，边界fine0.4204s；它们包含在2.5244s内，不能重复求和。
本次改变减少了搜索量，但没有解决最终执行和剩余选边/历史整理的全部开销。

[详细配对报告、逐层边比例与源码快照](../../outputs/value_region_av/chess128_boundary_regions/report.md)。
最终源码与实测快照SHA256一致；已包含非法fanout提前校验与空区域softmax屏蔽，52项相关测试通过。
`region_before_empty_region_guard`是空区域校验前的开发版本，不参与最终比较。
