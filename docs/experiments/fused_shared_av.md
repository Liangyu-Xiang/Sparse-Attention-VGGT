# 全层共享Value／演化Attention融合：实现与前置验收

## 状态

已实现独立融合入口、Tensor Core内核、正确性测试、性能分解和禁止越过失败门槛的实验流程。
**尚未达到最终验收，不是已完成的近无损2倍加速方案。**

128帧Chess前置测试，预热1次、正式3次取中位数：

| 指标 | Dense | 当前融合 |
|---|---:|---:|
| 完整前向 |16.6402s |13.9401s |
| 峰值显存 |14.3586GiB |14.9271GiB |
| 位姿AUC30 |88.4662% |87.3659% |

实际加速约1.194倍，未达2倍；显存增加约0.569GiB；此单场景AUC30下降1.1003个百分点，
也超过逐场景1个百分点的保护线。此处没有深度和重建完整评估，因此总体质量门槛为
“不可判定／未通过”，不能把缺失指标当作通过。
原始配对结果、独立profile及验收JSON：
[当前128帧报告](../../outputs/fused_shared_av/chess128_final_gate/report/report.md)。

按计划，失败前置门槛阻止ETH3D大规模校准、七场景完整评估和自适应预算试验。
没有静默提高20%上限、加入Dense前缀或Dense回退，也没有改动历史方案结果。

### 2026-10-19：SAF3R显存复核与300帧测试

均为Chess seq-01、均匀采样、392×518、BF16，预热1次、测量3次取中位数；
同帧数严格核对输入清单，只评估位姿，不代表七场景宏平均或完整三任务评估。

| 300帧配置 | 完整前向 | 相对Dense加速 | allocated峰值 | AUC30 |
|---|---:|---:|---:|---:|
|Dense |70.530s |1.000× |25.090GiB |88.3168% |
|当前融合，20%预算 |62.714s |1.125× |26.532GiB |87.6535% |
|SAF3R默认每帧Top-4 |25.033s |2.817× |26.480GiB |88.3653% |
|SAF3R此前对比的每帧Top-128 |OOM |— |— |— |

Top-128在300帧下仅选边索引就需44.46GiB，48GiB设备无法完成。Top-4是明确区分的
官方默认补测，不用其成功结果代替Top-128的失败。当前环境沿用已有PyTorch分块选边兼容
路径，SAF3R注意力内核保持不变。

128帧Top-128旧测试保留失效历史索引时，29.882GiB峰值可复现；每次选边前释放全部
失效索引引用后，峰值22.453GiB，预测指纹逐次完全一致，仍计入每次重新选边成本。
128帧默认Top-4为8.123s、14.612GiB。allocated峰值与allocator reserved峰值分别记录。

[复测报告及所有配置、输入、源码快照](../../outputs/fused_shared_av/recheck_20261019/report.md)。
复现入口为`evaluate_fused_shared_av.py --mode saf3r --saf3r-topk 128`（或4），
默认`--saf3r-cache-reset release`；`retain`只用于复现旧测试的额外显存。
300帧仍未达到融合方案速度、显存及近无损目标；本次没有调整参数或绕过前置门槛。

## 1. 分支和预算

仅替换VGGT全部24个global attention，frame/local层、投影、RoPE和模型权重不变。
所有head共享选出的索引，但保留各自Q/K/V、softmax和输出。

| 零基层号 | Attention／Value patch预算比例 | 初始总边预算 |
|---|---|---|
|0 |0／100% |20% |
|1–9 |20／80% |20% |
|10–16 |80／20% |20% |
|17–23 |20／80% |20% |

总Key预算 `floor(rho*N)` 包含5个特殊Key/帧。先扣除特殊Key，再按alpha划分patch预算。
Value代表先占其固定配额；历史中恰好等于当前Value代表的边计入Value，Attention补齐其他
不同Key，保证最终唯一边数与两个互斥配额均准确。不是两个独立输出的线性加权。

入口：`saf3r.patch.fused_shared_av.install_fused_shared_av(model, config)`。
`FusedSharedAVConfig`接受长度24的`attention_ratios`、`retentions`；预算限10%–20%，
第0层必须纯Value，无Dense种子。默认Query块32、Key计算tile64、处理窗口128个Query组、
历史比例0.5、局部更新2轮、每个固定局部源组最多32个基础簇。
人工配置为`configs/evaluation/fused_shared_av_manual20.json`。

## 2. 全覆盖Value

基础fine簇数为24层中最大的Value代表预算。第0层逐帧均匀初始化，准确维持簇数，
根据头平均V选择簇均值最近的真实成员，非精确欧氏k-medoids。
FP32头平均直接归约，不分配完整的`H*N*D` FP32 Value中间副本。

后续层执行两轮同步局部分配与代表更新。上层代表是锚点，不产生空簇。
固定源组内的成员表和候选表一次构建，批量FP32矩阵距离避免逐Token/逐簇Python循环。
源组在序列内固定、不能跨源组扩散；这与旧版每层重建特征coarse组的轨迹并不逐字节等价。

当前Value预算较小，就把全部基础fine簇按其稳定编号连续合并成指定数量的非空簇，
用所有真实成员重新计算均值和最近真实代表。没有通过截取fine代表子集丢弃部分区域。
合并映射按预算缓存；每Token始终属于一个Value簇。Value代表K/V每层收集一次。

## 3. 演化Attention

与Value分组独立，沿用逐帧空间fine网格及parent/child两级布局，成员表均缓存。
coarse Key是该组fine代表Key的等权平均，每层只池化一次，加真实成员数log bias。
池化直接从原始Key索引读取，不构造完整patch-only K副本。

每32个Query共享支持集，使用块内第4/12/20/28位置的4个真实Query感知。
每head先softmax，再平均head和探针概率。GPU流式归约coarse概率，不保留完整的
FP32 `H*Query组*4*coarse数` 概率张量。

每层先硬保留历史；Value或历史覆盖的fine格不再探索。parent选择覆盖至少半数
未覆盖fine格，并满足补边数量；仅这些parent中的fine格能进入child选区。
当前实现仍计算全部child池化分数，实际计数包含这些工作，不声称算力已按选区完全减少。
按child分数排序，然后线性压缩合法fine成员；不再排序大部分Token级fine分数。
同child内的边界平局按fine编号处理，这是显式的确定性选边规则。

对选中fine格，在4个探针、各head上计算格内真实Key的softmax，平均后选一个真实Key。
Tensor Core批量处理8个fine格；不同Query组可以选择同格内不同Key。
保留粗到细探索和局部真实Key桥接，不是固定Mask复用，也没有Dense精确TopK。

历史由同次最终QK产生的**支持tile概率质量**排序。行logsumexp使概率的分母包含
Value、Attention与特殊Key，但特殊Key不参加历史排名。选完整／部分支持tile，
这是原逐Token置信度TopK的近似，不冒充等价优化。
只传Key索引，不传旧权重；提前知道下一层人工预算，历史数取下一层Attention配额的一半，
并受当前支持容量约束。纯Value层也能提供历史。每次新输入清空所有状态。

## 4. 统一Tensor Core执行及内存边界

同一在线softmax状态依次处理：共享Value＋特殊Key连续bank、Query组的Attention Key索引。
使用`tl.dot`执行QK和AV；不会分别归一化两分支后相加。原始多头输出继续经原投影层。
循环按两类bank静态拆分，避免动态分支阻止加载／矩阵乘法流水化。

不构造Token级`N*N` Mask，不保存完整所选logits。历史证据只有每行每支持tile的质量及
行归一化数据，在受限Query窗口内生成并归约。临时池化bank在层结束释放。
统计最终唯一QK、显式最终tile padding、路由探测和局部成员探测；局部探测的获胜Key
对4个探针仍会在最终计算中再次出现。**完全消除这部分重复尚未实现**，不能宣称所有QK只算一次。
终端Tensor Core还使用16行探针tile以及成员padding，逻辑探测次数不等于硬件FLOPs。

独立profile（128帧，非主计时，部分项存在包含关系）：

| 项目 | 秒 |
|---|---:|
|所有global模块 |9.067 |
|Value维护／合并代表 |0.048 |
|最终统一执行 |3.806 |
|选边，含终端 |3.629 |
|其中终端真实Key选择 |1.807 |
|历史更新 |0.559 |

Dense独立profile：完整16.645s，global11.543s，其余5.102s。要实现完整2倍，
global及全部附属成本合计须约≤3.22s；目前仅最终统一执行已超过此上限。
当前瓶颈已经不是Value聚类，而是执行、探索和历史整理；必须继续优化，不能靠指标掩盖。

## 5. 复现与实验门槛

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_fused_shared_av.py \
  --output-dir outputs/fused_shared_av/new_gate --frames 128 --profile
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_fused_shared_av.py \
  --output-dir outputs/fused_shared_av/new_gate --frames 128 --mode dense --profile
PYTHONPATH=. python scripts/report_fused_shared_av.py \
  outputs/fused_shared_av/new_gate --scenes chess
```

默认预热1次、正式3次；主前向无逐层hook或同步，计入全部初始化和选边。
`--profile`另外运行一次独立profile，不混入正式耗时。`--capture`仅供小输入诊断，
有capture的时间不能作为性能验收。结果拒绝覆盖已完成输出，均保存源码、输入、配置和样本计时。

`scripts/run_fused_shared_av_stages.py`提供：

- `ablate`：纯Value、纯Attention、融合的同预算对照。纯Attention控制的第0层仍按约束使用Value。
- `calibrate`：仅ETH3D，沿仓库profile所列场景、校准文件顺序及过滤规则；先20/15/10%统一预算，
  再按浅／中／深段压缩，人工比例保持不变；选择通过门槛的最快候选，生成冻结配置和SHA。
- `final`：要求独立冻结配置，按SAF3R默认七场景128帧运行位姿、深度、无位姿重建。

这些阶段均要求有效128帧预热速度／显存门槛。拒绝使用50帧结果、冷启动结果或失败报告解锁。
启动前检查指定GPU空闲，不终止其他用户任务。

宏平均要求AUC30损失≤0.2pp、F1损失≤0.5pp、AbsRel增幅≤1%；逐场景保护线
分别为1pp、3pp、5%。速度≥2倍、每场景显存不高于Dense。缺少三类指标、场景不全或
输入／权重／源码／环境不匹配时，都不能报告最终通过。

## 6. 已验证与尚未执行

41项相关测试通过，包括CPU/GPU统一softmax、特殊Key、部分tile、真实代表全覆盖、
历史硬保留、去重、纯分支、20→15→10%预算缩小、全24层状态、新输入重置、
流式coarse概率、局部真实Key投票，以及验收条件和失败门槛拒绝执行。
另有2/50/128帧实机结果及旧版本profile保存在`outputs/fused_shared_av/`，不覆盖旧版本。

ETH3D预算搜索、七场景最终三任务评估和可选在线自适应预算尚未执行。
自适应控制器不作为当前已交付能力：人工版未通过门槛前，不启动自适应来绕过失败。
