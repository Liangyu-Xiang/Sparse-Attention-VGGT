# 当前完整流程：Dense 前后段 + 历史继承的中段注意力

本文描述当前修正版的实际执行路径，而非诊断 oracle。层号从 0 开始。

## 1. 层安排及共享范围

| 全局层 | 输出计算 | 跨层状态 |
|---|---|---|
| 0–8 | 稠密注意力 | 更新辅助 Value 层级 |
| 9 | 稠密注意力 | 更新 Value 层级，并生成传给第 10 层的历史边 |
| 10–16 | 历史保留 + 分层探索 + 局部真实 Key 选择的稀疏注意力 | 更新 Value 层级和下一层历史边 |
| 17–23 | 稠密注意力 | 继续维护 Value 状态；不再执行中段稀疏输出 |

仅替换 VGGT 的 global attention，frame/local attention 不变。中段没有混入
Value 分支的输出选边；Value 聚类在这里提供辅助层级，不等于用代表 K/V
直接代替所有原始成员计算输出。后段恢复 Dense 不会消除中段已造成的特征变化。

所有头共享同一 Query 的最终 Key 索引，但保留各自 Q/K/V、softmax 和输出。
选边分数是“各头先 softmax，再平均概率”，不是先平均 Q/K，也不是平均最终多头输出。
每个 Query 独立决策；query_chunk 只是显存分块。

## 2. 两套分组不要混淆

### 辅助 Value 层级

聚类特征是各头 V 的平均。第 0 层从逐帧空间均匀划分开始，fine 数量约为
patch token 数的 15%。之后继承 fine 状态，在当前特征上更新质心附近的真实代表
token，构造临时特征 coarse 组，执行一次同步 sibling 局部分配，再更新代表并
重建特征 coarse 组。coarse 数量约为 ceil(Npatch/128)，轻量聚类运行两轮。
除初始化外，不保证其成员空间连续。

### 注意力探索的空间 fine 网格

这是一套固定、逐帧、空间连续的网格，fine_ratio 约为 0.25，不是上述 Value fine。
392×518 输入的 patch 网格为 28×37，空间 fine 网格为 14×18，每帧 252 格，
每格含 2–6 个真实 patch。128 帧共 32,256 个 fine 格。
一级 parent 按 2×5 个 fine 格组织；二级 child 在 parent 内由 2–3 个 fine 格组成。
空间中心代表只用于池化，不再强制作为最终落点。

## 3. 预算及第 9 层初始化

每个 Query 的总预算 M=floor(rN)，包括所有必须保留的特殊 Key。
patch 预算 K=M−Nspecial。当前 128 帧运行 r=10%，history transfer=50%。
因此 Npatch=132608、Nspecial=640、N=133248、M=13324、K=12684。
同预算下每层传递 6342 条 patch 历史边。两帧实验另测试了 20% 最终预算，
但本次 128 帧不是 20%。

第 9 层输出保持 Dense。因为 fused SDPA 不返回完整概率，当前实现另行分块重算
第 9 层 Dense QK/softmax，以各头概率均值选出高分 patch Key，并传递其中
floor(0.5K) 条。特殊 Key 不参加历史排名，始终单独保留。
跨层只传真实 Key 索引，不传旧注意力权重；新输入清空历史。

## 4. 第 10–16 层逐 Query 执行

1. 硬保留全部有效历史边。一个 fine 格内的多个不同历史 Key 都保留，不折叠为一条。
   历史所在的整个 fine 格不再参加新探索。当前层 coarse gate 不得删除历史。
2. 计算缺额 E=max(K,H)−H，其中 H 是实际历史边数。若历史超过预算，扩容而非删历史；
   当前同预算固定 50% 实验未触发这种情况。
3. 仅对未覆盖 fine 格的代表 Key 做逐 Query、逐头池化。对候选组 g 使用
   `Q_h · mean(K_h,fine-representatives) / sqrt(d) + log(n_g)`，其中 n_g 是
   可探索真实 patch 数量。各头组间 softmax 后平均。log(n_g) 修正大小偏差，
   但不等于真实成员 logsumexp，无法恢复组内方差。
4. A 路径：按一级组分数选择 E 个 fine 格，格继承 parent 分数。
   B 路径：先保留高分一级组，使覆盖的未探索 fine 格不少于
   `max(ceil(0.5 × 未探索fine数), E)`；完整 parent 可使覆盖率超出 50%。
   然后仅在这些组内计算 child 池化分数，按 child 分数选 E 个 fine 格。
   边界组按 fine 编号截断，不额外对全部 fine 代表进行独立 QK 排名。
5. 对每个选中 fine 格的 2–6 个真实 Key 计算局部 QK；每头在该格内部 softmax，
   再平均头概率，选择一个真实 Key。不同 Query 可以选中同格的不同 Key。
   这不使用 Dense 全局归一化，也不是“全局最佳成员”oracle。
6. 合并历史 Key、新 Key 和全部特殊 Key，得到最终 Mask；在这组真实 K/V 上
   执行各头稀疏 QK、支持集内 softmax 和加权 V 求和。
7. 按最终注意力的头平均概率，从 patch 支持集选出前 50% 预算作为下一层历史。
   只有这部分继承，不是历层 Mask 全量累加。

辅助全局 coarse 感知仍计算每个 Query 对 Value coarse 的低分辨率概率：
各组 Key 是 Value-fine 代表 Key 的等权平均，加真实成员数量的 log bias。
它可产生覆盖 80% 概率所需组数等集中度诊断。当前固定 50% 传递不依赖该指标，
并且这份全局 coarse 概率不再作为删除历史或限制全部探索的硬门。

修正版安装器默认 A；本次 128 帧显式采用 B，以对应此前 37.51% 受控误差的版本。

## 5. 已完成验证及边界

两帧 Chess 0/10、相同 Dense 层输入、10–16 层平均输出相对 Frobenius 误差：

| 方法 | 最终 10% | 最终 20% |
|---|---:|---:|
| 原 B | 41.41% | 31.43% |
| 修正版 B，无 Dense oracle | 37.51% | 27.61% |
| Dense 概率共享 TopK | 38.82% | 26.09% |

这些是算子误差，不是数据集精度；10% 下接近或优于概率 TopK 不能证明长序列
端到端精度相同。另有真实稀疏传播测试，验证每个 Query 的历史保留、预算唯一性、
特殊 Key、Dense 前缀一致性和重复运行一致性。

当前性能瓶颈仍存在：逐 Query 池化、排序、随机 gather、Python 循环和同步；
局部成员探测额外计算 QK，获胜 Key 在最终注意力中还会重算。
两帧 B 的“最终边 + 局部成员探测”逻辑 QK 数，约为 Dense 的 29.78%/60.04%
（对应最终预算 10%/20%），还没计池化探测。因此最终 10% Mask 不等于总计算量 10%。
逻辑计数也不等于实际 padded GPU FLOPs 或延迟，不宣称已有加速收益。

精确组 mass、Dense 最佳成员、influence TopK 只在诊断脚本中使用 Dense 信息，
不是可部署快速选择器。完整受控报告见
[report](../../outputs/region_history_corrections/report/report.md)。

## 6. 128 帧评估状态与复现

当前 Chess 128 帧 B/10%/50%history 仍在运行。Dense 对照已完成：
AUC30=88.466207%，单次前向 17.412052 秒。修正版尚无最终 AUC，不能报告其精度或加速比。
本轮 warmup=0、repeats=1，并启用逐层同步记录；属于冷启动诊断，不是稳态计时。
运行目录的动态进度不作为完成结果提交。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_corrected_region_sequence.py \
  --output-dir outputs/region_history_corrections/chess128_first \
  --scene chess --frames 128 --retention .1 --variant B \
  --transfer-ratio .5 --query-chunk 32 --warmup 0 --repeats 1 --progress-layers
```

程序拒绝覆盖已完成结果；重新运行请使用新目录。当前测量包含第 9 层种子重算和
全部状态维护，没有沿用早期“排除第 2 层初始化时间”的另一套口径。
