# 第12–14层：k-medoids与等预算空间均匀采样

使用此前Chess/seq-01第0、20帧的相同dense VGGT输入。
第12–14个全局层对应L11–L13，每层16个头。各头均从2072个patch keys
中保留207个（约10%），额外完整保留两帧共10个特殊keys，评估全部2072个patch queries。

主基线是空间分层均匀采样：每帧划分为若干行带，每个行带划分矩形格，
每格取中心patch；格子数严格等于该头在该帧的medoid数量。
行带数按`round(sqrt(K_frame * H/W))`确定，并受合法行列数约束；
行带的列格数近似相等，行带高度按格子数分配，使单格面积近似相等。
帧内位置选择不利用Q、K、V或attention权重。额外提供每帧近似平分
总预算103/104点的均匀采样，避免帧预算设置的影响被忽略。

两种选点均使用原始QK scores，在保留的patch keys与全部special keys上
重新softmax。所有queries共用本头的选点集合，不按query另选keys，
不按medoid簇大小加权。原模型始终dense执行，每层均在相同dense输入上
独立评估，误差不跨层累积。重新计算的V与保存medoid时的V经过逐元素完全一致核对。

评估包含：

- 注意力输出NMSE：`sum_i ||o_sparse_i-o_dense_i||² / sum_i ||o_dense_i||²`，每头分别计算。
- 每query输出余弦相似度。
- 保留的dense attention质量（含特殊tokens与条件patch质量分别计算）。
- Value代表性：全部V到最近选中V的平均欧氏距离，以及最近V重构NMSE。

dense主干BF16，指标在FP32中计算、禁用TF32，误差与能量以FP64汇总。
以下平均值是16个头NMSE的算术平均，不是跨头按输出能量加权的NMSE。

| 全局层 | medoid平均输出NMSE | 均匀中心平均输出NMSE | 平均NMSE降低 | medoid胜出头数 |
|---|---:|---:|---:|---:|
| 第12层（L11） | 0.259862 | 0.493286 | 47.32% | 16/16 |
| 第13层（L12） | 0.334583 | 0.613389 | 45.45% | 15/16 |
| 第14层（L13） | 0.673128 | 0.841833 | 20.04% | 14/16 |

均匀中心采样更好的三个头：第13层H07、第14层H06/H14。
Value平均最近距离在48个头上均为medoids更低。
第12/13/14层平均距离分别为medoids的1.3524/1.2960/1.4662，
以及均匀中心的1.5814/1.5462/1.7491。

为检查格内采样相位敏感性，每格额外取(1/4,1/4)、(1/4,3/4)、
(3/4,1/4)、(3/4,3/4)四种位置，连同中心共五种布局。
从每头五种NMSE中事后挑最低者作为oracle，均匀oracle平均NMSE为
0.289008/0.359173/0.636163；medoids分别在11/13/11个头上优于该oracle。
第14层medoid虽然在11个头胜出，但平均NMSE比oracle高约5.8%，
说明结果对个别头和均匀布局相位仍敏感。
此oracle需要dense输出评判，不能作为免dense成本的实际选择器。

这些是两帧上单层、投影前的attention输出指标，不代表全模型位姿或重建质量。
也未对比选点耗时，k-medoids使用完整V与离线迭代，不能据此声称加速。

输出目录：`outputs/value_medoids/chess_f000000_f000020_layers_12_14/vs_uniform_10pct/`。
包含48张每头两方法的原图对照、3张16头总览、48页PDF、逐头误差图、
全部7种选点布局的CSV指标、保存的支持索引、主对照汇总JSON与HTML浏览页。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/compare_value_medoids_uniform.py evaluate --layers 11
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/compare_value_medoids_uniform.py evaluate --layers 12
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/compare_value_medoids_uniform.py evaluate --layers 13
PYTHONPATH=. python scripts/compare_value_medoids_uniform.py render
```

新增测试独立核对各帧预算和点唯一性，并以PyTorch SDPA核对选中keys后的
softmax输出误差，2项测试通过。
