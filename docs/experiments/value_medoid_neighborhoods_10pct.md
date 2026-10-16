# 保留10%代表性 Value，允许小幅位置偏移的跨层分析

沿用 Chess/seq-01 第0、20帧，每帧28×37个patch。
在前三个VGGT全局层每头的2072个patch values上分别运行k-medoids，
每头保留207个真实medoid（约10%）。48组聚类均达到单交换局部最优。
初始化仍为BUILD和固定随机种子的距离加权初始化，分别PAM优化后取目标较小者。
考虑到K增大，本轮单次运行的交换上限提升至1000。

跨层头编号无天然对应关系，本轮不使用同编号头统计。
对每个层对构造完整16×16位置相似度矩阵，然后依据每个半径的分数
分别选择头匹配。主要统计为每个源头在目标层独立选最佳头，允许多个
源头选同一目标头。附加提供全局一对一最佳分配结果作为对照，
不将一对一分配解释为头的功能对应。

头对的位置相似度定义为：

`0.5 * (源medoid在目标medoid半径内的比例 + 目标medoid在源medoid半径内的比例)`。

距离仅在同一帧的patch行列坐标间计算，为欧氏距离，分别使用0、1、2patch。
1patch约等于原图17像素，2patch约为34像素；斜向偏移一行一列为sqrt(2)patch，
不计入1patch但计入2patch。最近邻命中不要求medoid点之间一对一匹配。

随机对照在每帧内部置换目标层的patch位置，所有目标头共享该位置置换。
保持每头每帧代表点数量、目标层头之间的交集结构，打破跨层位置对齐。
共2000次，每次对每个半径重新选最佳头，因此基线包含事后选头的优势。

| 层对 | ≤1patch命中 | 随机也最佳匹配 | ≤2patch命中 | 随机也最佳匹配 |
|---|---:|---:|---:|---:|
| 第1→2层 | 48.54% | 45.52% | 79.70% | 77.55% |
| 第1→3层 | 52.63% | 45.28% | 81.43% | 77.37% |
| 第2→3层 | 52.58% | 45.48% | 81.22% | 77.70% |

完全相同坐标时三组分别为17.69%、21.29%、22.07%；相应随机基线为
13.37%、13.19%、13.19%。这些统计都是按该半径重新选择头后的结果。

小范围内存在高于随机水平的共享位置偏好。第1→3、2→3层在1patch下
比随机高约7.3和7.1个百分点，第1→2层约3.0个百分点。
2patch下约80%的命中率需结合随机也约77%的事实解释，不能直接看作强对应。
结果只针对当前两帧和这次k-medoids局部最优解，不说明跨层注意力输出的保真度。

复现：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 0 2 --ratios 0.10 --max-swaps 1000
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 1 --ratios 0.10 --max-swaps 1000
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. python scripts/analyze_value_medoid_neighborhoods.py
PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py render --ratios 0.01 0.05 0.10
```

输出 `outputs/value_medoids/chess_f000000_f000020/cross_layer_neighborhoods_keep_10%/`：

- `summary.csv/json`：3层对×3半径×3种汇总方式，共27组统计。
- `all_head_pairs.csv`：3层对×3半径×256头对，共2304行。
- `matched_heads.csv`：每个源头对应的最佳目标头，以及附加一对一分配。
- `neighborhood_vs_random.png`：主结果与随机基线及95%随机分布区间。
- `neighborhood_heatmaps.png`：各半径的16×16头对矩阵及每行最佳匹配。
- `best_layer_01_02_radius_*_on_images.png`：第2、3层最佳头对的原图位置标注。

新增测试核对网格膨胀与直接最近邻统计的一致性、不跨帧和头膨胀、
以及随机重排头编号前后匹配汇总不变。连同旧位置指标测试共6项通过。
