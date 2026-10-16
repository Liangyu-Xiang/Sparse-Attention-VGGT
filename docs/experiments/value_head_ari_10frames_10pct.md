# 10帧、10%代表点：前三层相邻层全部头对的Value聚类ARI

## 本次改变

保持已有10帧联合dense前向V和实验定义不变，只将medoid比例从1%提高到10%。
输入为Chess/seq-01第0、5、10、15、20、25、30、35、40、45帧，VGGT输入
518×392，每帧28×37个patch。前三个global layers L00–L02，各16头独立处理。
每头的全部10360个64维原始V都参与聚类，不做头平均或V归一化，不包含special tokens。

10%指代表点比例：K=1036，全部10360个patch仍按欧氏距离归到最近的真实V medoid。
不是只抽10%的token参与聚类。平均每簇10个patch，实际簇大小不要求均匀。
1%实验K=104的结果保留不动。

PAM使用BUILD和固定种子的距离加权初始化，各执行穷举最佳单交换，选目标较小者。
本次交换上限3000，以免沿用1000上限导致提前停止。48组选中解均收敛，选中解
执行403–1094次交换。这只证明容差下无改善单交换，不保证全局组合最优。
实验使用空闲GPU1/2，原始V缓存来自此前10帧dense前向，模型参数未修改。

## ARI结果

对第1→2层、第2→3层分别比较全部16×16个头对，共512个ARI；不假设头编号
或簇编号直接对应，不只保留最佳配对。统计同一批10360个patch的完整簇归属。

| 相邻层 | 全256头对平均ARI | 中位数 | 最小值 | 最大值 | 最大值的头对 |
|---|---:|---:|---:|---:|---|
| 第1→2层 | 0.212133 | 0.204342 | 0.102038 | 0.322730 | L00 H03 → L01 H02 |
| 第2→3层 | 0.313457 | 0.314777 | 0.188928 | 0.456451 | L01 H02 → L02 H11 |

若对每个源头仅选择下一层最相似的头，再对16个源头平均，分别为0.284241、
0.340316；这个次要统计不能与上表全部256头对的平均混淆。

后续按用户要求，正式匹配方向已改为“每个后层头选择最佳前层头”，即按每列
取最大值，对16个后层头平均得到0.240558、0.433264。当前图中红框及粗体
表示每列最优；完整选择见`all_head_ari_10pct/best_previous_heads.csv`。
与先平均16头V再聚类的结果并列说明，见`value_head_matching_and_mean_10frames_10pct.md`。

| 相邻层 | 原1%平均ARI | 当前10%平均ARI | 绝对变化 |
|---|---:|---:|---:|
| 第1→2层 | 0.164518 | 0.212133 | +0.047615 |
| 第2→3层 | 0.252156 | 0.313457 | +0.061301 |

本输入上，10%更细粒度划分的平均跨层ARI高于1%结果，第2→3层仍整体高于
第1→2层。ARI度量簇划分的一致性，不是未改变token百分比，不等同于注意力输出
误差或下游任务质量；不同K对应不同划分粒度，不能将ARI提高直接解读为模型更好。
结论仅限这10帧。

## 实现与验证

- 报告脚本新增`--ratio`，默认仍为1%；10%输出单独保存，旧结果不覆盖。
- ARI直接使用非零联合计数，避免K=1036时构造16×16×K×K稠密计数张量。
- 新计数方法独立复现已有全部512个1%ARI，与保存结果在1e-12内一致。
- 本次全部512个10%ARI逐项对照sklearn `adjusted_rand_score`，最大绝对误差5.55e-17。
- 48个头均用FP64 norm/GEMM距离核对全体patch为容差内最近medoid；选中medoid距离另用
  直接FP64向量差核对。检查medoid互异、自身归属、簇大小、目标值与收敛状态。
- 本次每头的聚类目标均低于同头原1%结果。13项相关单元测试通过。

## 输出和复现

目录`outputs/value_medoids/chess_n10_s5/`：

- `keep_10%/layer_XX/head_YY.npz`及同名JSON：48头聚类归属与全部优化信息。
- `all_head_ari_10pct/index.html`：两张完整16×16带数值热图，行是前层头，列是后层头。
- `all_head_ari_10pct/all_head_pairs_ari.csv`：全部512个全精度ARI。
- `all_head_ari_10pct/L00_L01_ari_matrix.csv`、`L01_L02_ari_matrix.csv`：矩阵格式。
- `all_head_ari_10pct/all_adjacent_head_ari.pdf`：两页矩阵。
- `all_head_ari_10pct/summary.json`：输入、比例语义、核对方式和汇总统计。

```bash
# 独立终端并行运行两个头分片；重用已有values.npz，不重新提取V：
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s5 --layers 0 1 2 \
  --heads 0 1 2 3 4 5 6 7 --ratios 0.10 --max-swaps 3000
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s5 --layers 0 1 2 \
  --heads 8 9 10 11 12 13 14 15 --ratios 0.10 --max-swaps 3000
PYTHONPATH=. python scripts/report_value_head_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s5 --ratio 0.10
PYTHONPATH=. python -m pytest -q tests/test_report_value_head_ari.py \
  tests/test_value_cluster_evolution.py tests/test_value_medoids.py tests/test_head_mean_value_clusters.py
```
