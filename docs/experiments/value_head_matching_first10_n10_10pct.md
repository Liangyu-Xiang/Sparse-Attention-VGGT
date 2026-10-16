# 前10层、10帧、10%代表点：最佳前层头匹配与头平均V的ARI

## 输入和不变的实验标准

继续原Chess/seq-01第0、5、10、15、20、25、30、35、40、45帧；原图640×480，
VGGT输入518×392，每帧28×37个patch，联合10360个patch。提取前10个global
attention layers，自然层编号1–10对应代码L00–L09，而不是运行到代码L10。
每层16头，V为原始64维向量，不归一化，排除每帧5个special tokens。

10%指medoid比例：K=1036，所有10360个patch仍参与独立聚类和ARI计算。
每份PAM拟合分别使用BUILD与固定种子距离加权初始化，穷举最佳单交换，
最多3000次，取目标较小的解。170份划分（160个独立头+10份头平均V）均收敛。
保证的是数值容差下无改善单交换，不是全局组合最优。

前10层重新dense前向提取的V中，前三层与旧10帧缓存逐元素完全相同；
因此安全复用旧前三层的48份逐头拟合和3份平均V拟合。原始前三层目录未覆盖。
新增结果使用单独目录`outputs/value_medoids/chess_n10_s5_layers_01_10/`。

## 两种统计，严格区分

1. **后层每头选择最佳前层头**：A[g,h]是前层g和后层h的完整簇划分ARI。
   每个后层头独立选择`argmax_g A[g,h]`，再对16个后层头的分数平均。
   不按相同头编号，不是每个前层头去选择后层头，不施加一对一约束，允许复用前层头。
2. **先平均16头原始V再聚类**：每层同一patch的16个64维V逐坐标FP32算术平均，
   再独立做K=1036的PAM，最后比较两层10360个patch的完整归属。
   不是平均簇编号或256个头对ARI，也不是平均attention概率或投影后的attention输出。

ARI不依赖簇编号，不需要先对齐簇标签。1是相同划分，约0是机会水平，可为负，
不能将其视为未改变patch的百分比。“最优头”仅指本输入上ARI最高，不证明稳定的功能性对应。

## 前10层的全部相邻层结果

| 相邻自然层 | 后层逐头最佳前层匹配：平均ARI | 先平均16头V再聚类：ARI |
|---|---:|---:|
| 1→2 | 0.240558 | 0.301342 |
| 2→3 | 0.433264 | 0.396125 |
| 3→4 | 0.569979 | 0.594791 |
| 4→5 | 0.577269 | 0.635126 |
| 5→6 | 0.583268 | 0.630772 |
| 6→7 | 0.613789 | 0.650110 |
| 7→8 | 0.605667 | 0.657995 |
| 8→9 | 0.563628 | 0.622079 |
| 9→10 | 0.579083 | 0.616828 |

本输入上，第3→4层开始的关联明显高于前两对，但并非随深度单调增加。
逐头最佳匹配均值最高为第6→7层0.613789；平均V划分最高为第7→8层0.657995。

平均V在9组中的8组高于逐头最佳匹配均值，但两项是不同表示、不同划分的描述性
统计。各头独立学习的特征基底没有对齐，原始V平均具有基底依赖性，可能抵消特征；
不能直接视作多头共识或稀疏注意力/下游质量提升。结论限于本序列这10帧。

## 验证与产物

- 160份独立头拟合全部收敛，选中解执行369–1094次交换。
- 每层16头的全部patch用FP64 norm/GEMM距离验证为容差内最近medoid；
  已分配medoid的距离另用直接FP64向量差核对。逐层证据在`validation/layer_XX.json`。
- 10份平均V拟合全部通过FP64距离检查；重用拟合再次确认原始V头平均和归属一致。
- 全9×16×16=2304个ARI独立逐项与sklearn核对；9项平均V层间ARI也独立核对。
- 核对144个后层头的匹配全部为对应列的argmax，无按编号匹配或错误方向。
- 19项相关单元测试通过，包括第10层端点、稀疏联合计数、匹配方向/复用、
  缓存完整性、平均V断点续算、分片报告保护、PAM交换与特殊token排除。

结果根目录下：

- `metadata.json`、`values.npz`：前10层dense V与完整输入信息。
- `keep_10%/layer_00`至`layer_09`：160个头的medoids、全部patch归属、距离、优化细节。
- `all_head_ari_10pct/index.html`：9个16×16矩阵，行是前层头、列是后层头；每列最优加粗并标红框。
- `all_head_ari_10pct/all_head_pairs_ari.csv`：2304个完整ARI。
- `all_head_ari_10pct/best_previous_heads.csv`：144个后层头的最佳前层匹配与分数。
- `all_head_ari_10pct/all_adjacent_head_ari.pdf`：全部9组矩阵PDF。
- `head_mean_10pct_ari/index.html`、`adjacent_layers.csv`：9组两种结果的并列表。
- `head_mean_10pct_ari/adjacent_layers_ari.png`：两种ARI随层深变化的折线图。
- `head_mean_10pct_ari/mean_values.npz`：10层平均后的V。
- `head_mean_10pct_ari/layer_XX/`：10份平均V拟合的完整归属和收敛细节。

## 复现与断点续算

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py extract \
  --source-metadata configs/analysis/value_medoid_chess_n10_s5.json \
  --output-dir outputs/value_medoids/chess_n10_s5_layers_01_10 \
  --layers 0 1 2 3 4 5 6 7 8 9
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s5_layers_01_10 \
  --layers 0 1 2 3 4 5 6 7 8 9 --ratios 0.10 --max-swaps 3000 --skip-existing
# 实际运行可用 --heads 将16头分成互不重叠的分片，并行使用空闲GPU。
PYTHONPATH=. python scripts/report_value_head_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s5_layers_01_10 --ratio 0.10 --num-layers 10
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/analyze_head_mean_value_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s5_layers_01_10 --ratio 0.10 --num-layers 10 --resume
# 平均V也可用 --fit-layers 和 --cluster-only 拆分拟合；所有层齐备后再用上面命令汇总。
```

`--skip-existing`仅跳过完整、收敛、簇大小/归属/目标一致的缓存；缺失、损坏、
未收敛的缓存自动重算。平均V分片不会发布不完整的共享报告。
