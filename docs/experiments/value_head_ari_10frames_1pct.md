# 10帧输入：前三层相邻层全部头对的Value聚类ARI

## 输入与实验定义

- Chess/seq-01，第0、5、10、15、20、25、30、35、40、45帧，沿用已有n10_s5实验。
- VGGT dense前向，权重`model_tracker_fixed_e20.pt`，只提取前三个global layers（L00–L02）的V。
- 图像输入518×392，每帧28×37=1036个patch，10帧联合共10360个patch。
- 每层16头，各头独立处理原始64维V；不平均头、不对V做L2归一化。
- 保持1%代表点比例，K=round(10360×0.01)=104，实际保留1.003861%。
- 每帧5个special tokens全部排除，不参与Value聚类或ARI。
- PAM两个初始化（BUILD、固定种子距离加权），每轮穷举最佳单交换，最多1000轮，选目标较小解。
- 48组解均达到数值容差内无改善单交换，选中解交换次数49–160；不是全局最优保证。

比较同一批10360个patch的完整簇归属。第1→2层、第2→3层各计算16×16
共256个ARI，总计512个。ARI不依赖簇编号，不需要先做簇匹配，也不假设
相同头编号具有跨层对应关系。结果没有只保留最佳头配对。

## 结果

| 相邻层 | 全256头对平均ARI | 中位数 | 最小值 | 最大值 | 最大值的头对 |
|---|---:|---:|---:|---:|---|
| 第1→2层 | 0.164518 | 0.155819 | 0.061786 | 0.273703 | L00 H11 → L01 H12 |
| 第2→3层 | 0.252156 | 0.254678 | 0.137977 | 0.417038 | L01 H12 → L02 H11 |

每个源头各选下一层最相似头后再平均，ARI分别为0.239695、0.284780；
这是次要统计，与上表“全部256个头对平均”不同。

第1→2层，16个源头的最相似目标头为H12（12个）或H02（4个）。
第2→3层，15个源头选择H11，另一个源头H07选择H08。
这些是聚类划分的相似性，不足以证明固定一对一功能性head对应。

本输入上，第2→3层完整划分的整体关联高于第1→2层，但并非完整划分保持不变。
ARI=1表示相同划分，机会水平约0，可为负，不能将其解释为不变token百分比。
结论仅对应本序列这10帧。两帧实验的K=21，这里K=104，因此跨输入比较
ARI时，帧数和簇数都发生变化，不能视作只改变帧数的受控消融。

## 核验

- 10项相关单元测试通过，包括2帧/10帧frame-major V重排与special token排除。
- 48个头全部10360个归属，经分块直接FP64欧氏差值独立检查，验证为数值容差内最近medoid。
- 核对104个medoid互异、自身归属、簇大小、目标值及优化收敛状态。
- 全512个ARI逐项与sklearn `adjusted_rand_score`核对，最大绝对差5.55e-17。

## 文件和复现

输出根目录`outputs/value_medoids/chess_n10_s5/`，不覆盖此前两帧实验。

- `metadata.json`、`values.npz`：10帧联合前向的输入定义和各层原始V。
- `keep_1%/layer_XX/head_YY.npz`：104个medoids、10360个patch归属、距离、簇大小。
- 各头同名JSON：完整优化目标、初始化、收敛状态。
- `all_head_ari_1pct/index.html`：两张完整16×16带数值热图。
- `all_head_ari_1pct/L00_L01_ari_matrix.csv`、`L01_L02_ari_matrix.csv`：矩阵形式全精度数值。
- `all_head_ari_1pct/all_head_pairs_ari.csv`：512条头对ARI。
- `all_head_ari_1pct/all_adjacent_head_ari.pdf`：两页完整矩阵。
- `all_head_ari_1pct/summary.json`：输入和汇总。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py extract \
  --source-metadata configs/analysis/value_medoid_chess_n10_s5.json \
  --output-dir outputs/value_medoids/chess_n10_s5 --layers 0 1 2
# 两个独立终端并行运行，分别使用空闲GPU1/2：
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s5 --layers 0 1 2 \
  --heads 0 1 2 3 4 5 6 7 --ratios 0.01 --max-swaps 1000
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s5 --layers 0 1 2 \
  --heads 8 9 10 11 12 13 14 15 --ratios 0.01 --max-swaps 1000
PYTHONPATH=. python scripts/report_value_head_ari.py --input-dir outputs/value_medoids/chess_n10_s5
```
