# 前三层：先平均16个头的原始Value，再做1%聚类

## 定义

沿用 Chess/seq-01 第0、20帧、前三个 VGGT global layers L00–L02 的缓存V。
每层原始V为 `[16,2072,64]`，特殊token已排除。在同一个patch位置上，对
16个头的64维V逐坐标取算术平均，FP32累加，得到 `[2072,64]`。

`mean_V[l,i,d] = sum_h V[l,h,i,d] / 16`

不对V进行L2归一化，不平均原有簇编号或聚类统计。每层在平均后的原始
V空间独立运行PAM k-medoids，K=21（实际保留1.0135%）。BUILD和固定种子
距离加权初始化各运行一次，穷举最佳单交换，选择目标更低的解。三层均达到
容差下无改善单交换的局部最优；不是组合问题的全局最优保证。

重要局限：各头学习的特征基底未对齐，逐坐标平均具有基底依赖性，可能出现
抵消。这里不是平均attention权重、经输出投影的多头输出，或平均各头距离/
共聚类矩阵；不能把平均V聚类直接解释为多头共识聚类。

## 逐层结果

| 自然层编号 / 代码层 | medoids数 | 帧0 / 帧20代表点数 | 最小/最大簇大小 | 平均V到最近medoid的欧氏距离 |
|---|---:|---:|---:|---:|
| 第1层 / L00 | 21 | 11 / 10 | 54 / 196 | 0.111236 |
| 第2层 / L01 | 21 | 7 / 14 | 43 / 198 | 0.399338 |
| 第3层 / L02 | 21 | 11 / 10 | 48 / 247 | 0.443843 |

不同层的V尺度不同，上表原始距离不能直接解释为跨层代表性质量的优劣。
三个层的全部2072个patch各自归到最近平均V medoid；这不是空间最近邻划分，
不是物体类别。同簇可以跨帧且不连续。63张逐簇原图高亮图展示每个代表点
覆盖哪些patch，总计6216条逐patch归属。

## 相邻层变化

相同位置patch的完整簇划分用ARI/NMI比较，不按簇编号对应。这里每层只有
一份平均V划分，不再进行16×16选头。展示颜色按最大交集的一对一簇标签
重排逐层对齐，仅用于阅读；簇编号仍为每层自己的C01–C21。

| 相邻层 | ARI | NMI | 最大交集簇标签对齐后同归属比例 |
|---|---:|---:|---:|
| 第1→2层 | 0.223566 | 0.470174 | 33.40% |
| 第2→3层 | 0.352890 | 0.580276 | 46.33% |

ARI=1表示完整划分相同，约0为机会水平，并非相同token比例。每帧内部随机
置换目标层位置500次的ARI均值分别为-0.000021、0.000009；随机分布的
2.5%–97.5%分位范围分别为[-0.001413,0.001338]、[-0.001453,0.001838]。

对这一输入，平均V得到的区域划分仍存在跨层关联，第2→3层比第1→2层更稳定，
但大量patch发生重新分组。与此前“每个源头在16目标头中选最佳”是不同的
统计对象，不能直接凭两组ARI大小断言平均操作更优。结论限于这两帧。

## 文件与复现

输出目录 `outputs/value_medoids/chess_f000000_f000020/head_mean_1pct/`：

- `index.html`：逐层结果、完整分类和每个medoid覆盖区域。
- `layerwise_aligned_regions.png`：三层对照总图，颜色按最大交集逐层对齐。
- `layer_XX/cluster_map.png`：本层原始簇编号/颜色、代表点及完整分类。
- `layer_XX/cluster_01.jpg`至`cluster_21.jpg`：逐代表点的全部原图成员区域。
- `layer_XX/clusters.npz`、`clustering.json`：聚类归属和优化收敛细节。
- `mean_values.npz`：三层平均后的V，不覆盖原始缓存。
- `all_patch_classes.csv`：6216条patch归属；`cluster_regions.csv`：63簇统计。
- `layer_summary.csv`、`adjacent_layers.csv`：逐层和相邻层指标。
- `L00_to_L01/`、`L01_to_L02/`：完整21×21转移计数、区域对照、全部簇流量图及单簇拆分图解。
- `cluster_flows.csv`：318条非零簇间转移。

```bash
CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. \
  python scripts/analyze_head_mean_value_clusters.py
PYTHONPATH=. python -m pytest -q tests/test_head_mean_value_clusters.py \
  tests/test_value_cluster_evolution.py tests/test_value_medoids.py
```

实验使用空闲GPU1，重用原始dense V，不重跑或修改模型参数。9项测试通过。
已独立核对FP32头平均、原始平均V到medoid的直接FP64距离、各层21个代表点
自身归属，以及两个转移矩阵的总数/行和/列和。
