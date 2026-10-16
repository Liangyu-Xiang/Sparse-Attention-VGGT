# 第1–2层最高ARI头对：各自的原图Value聚类结果

在已有10帧、10%代表点实验的第1层×第2层全部16×16头对中，最高为：

- 第1层H03（代码L00 H03）与第2层H02（代码L01 H02）。
- ARI=0.3227300155，按全部10帧的10360个patch归属计算；不是仅按预览的两帧。
- 两头各自独立聚为1036簇，所有patch都参与；代表点是真实V medoid。
- 第1层H03簇大小3–27、中位数9；第2层H02为2–36、中位数9。

输出根目录：
`outputs/value_medoids/chess_n10_s5_layers_01_10/best_pair_L00_H03_L01_H02_10pct/`。

每个头的`L00_H03/`、`L01_H02/`目录内：

- `preview_clusters.png`：第0、20帧预览，彩色表示完整Value簇归属，白点为medoid位置。
- `all_frames_clusters.jpg`：全部10帧的分类总图。
- `all_frames_medoids.jpg`、`preview_medoids.png`：在原图上仅显示代表点。
- `frame_XXXXXX_clusters.png`、`_medoids.png`、`_labels.png`：逐帧分类、仅代表点、纯标签图。
- `cluster_XXXX_members.jpg`：本头四个最大簇的全部原图成员区域，亮区为成员，其他区域暗化。
- `patch_labels.npz`：1起始簇编号`cluster_ids[10,28,37]`，用uint16保存，保留完整1036类。

根目录`index.html`汇总两头结果，`both_heads_all_frames.pdf`包含全部20页，
`all_patch_classes.csv`共20720条patch归属，`cluster_regions.csv`共2072簇统计。

色彩跨两头用Hungarian最大成员交集进行一对一重排，仅用于显示；不假定同编号簇
对应，不改变任何聚类归属、medoid或ARI。显示颜色索引为
`palette_rgb[cluster_to_colour[cluster_id-1]]`。1036类颜色不保证肉眼均可区分，
准确归属以标签数组/CSV为准；颜色不是语义类别或attention权重。

重要实现细节：渲染的整数标签图使用int32最近邻缩放，不能用uint8，否则超过255的
簇编号会截断。本次验证两头标签完整覆盖1–1036，且与原聚类逐项相等。
测试覆盖高编号保留、1036个唯一可复现颜色、非同编号的颜色对齐。

```bash
PYTHONPATH=. python scripts/visualize_best_value_head_pair.py
PYTHONPATH=. python -m pytest -q tests/test_best_value_head_pair.py
```
