# 第12–14个全局层的代表性Value原图可视化

输入继续使用此前的 Chess/seq-01 第0、20帧，原图640×480，
VGGT处理后为518×392、每帧28×37个patch。
本实验的“第12–14层”按自然顺序计数，对应代码索引L11、L12、L13。
每层分别分析16个头（H00–H15），每头两帧联合聚类。

从dense VGGT的QKV投影提取原始V，排除特殊token，
使用欧氏距离的PAM k-medoids，保留10%（207/2072个真实medoids）。
每头分别从BUILD与固定随机种子的距离加权初始化开始优化，最多各交换1000次，
取最终目标值更低的解。48组最终解均满足数值容差内无改善单次交换；
这不是全局最优保证。模型与聚类方法均沿用此前的10%实验。

输出目录：`outputs/value_medoids/chess_f000000_f000020_layers_12_14/`。

- `overview_layer_11_keep_10%.jpg`：第12层16头总览。
- `overview_layer_12_keep_10%.jpg`：第13层16头总览。
- `overview_layer_13_keep_10%.jpg`：第14层16头总览。
- `index.html`：48张逐头原图标注图的中文浏览页。
- `all_heads_value_medoids.pdf`：48页完整图册。
- `metrics.csv`：每头聚类指标、收敛状态与每帧代表点数量。
- `medoid_locations.csv`：9936个代表点的patch坐标、原图坐标与簇大小。
- `values.npz`、`metadata.json`：V缓存与可复现输入信息。

黄色点标记真实medoid的patch中心，青框标记其原图范围。
两帧联合选点，每帧点数由聚类结果决定。

复现命令（建议依次提取、并行聚类、最后渲染）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py extract --layers 11 12 13 --output-dir outputs/value_medoids/chess_f000000_f000020_layers_12_14
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 11 --ratios 0.10 --max-swaps 1000 --output-dir outputs/value_medoids/chess_f000000_f000020_layers_12_14
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 12 --ratios 0.10 --max-swaps 1000 --output-dir outputs/value_medoids/chess_f000000_f000020_layers_12_14
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 13 --ratios 0.10 --max-swaps 1000 --output-dir outputs/value_medoids/chess_f000000_f000020_layers_12_14
PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py render --layers 11 12 13 --ratios 0.10 --output-dir outputs/value_medoids/chess_f000000_f000020_layers_12_14
```

提取/渲染脚本现支持全部24个全局层的显式索引。
原来的前三层默认参数不变。PAM的3项独立数值测试通过。
