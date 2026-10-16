# 前三层每头 Value 的 k-medoids 与原图位置

输入沿用此前全局注意力矩阵可视化的 `Chess/seq-01` 第 0、20 帧。
读取原有 metadata 中的图像路径，并使用相同的 VGGT 图像预处理和
`model_tracker_fixed_e20.pt` 权重。原图为 640×480，输入为 518×392，
每帧 patch 网格为 28×37；两帧合计 2072 个 patch values。

分析 dense VGGT 的前 3 个 global attention layers（代码 L00–L02），
每层独立处理 16 个头（H00–H15）。从各层 attention 输入的 QKV 投影中
提取 V，不对 V 做 L2 归一化，也不包含每帧的 5 个特殊 token。
两帧 values 共同聚类，不预先规定每帧各分到多少 medoids。

目标为最小化每个 value 到最近 medoid 的欧氏距离之和：

`min_{|S|=K} sum_i min_{j in S} ||v_i - v_j||_2`。

保留比例 1% 对应 K=21（实际 1.0135%），5% 对应 K=104（实际 5.0193%），
扩展后的 10% 对应 K=207（实际 9.9903%）。
每个 medoid 都是输入中真实的 value，不是 k-means 的合成中心。
BUILD 初始化和固定随机种子的距离加权初始化分别运行 PAM。
每轮计算所有插入/删除组合的精确目标差值，执行最佳单次交换，
最终选择两次运行中目标更低的结果；1% / 5% 最多各执行 200 次交换，
10% 最多各执行 1000 次交换。
浮点容差为 `max(1e-6, 1e-7 * initial_objective)`。
最终选出的 144 组解均通过无改善单次交换检查；这只保证该容差下的
单交换局部最优，不保证组合问题的全局最优。

实验在 GPU 5、6 执行；dense 模型使用 BF16 autocast，距离和交换评估
使用 FP32，目标汇总使用 FP64。提取 V 时禁用 TF32，结果可独立缓存重用。

可复现入口：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py extract
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 0 2
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster --layers 1
PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py render
```

输出位于 `outputs/value_medoids/chess_f000000_f000020/`：

- `index.html`：中文浏览页，可点击每头原图叠加图。
- `all_heads_value_medoids.pdf`：144 页，每页展示一个层/头/比例下的两帧。
- `overview_layer_XX_keep_Y%.jpg`：每层 16 头的总览，共 9 张。
- `metrics.csv`：144 组解的误差、收敛状态、交换次数和每帧 medoid 数量。
- `medoid_locations.csv`：15936 个 medoids 的原图坐标、patch 坐标和簇大小。
- `keep_Y%/layer_XX/head_XX.npz`：medoid 索引、每个 value 的簇分配及距离。
- `values.npz`、`metadata.json`：前三层每头的原始 V 和输入定义。

黄色点表示 medoid 的 patch 中心，青框表示该 patch 在原图中的范围。
位置通过反演 VGGT resize / center crop 映射回原图。
这些图展示 value 空间中的代表性，不表示注意力权重大小或几何对应置信度。

验证：`tests/test_value_medoids.py` 的 3 项测试通过，包括对全部交换的
独立穷举核对、目标严格递减及单交换局部最优检查、相同向量退化情况。
