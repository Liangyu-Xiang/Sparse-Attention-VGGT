# VGGT 前9个global层：Neighborhood Recall

完整结果表和曲线见 [实验报告](../../outputs/value_medoids/chess_n10_s10_layers_00_08/neighborhood_recall_1pct/report.md)。

## 实验定义

- 原始 dense VGGT，仓库权重 `checkpoints/vggt/model_tracker_fixed_e20.pt`，严格加载官方模型结构，不应用 SAF3R patch。
- Chess/seq-01，第0、10、20、30、40、50、60、70、80、90帧；输入518×392，每帧1036个patch。
- 前9个global attention层（零基L00–L08）；各层之前的frame attention按原模型正常执行。
- 每头10360个64维原始V，排除每帧5个special tokens，不做V归一化；K=round(10360×1%)=104。
- 各头独立PAM：BUILD与固定种子距离加权两种初始化，每步穷举所有单交换，最多1000步，选目标较低解。所有选中解均达到数值容差内单交换局部最优，不保证全局最优。
- 逐层分析先对每个token的16个头的原始V算术平均，得到64维向量，再按相同K独立聚类。
- 跨层匹配：每个源头选择下一层ARI最高的目标头，允许多对一；固定头匹配后计算所有q。匹配与评估使用同一输入，结果不是独立测试集上的对应关系。

## 指标

上一层medoid间的原始V欧氏距离决定簇的q个近邻，使用有向q近邻，另加自身簇。
对每个token，取下一层全局PAM分配的medoid **token位置**，检查该位置在上一层属于的簇是否在候选簇集合中。
成功比例为Recall@q，不需要对齐两层簇标签。下一层PAM是充分优化的全局搜索基准，而不是已证明的全局最优。

这里候选范围是q+1个簇中的全部token，不能将Recall解释为只比较q+1个旧medoid就能达到相同结果，也不能据此直接宣称加速倍数。
`candidate_token_fraction`给出候选token比例；`independent_assignment_recall`是独立置换下一层代表分配后的期望Recall。

## 结果解释

前两次层间变化的局部性较弱。从第3→4层开始，逐头匹配Recall@4为83.91%–88.61%，Recall@8为89.85%–93.43%；
先平均V再聚类的Recall@4为87.51%–90.74%，Recall@8为92.06%–95.47%。
在全部8组相邻层上，逐头匹配后平均Recall@4为79.55%，平均V方式为82.92%；Recall@8分别为86.35%、88.64%。

q=4时，逐头匹配的候选区域平均包含总token的5.22%–5.42%；其独立置换期望Recall为5.13%–5.36%。
q=103覆盖全部104簇，所有头对Recall均为100%，作为实现检查。
这些是单一序列、单次10帧输入的测量，说明后续几层更适合局部候选搜索，不能直接推广到所有输入或任务精度。

## 核验与文件

153个选中聚类（144逐头+9平均V）全部收敛。所有1585080个token归属用直接FP64坐标差值重新计算到104个medoid的距离并检查。
其中1个token与FP64严格argmin不同，距离差仅2.87e-7，在FP32数值容差内；其他全部相同。
同时核对medoid互异、自身归属、簇大小、目标值和终止交换收益。2048个头对的Recall曲线均单调，q=103均为1。
12项相关测试通过，包括对显式候选簇并集的逐token核对、目标簇标签置换不变性、重复向量、自簇强制保留、PAM和头平均。

输出目录 `outputs/value_medoids/chess_n10_s10_layers_00_08/`：

- `values.npz`、`metadata.json`：原始V及输入/模型定义。
- `keep_1%/layer_XX/head_YY.npz`及JSON：逐头medoid、全部归属、距离及优化记录。
- `head_mean_1pct_ari/layer_XX/`：平均V的独立聚类及优化记录。
- `neighborhood_recall_1pct/head_matching.csv`：128条固定跨层头匹配及ARI。
- `neighborhood_recall_1pct/matched_heads.csv`：选定头对的全部q结果，共1152条。
- `neighborhood_recall_1pct/all_head_pairs.csv`：2048个头对×9个q，共18432条。
- `neighborhood_recall_1pct/layer_summary.csv`：逐头匹配后平均、平均V聚类、同编号头和全部头对四种汇总。
- `neighborhood_recall_1pct/clustering_checks.csv`：153个聚类的收敛和FP64距离核验。
- `neighborhood_recall_1pct/recall_curves.png`、PDF、`summary.json`、`report.md`：曲线及报告。

## 复现

使用已安装torch的环境；本次为 `/data/mmc_syang/miniconda3/envs/fastvggt/bin/python`。GPU编号可根据空闲情况调整。

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py extract \
  --source-metadata configs/analysis/value_medoid_chess_n10_s10.json \
  --output-dir outputs/value_medoids/chess_n10_s10_layers_00_08 --layers 0 1 2 3 4 5 6 7 8
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/visualize_vggt_value_medoids.py cluster \
  --output-dir outputs/value_medoids/chess_n10_s10_layers_00_08 --layers 0 1 2 3 4 5 6 7 8 \
  --ratios .01 --max-swaps 1000 --skip-existing
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. OPENBLAS_NUM_THREADS=4 python scripts/analyze_head_mean_value_ari.py \
  --input-dir outputs/value_medoids/chess_n10_s10_layers_00_08 --ratio .01 \
  --num-layers 9 --max-swaps 1000 --cluster-only --resume
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 python scripts/report_value_neighborhood_recall.py \
  --input-dir outputs/value_medoids/chess_n10_s10_layers_00_08
PYTHONPATH=. python -m pytest -q tests/test_value_neighborhood_recall.py \
  tests/test_value_medoids.py tests/test_head_mean_value_ari.py tests/test_head_mean_value_clusters.py
```
