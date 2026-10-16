# Global / Frozen / Neighbor-Restricted Value distortion

本实验遵循用户提供的三种方案，retention改为10%，不复用此前1%的划分与头匹配。
小输入为Chess/seq-01第0、10、…、90帧；前9个global层；每头N=10360，K=1036。
100帧扩展复用已提取的5个完整序列Value，N=103600、K=10360；排在既有1% Recall批处理之后，避免争用其GPU。

## 已完成的10帧结果

完整要求表、逐头和平均V结果见 [报告](../../outputs/value_medoids/chess_n10_s10_layers_00_08/distortion_comparison_10pct/report.md)。
从第3→4层至第8→9层，逐头匹配的Frozen Gap平均9.37%，两轮Local q4为5.32%、q8为4.27%；先平均V的Frozen Gap平均5.89%，Local q4为3.49%、q8为2.89%。
这里的平均是6组transition的Gap算术平均，不是把不同层的原始D合并后计算比值。
前两组差距仍大：1→2层平均V的Frozen Gap=65.14%，两轮q4=29.08%、q8=21.29%；2→3层分别为24.88%、15.43%、12.25%。

第二轮有额外改善：后6组逐头q8 Gap均值从第一轮5.06%降至第二轮4.27%，平均V从3.37%降至2.89%。
153个Global全部通过独立FP64距离核验；2个token与FP64严格最近medoid不同，但最大距离差仅2.85e-7，在FP32容差内。
136个配对均按要求保存Frozen及q4/q8的两轮结果，锚点、原始邻域和簇数约束全部核对。

## 对比定义

三种方案都用当前层原始V的平均欧氏距离（不是平方距离）作为distortion，特殊token不参与。

1. Global：当前层独立从头全局PAM，不使用上一层划分进行初始化；已收敛的缓存可以复用。每个缓存保留其算法、初始化及交换收敛记录。
2. Frozen：固定上一层全局划分的全部成员，用当前层V在每个簇内穷举所有真实成员，选取最小距离和的medoid。
3. Local：继承上一层全局medoid的token位置，用当前层对应V作为初始中心。q=4/8的有向近邻图只用上一层medoid的V欧氏距离构建，另加自身簇；每个token始终使用其原始上一层标签对应的候选簇。同步重新分配全部token后，再对所有新簇求精确medoid。严格执行2轮，分别保存第1、2轮结果。

Local中上一层medoid token始终固定在其原簇作为成员锚点，保证K不变；允许簇的新medoid换成其他成员。
锚点在第2轮即使更接近另一簇也不迁移，不能将“当前medoid自归属”误当成固定上一层锚点。
每对相邻层的源划分均为该层独立Global结果；这是单步转移实验，不包含跨多层递归Local带来的累计误差。

逐头分析对本次10%划分计算全部16×16个ARI，每个源头选下一层ARI最大的目标头（允许多对一）；固定配对后运行三种方案。
逐层分析先平均16头的原始64维V，再独立执行三种方案。不能混用逐头与平均V的distortion量纲或把两种结果直接合并。
逐头主表先平均16个配对的D，再计算相对平均Global D的增量；CSV另存逐头百分比Gap的均值以免混淆。

## Global参考与精度

小输入复用本轮已完成的独立PAM结果；新增计算采用两种固定随机种子的FasterPAM，并继续用FastPAM1穷举单交换至局部最优，再额外用1次FastPAM1扫描验证无改善交换。
部分先完成的缓存使用BUILD/距离加权best-swap PAM或距离加权eager PAM；每个缓存都是独立从头全局搜索且有收敛证据，具体算法写入global_fit_checks.csv。
100帧参考采用两种固定种子的距离加权eager PAM，分块距离与交换代价，终止轮完整候选扫描无改善交换。
这些均不保证数学全局最优。负Gap不截断；更大的q在固定2轮后也不保证一定更优。
Frozen和Local簇内medoid用直接FP64欧氏距离穷举，避免“离均值最近成员”替代真正medoid。

## 核验

- 所有Global的归属、距离独立用分块直接FP64坐标差检查，并核对K、medoid唯一、自归属、簇大小和目标值。
- Local每轮核对原始邻域约束、固定源锚点、非空簇与K不变。
- 核对assignment及medoid update分别不增加本次局部目标；保存每轮目标与迁移数量。
- 单元测试包含冻结划分精确medoid、q=0退化、重复向量、固定两轮快照、真正会迁移的旧锚点被锁定、输出均值而非总和、头汇总Gap公式，以及增量最近两medoid和穷举交换对照。

## 输出与运行

- 总状态：`outputs/value_distortion/status.json`；日志：同目录`logs/`。
- 10帧报告：`outputs/value_medoids/chess_n10_s10_layers_00_08/distortion_comparison_10pct/report.md`。
- 100帧各序列：`outputs/value_medoids_100frames/<scene>_seq01_n100_s10/distortion_comparison_10pct/`。
- 100帧全部完成后：`outputs/value_medoids_100frames/report_distortion_10pct.md`与`all_sequences_distortion_10pct.csv`。
- 每个报告目录：`layer_summary.csv`为要求的宽表与第1/2轮指标，`per_head.csv`为128个逐头配对，`head_matching.csv`为10%下新匹配，`global_fit_checks.csv`为独立核验，PNG/PDF为Gap曲线。
- 每个case保留Frozen及q4/q8两轮的完整token归属、medoid、距离、簇大小、邻居图和迭代历史。

使用含torch的环境，本次为`/data/mmc_syang/miniconda3/envs/fastvggt/bin/python`。
CPU备选依赖`kmedoids==0.5.5`，临时安装位置`/tmp/saf3r_value_recall_kmedoids`，不改动已有Python环境。
算法来源：[kmedoids官方文档](https://python-kmedoids.readthedocs.io/en/latest/)。

```bash
PYTHONPATH=/tmp/saf3r_value_recall_kmedoids:. OPENBLAS_NUM_THREADS=2 python scripts/batch_value_distortion.py
# 仅重建已完成Global缓存的三种方案比较
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 python scripts/compare_value_distortion.py \
  --input-dir outputs/value_medoids/chess_n10_s10_layers_00_08 --ratio .10
```

不要在同目录已有worker运行时再次启动同一批处理。CPU路径用于10帧小输入，100帧批次等待此前Recall任务全部完成后使用GPU4/5/6。
