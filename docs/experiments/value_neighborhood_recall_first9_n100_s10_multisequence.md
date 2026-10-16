# VGGT 100帧、多序列 Neighborhood Recall

本次是长时批处理实验；完成状态以 `outputs/value_medoids_100frames/status.json` 为准，不能把任务启动或Value提取完成当作统计完成。

## 设置

- 5个完整序列：Chess、Fire、Heads、Office、RedKitchen 的 seq-01。
- 每序列第0、10、…、990帧，共100帧、间隔10帧。
- 原始dense VGGT及既有权重 `checkpoints/vggt/model_tracker_fixed_e20.pt`；前9个global层，frame attention按原始网络运行；不应用稀疏patch。
- 每帧1036个patch，每头N=103600，64维原始V；排除每帧5个special tokens；K=1036，恰为1%。
- 逐头独立聚类；逐层先对16个头的64维V取算术平均，再独立聚类。
- 对每个源头，在下一层16个头中选择ARI最大的目标头，固定配对用于所有q，允许多对一。
- q=0、1、2、4、8、16、32、64、128、256、512、1035；q不含自身簇。全簇覆盖q=1035用于检查Recall=100%。
- Pumpkin在目标100帧中缺少22帧，Stairs/seq-02仅500帧，均不满足同一采样设置，未纳入。

## 大输入聚类

N×N的FP32距离矩阵约42.93 GB（39.98 GiB）。原实现会同时创建多个同等大小中间量。
本次新增分块距离与eager PAM，避免多个N×N临时矩阵；以两个固定种子的距离加权初始化，选目标较低解。
每次接受一个降低真实目标值的交换；一轮完整候选扫描没有低于数值容差的改善交换才标记收敛。
维护每个token的前两近medoid，仅对失去前两近medoid的token重新搜索全部K个代表，其他token插入新候选更新前两近距离。
距离矩阵显式对称，FP32距离、FP64目标求和；不抽样token、不量化矩阵、不改变欧氏距离定义。

这与此前10帧实验的BUILD+距离加权初始化、best-single-swap顺序不同；两者均为单交换局部最优，不保证全局最优。
不同帧数的K也不同，比较Recall时应同时查看候选token比例，不能直接把相同q当作相同相对搜索规模。
测量仍然检查下一层代表token是否位于上一层自身簇及q个邻近簇的并集中，而不是检查旧medoid能否直接替代新代表。

实现还保留CPU版FasterPAM以及主存距离矩阵/GPU交换计算作为资源不足时的备选；当前批处理使用GPU驻留距离矩阵。
16项相关测试通过，覆盖分块距离、所有候选交换代价、CPU/GPU及主存路径的局部最优核对、增量前两近距离更新、Recall显式并集与标签置换不变性。
各序列完成拟合后会自动对全部token到全部medoid的直接FP64距离进行独立核验，然后报告Recall。

## 进度与输出

总计5×153=765个选中聚类，各有2次初始化。批处理按序列排队并行运行；已验证收敛的单个聚类可恢复，失败不会写成最终成功报告。

- `outputs/value_medoids_100frames/status.json`：各序列阶段、已完成聚类数、worker PID及更新时间。
- `<scene>_seq01_n100_s10/extract.log`、`cluster.log`、`report.log`：逐序列日志。
- 各序列 `values/layer_X.npy`：原始dense V，按层保存以免反复解压大归档。
- `keep_1%/layer_XX/head_YY.npz`及JSON：逐头聚类与收敛记录。
- `head_mean_1pct_ari/layer_XX/clusters.npz`及JSON：平均V的聚类。
- `neighborhood_recall_1pct/`：逐头配对、逐头Recall、逐层汇总、FP64核验和曲线。
- 所有序列完成后生成 `outputs/value_medoids_100frames/report.md` 和 `all_sequences_layer_summary.csv`。

该批处理只对ARI最佳头对计算Recall，避免重复计算2048个未选定头对；选择最佳头时仍比较每组相邻层的全部16×16个ARI。
选头与评估使用同一输入。不同头的通道并无天然对齐，平均V是按用户指定执行的算术操作。

## 运行与恢复

使用含torch的Python；本次环境是 `/data/mmc_syang/miniconda3/envs/fastvggt/bin/python`。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 python scripts/batch_value_recall_long.py \
  --backend torch-gpu --cluster-gpus 4 5 6 --cpu-workers 3 --cpu-threads 4

# 独立重新生成某个已完成序列的报告
PYTHONPATH=. OPENBLAS_NUM_THREADS=4 python scripts/report_value_neighborhood_recall.py \
  --input-dir outputs/value_medoids_100frames/chess_seq01_n100_s10 \
  --matched-only --qs 0 1 2 4 8 16 32 64 128 256 512 1035
```

GPU编号按空闲资源调整；不要在已有本批处理worker运行时重复启动相同目录的任务。
CPU备选依赖 `kmedoids==0.5.5`（本次临时安装于 `/tmp/saf3r_value_recall_kmedoids`，不改动现有Python环境），GPU路径不依赖它。
CPU算法参考：[kmedoids官方文档](https://python-kmedoids.readthedocs.io/en/latest/)。
