# 50帧、前9个全局层：平均V后选择25%共享Key

## 实验定义

7Scenes按此前评测的序列规则和uniform抽帧方式，每个场景取50帧，预处理保持518×392。每帧1036个patch，合计51800个patch。

前9个全局注意力层，在当前在线前向产生的原始V上先对16个头逐坐标求FP32平均，再用已有完整PAM实现选12950个真实代表点。BUILD与seeded sample两个起点、完整单交换搜索及收敛判断保持不变；PAM是单交换局部最优，并非全局最优保证。

每层所有头共享代表点对应的Key/Value子集，不删除任何Query。patch queries只访问选中的patch keys和全部special keys；special queries保持dense。后15个全局层和全部frame attention不变。PAM距离/优化显式关闭外层BF16 autocast，保持FP32。

## 启动与状态

入口：`scripts/run_value_medoid_sparse.py`。最初在GPU 1串行启动；现已按用户要求改为不同场景分配不同GPU，保留GPU 1正在进行的chess计算，不重启该任务。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/run_value_medoid_sparse.py
```

不要在已有进程运行时重复启动同一输出目录。并行调度入口是`scripts/parallel_value_medoid_sparse.py`，实时总状态以`outputs/value_medoid_sparse_50/parallel_status.json`为准。原`status.json`只代表旧GPU 1进程，不能再用来判断全部并行任务是否完成。

最新范围仅保留用于三方法比较的chess、fire、heads：GPU 1继续chess，GPU 2继续fire，GPU 3继续heads。office、pumpkin、redkitchen进程已按用户要求停止，stairs排队已取消，现有部分产物和日志保留，不删除。GPU 4–6用于上述三个比较场景的重建补算，而不是其余Value-PAM场景。两个入口的默认场景也已收敛到这三个。

旧七场景调度器已停止；新调度器通过`--scenes chess fire heads --adopt-status .../parallel_status.json`接管既有fire、heads进程，不重启或丢弃其进度。GPU 1的完整chess结果落盘后，调度器核对原进程PID和启动标识，再结束原串行进程，防止它继续计算已取消或已分派的其他场景；chess结果及已有进度不会丢失。

各场景单独运行相同程序、相同算法和50帧设置，独立日志及状态保存在`workers/{scene}_gpu{id}.{log,json}`，输出仍统一保存在`{scene}/`下，比较报告无需更换输入路径。调度器为每场景持有进程锁，并为总调度持有独占锁；不在同一GPU放两个任务，不重试OOM任务。某场景显存检查失败或实际OOM时该场景停止，不降低帧数、分辨率或改用近似算法，其他独立场景可继续。

总状态为completed才代表所选三个场景全部有有效结果；finished_with_incomplete_scenes表示所选场景存在失败或停止。已取消的四个场景不再参与完成条件，不会自动重启。新结果出现时自动重新生成三场景VGGT/SAF3R/Value-PAM比较报告，不把缺失结果当作零。

## 已完成的首层显存验证

模型加载并实际推进至第1个全局层后，可用显存为43084677120字节（约40.13 GiB）。单张FP32距离矩阵约10.00 GiB；完整交换搜索主要矩阵峰值约39.98 GiB。

一次完整N=51800、K=12950规模的真实交换搜索冒烟检查已通过，累计PyTorch峰值分配为46.798995 GiB。该检查的临时代表点仅用于测试显存，不用于稀疏attention，也不替代PAM初始化。

随后已开始首层完整BUILD初始化。这只证明首层同规模交换搜索可运行，不保证后续层可运行；也不代表聚类、九层前向或7Scenes评测已完成。后续结果以状态文件及各场景`result.json`为准。

本次尚无准确率或完整推理耗时结果，不能用显存冒烟耗时代替方法运行时间。

## AUC@3°与点云重建补充

对比报告已包含AUC@3°、5°、15°、30°。VGGT与SAF3R的chess/fire/heads三场景50帧位姿评测已完成，AUC@3°宏平均分别为42.5397%与40.9252%；Value-PAM尚待在线前向完成。

原推理程序只保存了位姿，没有保存深度，不能直接用旧位姿文件生成真实预测点云。现已增加两阶段补算入口：`scripts/evaluate_value_sparse_reconstruction.py`。capture阶段在空闲GPU重做深度前向；Value-PAM使用原在线运行保存的9层medoid集合重放，不重新聚类，不修改原算法。所有方法都核对补算位姿与原位姿数值一致，失败则拒绝接收补算结果。

score阶段在CPU使用SevenScenes的`_prep_unposed`和仓库TSDF融合及最近邻指标。使用预测深度和预测位姿，经Umeyama/RANSAC与GT做尺度/刚体对齐，沿用GT深度有效像素mask；不使用GT相机位姿直接融合。TSDF体素及评测下采样7.8125 mm，SDF截断4 cm，各网格采样100万点。GT点云按固定seed=42采样并跨方法共享缓存；空网格或采样失败直接报错，禁止随机点云兜底。

补充指标为Accuracy、Completeness、CD=(Accuracy+Completeness)/2，以及5 cm阈值下Precision、Recall、F-score。距离报告单位cm，比例报告为百分数。保存预测深度、融合网格及点云，便于后续复核。补算/融合耗时不替换原方法在线耗时，也不将重放耗时当成Value-PAM端到端加速。

队列入口`scripts/queue_value_sparse_reconstruction.py`已启动，统一状态在`outputs/compare_value_sparse_50/reconstruction/queue_status.json`。取消非比较场景后，释放的GPU 4–6已用于chess/fire/heads的VGGT及SAF3R深度补算；VGGT深度已捕获，CPU融合评分已开始，SAF3R深度捕获正在继续。CPU评分与GPU捕获分进程运行。若Value-PAM某场景未得到完整结果/9层mask，不产生该场景重建指标，也不重新启动PAM。

最终汇总位于`outputs/compare_value_sparse_50/report.md`，含逐场景和三场景宏平均；重建原始数值在`reconstruction.csv`（距离为m，比例为0–1）。仅全部三个场景完成后计算对应方法的重建宏平均，缺失结果显示待计算，不填零。
