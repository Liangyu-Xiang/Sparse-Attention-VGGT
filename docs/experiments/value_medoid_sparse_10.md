# 10帧三场景：VGGT / SAF3R / Value-PAM

按用户最新要求，已停止50帧Value-PAM及其调度/等待队列，原有50帧结果和部分产物保留，不与本次混用。新实验仅包括chess、fire、heads。

## 固定设置

- 每个序列uniform采样10帧，VGGT crop后518×392，每帧1036个patch，合计10360个。
- Value-PAM前9个global层先对16个头的原始V做FP32均值，再独立选25%（2590个）真实medoids，所有头共享对应Key/Value；全部Query及special tokens保持原约定。
- 仍使用原完整FP32距离矩阵和PAM，BUILD与seeded sample两个起点，完整最佳单交换搜索。未改成分块近似或随机替代，收敛条件与3000次交换上限未改变。
- 新增可选进度回调，仅记录初始化计数、交换次数与当前目标值，不改变任何选点运算；单元测试核对有/无回调产生相同medoids和目标历史。
- 重新执行10帧VGGT、SAF3R基线，不复用50帧指标。三种方法使用共享`inputs.json`，包含AUC@3°/5°/15°/30°、在线前向耗时、峰值显存，以及相同recon_unposed点云重建评测。

## 调度与产物

GPU 1、2、3分别运行chess、fire、heads的Value-PAM。GPU 2、3在此之前已完成本次10帧的VGGT与SAF3R基线。其他GPU上的外部任务未触碰。重建补算等待GPU空闲，且遵守PAM调度器的GPU预留，不会在模型尚在CPU加载时误将同卡当作空闲。

- Value总状态：`outputs/value_medoid_sparse_10/parallel_status.json`
- Value首个任务日志：`outputs/value_medoid_sparse_10/chess.log`
- 后两任务日志：`outputs/value_medoid_sparse_10/workers/{scene}_gpu{id}.log`
- 每层进度：`outputs/value_medoid_sparse_10/{scene}/layer_{index}/progress.json`
- 共享输入和汇总：`outputs/compare_value_sparse_10/inputs.json`、`report.md`
- 重建补算状态：`outputs/compare_value_sparse_10/reconstruction/queue_status.json`

重建仍使用预测深度、预测位姿、仓库Umeyama/RANSAC对齐与GT深度有效mask、TSDF体素7.8125 mm、SDF截断4 cm、100万点采样、5 cm的P/R/F-score阈值。Value-PAM深度通过原已完成9层mask重放生成，不重跑聚类；重放输出位姿必须与原实验一致。距离报告为cm，比例为百分数。重建补算耗时不替换在线PAM耗时。

报告、GPU调度及重建队列均显式传入本次10帧输出路径，不再写入50帧目录。全部三个场景完整后才计算对应方法的宏平均；未完成数据不填零。
