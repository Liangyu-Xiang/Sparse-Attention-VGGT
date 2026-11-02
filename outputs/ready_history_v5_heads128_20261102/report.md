# 上一层准备历史条件概率：heads128

本轮是7Scenes heads单序列、128帧实现/数值/速度诊断，不是七场景质量宏平均。使用原checkpoint、固定输入列表、预处理、20%预算、32-query分组、64组窗口、每层聚类和16轮预算。保留所有head/4探针，canonical grid继续使用整个grid的池化Key；未改为剩余成员均值。

首次稀疏Attention直接导出所需探针原始logit，本层结束前校准并缓存条件概率；下一层仅使用当前grid观测和缓存，重建旧QK及校准专用探针QK均为零。原Attention第二次QK仍保留，用于原grid/Exact观测，不声称所有重复QK已消除。

缓存先做FP16量化，再对0~1正FP16数去掉恒零的两位，14位打包本身逐位无损。单层原地覆盖、GPU缓存上限8GiB、尾窗口pinned CPU缓存、双缓冲异步预取/回写；窗口流事件保护旧数据读取及原地覆盖，不同时保留两整层。最后一层不再准备无人使用的历史缓存。

|范围|完整前向中位数 s|峰值 allocated GiB|峰值 reserved GiB|
|---|---:|---:|---:|
|旧版历史记录|28.18086|18.66585|24.43555|
|新版本次实测|23.54043|25.79308|30.15039|

当前1次预热+3次完整前向，包含初始化、分配、聚类、DMA等待、校准、重排及输出头。冷前向27.11944s。旧时间仅作历史参考，不能当同批配对验收；Dense速度N/A（未重跑）。GPU独占监控无同卡外部重叠。

GPU算法缓存 9.35083 GiB，其中GPU边缓存 7.98697 GiB，另含grid统计与双缓冲。CPU固定内存缓存 3.30982 GiB；这不是总CPU峰值。相对于旧重算方案，整体显存上升；相对于原样FP32逐边缓存约26GiB，压缩后的逻辑边缓存约11.3GiB，分布在GPU与CPU。

## 验证

原子/归约FP32路径、FP16缓存路径、无历史、全观测、尾query、全部正FP16编码打包解包及四种logsumexp长度测试通过；三层强制CPU溢出测试通过，prior相对L2=6.6822075e-05。
真实输入每层首尾窗口各抽一个query group，共92项producer/consumer检查。导出logit相同；条件概率相对L2≤0.00021467468，prior≤4.6314137e-05，簇质量pi≤2.0563752e-05。
FP16相对FP32会量化，抽查有20条历史候选阈值翻转，另有323088个极小逐头/探针系数下溢为零。不能声称逐位等价或最终质量不降。缓存生命周期/顺序校验、budget上限及源哈希校验通过。

## 已有新版profile

独立插桩wall=24.47011s。下列是包含式主流区间；DMA在复制流上重叠执行，不能额外把DMA持续时间加到这些区间上。主流等待和最终全设备同步计入完整前向。CUDA区间含CPU发射/排队空隙。

|区间|秒|
|---|---:|
|sparse_attention_kernel|4.176191|
|m5c_evidence_statistics|2.901069|
|history_probability|1.226467|
|attention_mass_second_qk|1.462740|
|ready_allocator_trim|0.099352|

|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|
|---|---|---|---|---|---|
|N/A|N/A|N/A|N/A|N/A|N/A|

Numerical/timing diagnostic only; pose/recon_unposed quality not evaluated. No GT parameter tuning.

纯GPU缓存原型的OOM原始记录保存在outputs/ready_history_heads128_20261102以及ready_history_v2/v3/v4_heads128_20261102；它们不是完成的质量评测，也没有被用作最终速度。最终模型源码归档与manifest位于本目录。复现脚本为scripts/setup_ready_history.py，计时脚本为scripts/run_ready_history_v5_128.py。
