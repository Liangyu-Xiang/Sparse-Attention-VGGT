# M5c 等价公式实现优化：heads128

范围：7Scenes heads 单序列、128帧，不是全七场景质量宏平均。相同checkpoint、输入列表、预处理、20%预算、32-query组、64组窗口、每层聚类、16轮预算和历史/Rep/Exact规则；无Dense或SAF3R重跑。

## 完整前向计时

同一GPU独占租约内依次运行旧清理版和新优化版；各1次预热+3次无插桩完整前向取中位数。初始化/聚类/历史/分配/Attention/全部输出头均在forward内。模型加载及计时外数值检查不计入。

|实现|中位数 s|峰值 allocated GiB|峰值 reserved GiB|相对旧清理版|
|---|---:|---:|---:|---:|
|原清理版|53.63070|19.61151|23.04297|1.0000×|
|M5c融合+GPU缓存|33.10490|19.06346|24.98047|1.6200×|

耗时减少 38.27%。这个加速比的分母是旧清理版，不是Dense。

## 实现变化

- 历史边旧QK、概率恢复、M5c修正和探针求和融合；保留独立头结果后使用原Torch head mean，避免跨head原子累加。
- 校准统计保持FP32、逐窗口留在GPU。消费旧窗口后保存新窗口，不长期同时保存两整层；删除下层不读取的special概率缓存。
- 当前层校准中的exp与grid scatter融合，避免单独落地完整逐边概率张量。
- 第二次Attention QK保留，网格证据/Exact资格继续使用其结果；没有调整预算、状态或聚类。

缓存GPU占用最大 1.52590 GiB；实际总峰值见上表。逐层记录中的history_cache_device是继承的旧文本标签；真实设备以history_cpu_bytes=0及history_calibration_gpu_bytes为准。

## 数值验证

扩展合成测试12例，包含FP32/BF16、尾query、无历史组和全Key观测；校准、概率、历史筛选、4个预算下的状态选择通过。
真实128帧抽查每层首尾窗口，共94项prior/cache检查；prior最大相对L2=7.89602481e-08，pi最大相对L2=7.85268881e-08。
严格概率阈值附近有7条qualified历史候选翻转；Rep移除身份未变。公式等价不等于逐位等价，不能声称完整状态轨迹或GT精度完全不变。完整输出自身也存在重复运行波动。

|输出|原版自身重复相对L2上限|新版自身重复上限|原新版相对L2上限|
|---|---:|---:|---:|
|pose_enc|0.00509855|0.00498490|0.00560835|
|depth|0.01366164|0.01224247|0.01267883|
|depth_conf|0.07145200|0.05811815|0.06078724|
|world_points|0.01373794|0.01311942|0.01366396|
|world_points_conf|0.06479578|0.09172898|0.06496242|

## 新版独立插桩

完整wall=34.29488s；以下是包含式区间，不能互相全部累加；含CPU发射/排队空隙，非纯GPU活跃时间。

|区间|秒|
|---|---:|
|history_probability|3.421337|
|m5c_evidence_statistics|6.390483|
|drop_vector_sum|5.027351|
|sparse_attention_kernel|3.729626|
|attention_mass_second_qk|1.447822|
|mass_transfer|1.406639|
|value_update|1.063103|
|budget_solve4|0.274588|

## 六项质量及Dense速度

|实现|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|同批Dense速度|
|---|---|---|---|---|---|---|---|
|原清理版|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|M5c融合+GPU缓存|N/A|N/A|N/A|N/A|N/A|N/A|N/A|

Implementation/numerical/timing diagnostic only; no pose or recon_unposed GT evaluation; Unchanged Dense/SAF3R not rerun.

源文件归档、原始JSON、输入列表、逐层统计和数值测试分别保存在本目录及outputs/m5c_equivalent_20261102、outputs/m5c_equivalent_real128_20261102。未根据GT选择参数。
