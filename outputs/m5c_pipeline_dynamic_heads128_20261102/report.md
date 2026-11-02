# Drop与统计流水线实现优化：heads128

范围：7Scenes heads 单序列、128帧，不是全七场景质量宏平均。相同checkpoint、输入列表、预处理、20%预算、32-query组、64组窗口、每层聚类、16轮预算和历史/Rep/Exact规则；无Dense或SAF3R重跑。

## 完整前向计时

同批次同一GPU独占测量上一版和新优化版；参考计时复用自本批次已完成阶段；各1次预热+3次无插桩完整前向取中位数。初始化/聚类/历史/分配/Attention/全部输出头均在forward内。模型加载及计时外数值检查不计入。

|实现|中位数 s|峰值 allocated GiB|峰值 reserved GiB|相对上一版M5c融合|
|---|---:|---:|---:|---:|
|上一版M5c融合+GPU缓存|33.03684|19.06634|24.97266|1.0000×|
|新Drop与统计流水线|28.18086|18.66585|24.43555|1.1723×|

耗时减少 14.70%。这个加速比的分母是上一版M5c融合，不是Dense。

## 实现变化

- 按现有簇的成员数分桶，直接求sum(p_i*z_i)的L2范数；不再逐query group建立大型三维向量库。分桶只改变kernel调度，不改变簇归属、Rep或状态。
- 窗口内QK计数保持在GPU，合法性检查及计数同步集中到层末；检查未删除，异常输入仍在返回该层结果前报错。
- 桶长度COUNT为不参与JIT特化的运行时参数，避免簇成员数量轻微变化造成反复编译；M5c概率公式和GPU FP32校准缓存保持上一版。
- 第二次Attention QK保留，网格证据/Exact资格继续使用其结果；没有调整预算、状态或聚类。

缓存GPU占用最大 1.52590 GiB；实际总峰值见上表。新版逐层记录标明GPU校准缓存；history_cpu_bytes=0。

## 数值验证

扩展合成测试12例，包含FP32/BF16、尾query、无历史组和全Key观测；校准、概率、历史筛选、4个预算下的状态选择通过。
真实128帧抽查每层首尾窗口，共140项prior/cache/Drop检查；prior最大相对L2=0，pi最大相对L2=6.26918535e-08。
Drop最大相对L2=5.26293533e-08。严格概率阈值附近有0条qualified历史候选翻转；Rep移除身份未变。公式等价不等于逐位等价，不能声称完整状态轨迹或GT精度完全不变。完整输出自身也存在重复运行波动。

|输出|原版自身重复相对L2上限|新版自身重复上限|原新版相对L2上限|
|---|---:|---:|---:|
|pose_enc|0.00616354|0.00550851|0.00493173|
|depth|0.01323773|0.01314905|0.01407973|
|depth_conf|0.05697537|0.06569140|0.05811019|
|world_points|0.01454426|0.01342568|0.01414617|
|world_points_conf|0.06626015|0.08432990|0.09201908|

## 新版独立插桩

完整wall=28.95197s；以下是包含式区间，不能互相全部累加；含CPU发射/排队空隙，非纯GPU活跃时间。

|区间|秒|
|---|---:|
|history_probability|3.324712|
|m5c_evidence_statistics|6.155582|
|drop_vector_sum|0.553461|
|sparse_attention_kernel|3.612487|
|attention_mass_second_qk|1.447135|
|mass_transfer|1.394961|
|value_update|1.068063|
|budget_solve4|0.274979|

## 六项质量及Dense速度

|实现|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|同批Dense速度|
|---|---|---|---|---|---|---|---|
|上一版M5c融合|N/A|N/A|N/A|N/A|N/A|N/A|N/A|
|新Drop与统计流水线|N/A|N/A|N/A|N/A|N/A|N/A|N/A|

Implementation/numerical/timing diagnostic only; no pose or recon_unposed GT evaluation; Unchanged Dense/SAF3R not rerun.

原始JSON、逐层统计、源码归档及数值测试保存在本目录、outputs/m5c_pipeline_dynamic_20261102和outputs/m5c_pipeline_dynamic_real128_20261102。旧constexpr COUNT候选因反复编译中止，原始诊断保留在outputs/m5c_pipeline_heads128_20261102；其完整参考计时复用于本次比较，避免重跑未改动参考。不使用中止候选的速度作为最终结果。未根据GT选择参数。
