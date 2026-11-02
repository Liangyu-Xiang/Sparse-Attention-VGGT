# Grid局部归一化历史证据：技术与实验报告

远程基点：908ae1fc43f1b54874e7ae5ae2c4cd3514655bb6。独立分支codex/grid-local-history-20261102。保留默认reference；新配置configs/evaluation/reuse_grid_local_history.json。

## 实际调用链与数学定义

ReuseResidualExploration.forward → 原observe（逐头Coarse/Special归一化不变）→ reuse_single_qk_grid.weighted_attention（正式AV及原始Probe QK导出不变）→ reuse_grid_local_history.save_evidence → Grid局部LogSumExp → 条件概率/Head-mean/FP16-u14缓存 → 原build_prior。

c=exp(raw-logD)，u=exp(grid_logit-logD)。Head在局部归一化后才消去，不平均Q/K或Logit。Special始终参与正式Attention。第10步没有QK，也没有调用selected_probe_logits。复用Attention已经计算的稳定排序与offsets，支持任意合法逻辑Key顺序，没有另付排序成本。

EPS兼容：原corrected_grid.clamp_min(1e-30)在极小Grid质量时阻止全局分母完全消去。为保持旧定义，额外用小型Grid分母和已有Special Logit计算logZ，再乘min(1,exp(logD-logZ-logEPS))。不构造逐边Dense概率，也不扫描RAW做全局LogSumExp。未删除observe或build_prior中的Special依赖。

## Tensor与内核审计

典型RAW：[64,16,4,26009] FP32，426131456字节（406.39MiB）；Grid：[64,16,4,512] FP32；Special：[64,16,4,640] FP32。原observe的Grid/Special为非连续head-major view；新路径显式连续化，这些复制包含在计时中。RAW保持逻辑顺序，order/offset只用于局部分母的Gather。详细shape/stride见kernel audit原始JSON。

|实现|GPU Kernel事件数/窗口|临时峰值增量MiB|边缓存MiB/64组|
|---|---:|---:|---:|
|reference|55|64.576|11.112|
|local_packed|11|34.016|11.112|
|local_fp16|11|34.016|12.700|

Kernel审计使用实际heads128尺寸的合成合法支持集；内核活跃时间与CUDA事件区间分开保存。事件数包含所有CUDA kernel（含零填充），CPU launch API计数可能因Memset/图层实现不同而不同。不是端到端计时。

## 正确性

20组校验覆盖1/3/16 heads，FP32、packed与普通FP16缓存，极端Logit、无已选Key、全选、仅一个成员的Grid、Padding、末尾Probe重复权重及非连续观测Tensor。另有真实BF16 Attention执行测试，确认新增布局元数据不改变Attention输出，Special和无补偿原始Probe Logit正确，覆盖末尾15行Query。逐边条件概率、未观测概率、每Grid质量守恒、下一层prior及pi均通过阈值。FP32算法误差与缓存量化误差单独保存。极端合成数据中全选Grid的未观测概率无实际成员对应，数值可能很大，此时必须同时查看相对误差，不能只看最大绝对差。

真实64-group窗口固定相同RAW、Mask、QKV和Grid：
- L0：旧生产3.9301ms，新生产2.7129ms；缓存relative L2=9.78e-06，uniform最大绝对差=1.86e-09。
- L12：旧生产1.7513ms，新生产1.3483ms；缓存relative L2=8.91e-06，uniform最大绝对差=1.16e-09。

完整24层支持集指纹、边数变化及抽样Attention输出差异见rollout_comparison.json。不声称Mask逐位一致：微小量化差异和原聚类的浮点归约可能通过离散决策传播；原实现自身重复稳定性单独诊断。没有把输出变化隐藏为零误差。

## 独占同卡配对时间

heads128、相同checkpoint/输入/环境；严格context监控，其他进程出现则终止。一个方法预热一次，三个方法交替三次取中位数；分段另跑一次插桩前向。插桩有Host发射间隙，不能把GPU Event区间说成纯内核活跃时间。

|实现|完整Forward中位数s|Global Attention s|第9步s|第10步s|第9+10步s|构建及维护s|峰值allocated GiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|reference|26.829269|23.985799|10.547137|2.565543|13.112680|13.938885|17.223868|
|local_packed|26.033921|23.119988|10.591173|2.020422|12.611595|13.023994|17.209738|
|local_fp16|25.965056|23.584934|10.538149|2.014604|12.552753|13.570249|17.305655|

内核细分见paired_timing.csv：条件概率、Head-mean和边缓存编码融合，不能分别独立计时或重复累加。旧全局RAW LogSumExp已删除；新局部normalizer与小型EPS兼容归约分别列出。

## 两数据集质量（等场景宏平均）

仅运行grid_local packed的质量；原reference、Dense和SAF3R复用完整历史结果。质量运行1预热+1次输出，不以该次耗时进行速度验收；上表配对计时独立采用3次中位数。未基于测试质量调参。

|数据集|方法|场景数|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Dense配对加速|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|7scenes|dense|7|23.668916|84.638873|0.153741|0.125616|0.139678|48.012766|1.00x|
|7scenes|saf3r|7|23.934899|84.713102|0.143263|0.127883|0.135573|47.315421|N/A|
|7scenes|reference|7|22.641896|84.438566|0.144104|0.122980|0.133542|46.129941|N/A|
|7scenes|grid_local|7|22.123406|84.417065|0.146727|0.124312|0.135520|45.217742|N/A|
|eth3d|dense|11|29.183465|81.816212|0.313193|0.848854|0.581024|62.573136|1.00x|
|eth3d|saf3r|11|23.332202|80.174741|0.303106|0.857720|0.580413|62.830390|N/A|
|eth3d|reference|11|22.344804|78.418954|0.355829|0.894302|0.625066|54.918515|N/A|
|eth3d|grid_local|11|23.953378|79.269777|0.326859|0.875317|0.601088|56.868204|N/A|

AUC/F1为百分比，距离为米，CD=(ACC+Comp)/2（未平方）。7Scenes阈值0.05m、降采样0.0078125m；ETH3D阈值0.25m、降采样0.0390625m。SAF3R为本地lazy first-correspondence Top-4，非论文DINO Top-K；完整profile见引用result.json。7Scenes全7场景各128帧，ETH3D全11场景实际14–76帧。Dense相对加速N/A：未获要求重跑Dense，没有本轮配对分母。

完整审计通过：True；缺失场景：[]。

## 验收结论与数值稳定性

速度收益成立，但不能宣布保精度优化成功。7Scenes的ACC、Comp、CD和F1均有超过1%的相对退化，AUC@3也下降超过1%；ETH3D六项均改善。保留本轮结果，不据此调整参数或挑选重复质量结果。默认reference不变。

|数据集|AUC@3 相对变化%|AUC@30 相对变化%|ACC 相对变化%|Comp 相对变化%|CD 相对变化%|F1 相对变化%|
|---|---:|---:|---:|---:|---:|---:|
|7scenes|-2.289958|-0.025464|+1.820131|+1.083100|+1.480763|-1.977455|
|eth3d|+7.198874|+1.084972|-8.141667|-2.122885|-3.836033|+3.550148|

原方法重复稳定性诊断（固定输入，未重跑GT质量）：
- reference_repeat：支持集指纹变化91611/99936组，24层首32个Query抽样Attention输出最大relative L2=0.174633。
- local_packed：支持集指纹变化91624/99936组，24层首32个Query抽样Attention输出最大relative L2=0.357343。

原实现自身重复推理也存在离散Mask漂移，因此不能把历史批次质量差异全部归因于局部归一化；本轮也不据此豁免质量风险。新旧固定输入概率对照与完整推理传播误差是两个不同层次的验证。普通FP16的L0差异也不能归因于尚未消费的历史缓存。

## 保留的限制与失败记录

最初真实窗口测试因忽略Grid/Special非连续stride失败，结果保留在grid_local_real_window_20261102，已由v2回归覆盖；该失败版不计入性能结论。首批质量作业的部分归档阶段因远程分支缺少baseline_source_hashes引用的旧配置而失败；按原SHA补齐依赖后，仅重跑失败场景，原失败记录保留。初次合成数值测试不足以替代真实布局验证。

本次保留406MiB RAW导出与两遍读取，尚未把局部分母直接融合进Attention。没有为形式上的单遍融合改变Attention的Key顺序、BF16累加或算法含义。
