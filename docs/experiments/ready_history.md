# 上一层准备历史条件概率

本实现从冻结的 28.18 秒 repaired RMS Drop + historical states 版本生成隔离副本。20%预算、32-query组、64组窗口、Value聚类、状态集合、历史Rep规则、Exact资格、16轮预算及质量转移保持原设置。canonical grid仍对整个grid内的Key池化，未改为排除保留Key后的均值。

## 执行流程

1. 首次稀疏Attention在同一次QK计算中导出4个探针的原始、未补偿logit。query总数不变，不再另做校准专用的探针QK。
2. 本层结束前完成原网格校准，准备条件概率 `p(edge) / P(grid)` 与未观测成员的条件概率、coarse grid概率。分母、特殊Token和全grid近似沿用原公式。logsumexp按行融合归约，避免完整逐边exp临时矩阵；网格累加按原Key顺序分段归约，不排序或变更支持集。
3. 保留每个head及每个探针的值。边条件概率先做FP16量化，0~1正FP16的最高两位恒零，再按14位打包。14位打包对FP16编码逐位无损，但FP32→FP16本身是明确的数值近似。
4. GPU边缓存容量固定8GiB，按完整窗口对齐；尾窗口使用pinned CPU缓存。两个GPU缓冲区通过独立复制流预取/回写，事件保护旧概率读取与原地覆盖。全程只保留一层逻辑缓存，逐窗口消费后覆盖。最后一层不准备无人使用的下一层缓存。
5. 下一层只做当前canonical grid观测和M5c概率更新，不再计算上一层QK，也不缓存上一层完整Q/K。原Attention第二次QK仍保留，用于原grid统计和Exact资格。

概率按query–Key边保存，同一Key面对不同query组有不同概率；token特征只保存一次。heads128的原FP32逐边缓存约为 `4164*16*4*26009*4` 字节，即25.82GiB，而非一个每token特征数组。FP16/u14后的逻辑边缓存约11.3GiB；其中GPU约7.99GiB、CPU约3.31GiB，另有GPU grid统计及双缓冲。

## 使用

```bash
python scripts/setup_ready_history.py --destination /tmp/zdrop_ready_history_local
```

默认源为 `/tmp/zdrop_m5c_pipeline_dynamic_20261102`，先校验冻结manifest；目标必须不存在。生成的model源码、内核及配置已与最终计时副本逐字节核对。固定输入数据、checkpoint及Python环境仍需保留。实际运行配置为 `configs/evaluation/reuse_zdrop_history_states.json`；独立副本中的 `scripts/evaluate_zdrop_history_states.py` 可用于后续质量评估。运行时使用：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python /tmp/zdrop_ready_history_local/scripts/evaluate_zdrop_history_states.py \
  --dataset 7scenes --method zdrop_history_states --scene heads --frames 128 \
  --config /tmp/zdrop_ready_history_local/configs/evaluation/reuse_zdrop_history_states.json \
  --output-dir /absolute/path/to/a/fresh/output
```

源代码在 `saf3r/patch/reuse_ready_history.py`、`saf3r/triton/reuse_ready_history.py`、`saf3r/triton/reuse_explore_av_ready.py`。GPU驻留上限默认8GiB，允许通过安装后state的 `ready_gpu_limit_bytes` 调整资源布局；本轮没有按GT调参。压力下释放inactive allocator页的开销计入前向。

## 已完成验证与实测

- 8个FP32/BF16及FP32/FP16缓存组合案例；无历史、全部观测、尾query、奇数Key数。
- 全部0~1正FP16编码逐位打包解包；4种logsumexp长度，包含全负无穷行。
- 三层、GPU/CPU混合、双缓冲强制溢出验证。
- 真实heads128每层首尾窗口各抽一组，共92项producer/consumer核对。导出logit一致；条件概率相对L2≤2.147e-4，prior≤4.632e-5。FP16量化造成20条候选阈值翻转，极小系数有下溢；不声称逐位等价或GT质量不下降。

RTX5090 D，heads128，一次预热后三次完整前向中位数23.54043秒；峰值allocated25.79308GiB、reserved30.15039GiB，pinned CPU缓存3.30982GiB。GPU算法缓存约9.351GiB，含驻留边缓存、grid统计和双缓冲。冷前向27.11944秒。旧版28.18086秒只作历史参考，并非同批配对加速验收；Dense/SAF3R未重跑，Dense加速比N/A。

这是数值和速度诊断，未评测pose/recon_unposed，AUC@3、AUC@30、ACC、Comp、CD、F1均为N/A。未做七场景质量宏平均，也没有根据测试GT选择缓存或预算参数。整体GPU显存比旧重算版上升，不能把边缓存压缩说成整体显存下降。

原始JSON、日志、源码归档/哈希、逐层CSV和报告：`outputs/ready_history_v5_heads128_20261102/report.md`；真实输入核对：`outputs/ready_history_v5_real128_20261102`；数值/强制溢出测试：`outputs/ready_history_20261102`。早期全GPU缓存原型OOM均保留，不计作成功质量评测。
