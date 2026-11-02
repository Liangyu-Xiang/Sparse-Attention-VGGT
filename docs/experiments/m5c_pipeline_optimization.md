# Drop 与统计流水线优化

当前实现是修正 RMS Drop + 历史状态方案的独立优化版。保持 20% 预算、原簇数、32-query 分组、64 组窗口、每层 Value 聚类、History/Rep/Exact 规则、16 轮预算求解及概率质量转移；不改变第二次 Attention QK。

## 改动

1. Drop 仍为 `||sum_i p_i z_i||_2`。直接读取簇成员计算加权向量范数，不再建立 `[query group, cluster, value dimension]` 的大向量库。按成员数分桶仅用于内核调度，不改变任何 token 的簇归属或 Rep 身份。全程 FP32，不使用均值近似。
2. M5c 的窗口级 QK 计数保留在 GPU；合法性检查和统计同步集中到层末。检查仍保留，非法输入会在该层返回前报错。
3. 桶长度 COUNT 使用不参与 JIT 特化的运行时参数，避免层间簇成员数轻微变化触发反复编译。早期 constexpr COUNT 候选已中止，保留诊断，不使用其耗时作为最终结果。

## 使用

```bash
python scripts/setup_m5c_pipeline.py --destination /tmp/zdrop_m5c_pipeline_local
```

源目录默认为 `/tmp/zdrop_m5c_equivalent_20261102`，先按上一版冻结 manifest 校验；目标目录必须不存在。生成源码已与本次计时内核、forward 和配置逐字节核对。数据、checkpoint 和项目 Python 环境仍需保留。源码改动在 `saf3r/triton/reuse_drop_moment.py`、`saf3r/patch/reuse_m5c_pipeline.py`，forward 接线由 setup 脚本生成；其他实验分支的工作区文件不被覆盖。未来评估器源码归档包含新增模块。

## 验证与结果

7Scenes heads，128 帧，同批次同一 RTX5090 D；每个实现 1 次预热、3 次无插桩完整前向取中位数。上一版已完成计时复用于修正后的候选，未重跑未改变的参考。最终候选的 GPU lease 检测到同卡外部任务重叠为 false。

|实现|完整前向 s|峰值 allocated GiB|峰值 reserved GiB|相对上一版|
|---|---:|---:|---:|---:|
|上一版 M5c 融合 + GPU 缓存|33.03684|19.06634|24.97266|1.0000×|
|新 Drop 与统计流水线|28.18086|18.66585|24.43555|1.1723×|

完整前向耗时减少 14.70%。新版独立插桩中 Drop 区间为 0.55346 秒；上一轮历史诊断为 5.02735 秒（阶段参考，非本批次旧版插桩）。

12 个 FP32/BF16 合成案例通过概率、缓存、历史筛选和四种预算状态选择测试。真实输入各层首尾窗口共 140 项 prior/cache/Drop 检查：prior 相同，pi 最大相对 L2 约 6.27e-8，Drop 最大相对 L2 约 5.26e-8，抽查中的历史候选筛选差异为零。这不是逐位全模型或最终质量保证；原方法自身也存在重复运行浮点波动。

本次仅为实现、数值和速度诊断，AUC@3、AUC@30、ACC、Comp、CD、F1 全部 N/A，原因是未做 pose/recon_unposed GT 质量验收。Dense/SAF3R 未重跑，同批 Dense 相对速度 N/A。没有使用 GT 调参；不是七场景宏平均。

原始报告、计时、逐层 CSV、源码归档及哈希：`outputs/m5c_pipeline_dynamic_heads128_20261102/report.md`。最终数值/微基准：`outputs/m5c_pipeline_dynamic_20261102`；真实输入影子核对：`outputs/m5c_pipeline_dynamic_real128_20261102`。编译参数错误候选及中止原因：`outputs/m5c_pipeline_heads128_20261102/interruption.json`。

## 上传分支

本分支的 `saf3r/patch/reuse_residual_exploration.py` 已直接接入最终优化实现，可以使用 `configs/evaluation/reuse_zdrop_history_states.json` 配合 `scripts/evaluate_zdrop_history_states.py`；setup 脚本用于从旧冻结目录重新生成隔离副本。基准脚本保留本地 `/tmp` 路径和固定输出目录，异机复现前需配置数据及 checkpoint 路径并选择新输出目录。归档中的源码为原始计时版本；本分支评估器额外补齐新增模块的源码归档，仅影响记录。
