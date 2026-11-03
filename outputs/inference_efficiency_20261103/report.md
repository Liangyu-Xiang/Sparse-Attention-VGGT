# 推理 GPU 优化与完整验收报告（2026-11-03）

本轮未达到“质量不降低且不慢于 Dense”的要求。原实现继续作为默认路径；优化代码以独立配置提供，未将失败候选替换为正式方法。

## 完整前向速度与显存

同一 heads 序列128帧、RTX 5090 D；每个批次交替执行原实现与候选，各一次预热、三次完整前向取中位数。GPU进程独占监测未发现同卡重叠。两个候选属于不同计时批次，不能把它们的绝对差当作严格配对消融。

| 批次/候选 | 同批原实现 s | 候选 s | 原实现/候选 | 独立 profile 峰值 allocated GiB | Dense 配对加速比 |
|---|---:|---:|---:|---:|---:|
| v7 | 26.36830 | 18.26666 | 1.444× | 17.790 | N/A |
| v8 | 26.99932 | 19.98102 | 1.351× | 17.789 | N/A |

按用户要求，Dense与SAF3R均未重跑。历史 Dense heads128 完整前向为12.2160s，来自20261029批次；两条候选均高于这个历史参考，不能声称达到不慢于Dense的目标。历史值不是本轮配对测量，相对Dense的正式加速比为N/A。Dense作为自身参照为1.00×。

独立profile含事件插桩，其完整耗时不是三次中位数。配对进程的Graph静态池在切换方法后继续存活，原始峰值保留在timing.json中，不能拿共享进程峰值证明独立显存节省。

## 六项质量，全部场景等权宏平均

AUC/F1为百分比，ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离。Dense、SAF3R、reference均复用已有完整质量结果，仅候选重新评测。7Scenes全七场景128帧；ETH3D全11场景14–76帧。

| 数据集 | 方法 | 场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7scenes | dense | 7 | 23.6689 | 84.6389 | 0.153741 | 0.125616 | 0.139678 | 48.0128 |
| 7scenes | saf3r | 7 | 23.9349 | 84.7131 | 0.143263 | 0.127883 | 0.135573 | 47.3154 |
| 7scenes | reference | 7 | 22.1234 | 84.4171 | 0.146727 | 0.124312 | 0.135520 | 45.2177 |
| 7scenes | candidate_v7 | 7 | 22.2558 | 84.3907 | 0.143478 | 0.122613 | 0.133045 | 47.3148 |
| 7scenes | candidate_v8 | 7 | 21.9102 | 84.2025 | 0.144386 | 0.121930 | 0.133158 | 46.8951 |
| eth3d | dense | 11 | 29.1835 | 81.8162 | 0.313193 | 0.848854 | 0.581024 | 62.5731 |
| eth3d | saf3r | 11 | 23.3322 | 80.1747 | 0.303106 | 0.857720 | 0.580413 | 62.8304 |
| eth3d | reference | 11 | 23.9534 | 79.2698 | 0.326859 | 0.875317 | 0.601088 | 56.8682 |
| eth3d | candidate_v7 | 11 | 21.9408 | 77.7391 | 0.404517 | 0.956203 | 0.680360 | 55.5475 |
| eth3d | candidate_v8 | 11 | 21.9385 | 78.2346 | 0.308629 | 0.872234 | 0.590431 | 59.5813 |

7Scenes重建阈值0.05m/下采样0.0078125m；ETH3D阈值0.25m/下采样0.0390625m。SAF3R为本地lazy first-correspondence-head Top-4，不标为论文DINO配置。所有候选与reference的输入路径、帧索引、预处理形状、checkpoint实际路径和完整质量协议逐场景核对一致，见各版本protocol_audit.json。

## 验收结论

- v7 / 7scenes：未通过1%退化门槛的指标为 无；六项相对变化百分比：{"auc03": 0.5984852497219251, "auc30": -0.031230524298031526, "acc": -2.214542420450427, "comp": -1.3669943771590254, "cd": -1.8258153300981794, "f1": 4.637684815986076}。
- v7 / eth3d：未通过1%退化门槛的指标为 auc03, auc30, acc, comp, cd, f1；六项相对变化百分比：{"auc03": -8.402184742431851, "auc30": -1.930913734137607, "acc": 23.758781722602684, "comp": 9.240780329818122, "cd": 13.188072578772303, "f1": -2.322322378653563}。
- v8 / 7scenes：未通过1%退化门槛的指标为 无；六项相对变化百分比：{"auc03": -0.9639319951273873, "auc30": -0.25414706662095776, "acc": -1.595430408200249, "comp": -1.9164391930922675, "cd": -1.742660776928362, "f1": 3.709439072144005}。
- v8 / eth3d：未通过1%退化门槛的指标为 auc03, auc30；六项相对变化百分比：{"auc03": -8.411829222674161, "auc30": -1.305919611275419, "acc": -5.577436363073874, "comp": -0.35221449031505037, "cd": -1.7728974920164786, "f1": 4.770848305608033}。

v8的ETH3D重建四项均优于reference，但位姿AUC@3/AUC@30未通过，因此整体仍不合格。1%门槛也不是数学上零质量变化的证明。没有根据测试真值调整预算、状态、阈值或概率模型，也没有挑选最好重复结果。

## 实现与正确性

- 预算固定20%，状态集合、损失、历史概率估计和概率质量转移公式保持原设定；每层重新处理当前Value、聚类/Rep、历史概率及分配。
- CUDA Graph仅缓存执行图和缓冲区，每层刷新几何数据，每个窗口刷新prior、历史Key、Exact资格；缓存容量和shape桶仅影响执行调度。
- 合并状态损失、gate、支持打包和补偿访存；固定探针索引按输入形状准备一次；范围与预算检查仍执行，合并到每层读取。
- v7在一次选中边QK中同时收集Probe Logit和grid局部归约，避免历史生成再次扫描raw logits求分母，并对Attention作分区softmax合并。FP32小测试误差低于1e-5；BF16 Attention相对L2约0.0004–0.0011，但完整质量仍失败。
- v8保留原串行Attention计算顺序，只将num_stages设为1；FP32、BF16以及真实张量尺寸的输出、grid概率、Probe Logit逐位一致。v8历史分母采用参考扫描内核，因此比v7多约1.5秒诊断开销。
- 新增融合状态内核曾错误假设Exact资格掩码连续。真实路径的掩码来自转置稀疏矩阵乘法并保留列布局，导致v6读取错误。已在内核入口转为连续布局，增加非连续掩码测试；v6失败结果完整保留。
- 修复后ETH3D relief真实序列抽查每层首窗口：23个状态分配窗口的保留Key集合与参考一致，Graph与即时计算集合一致；此范围之外没有宣称逐位等价。
- v7/v8剩余的完整质量差异尚未被唯一定位。v8恢复原Attention顺序后重建指标改善，但位姿仍失败，不能把全部退化归因于分区Attention。融合归约和coarse归一化仍有浮点差异，原实现也含非确定性原子归约；这些是待隔离因素，不是豁免质量门槛的理由。

## 互斥耗时

以下为v8单独profile，只列exclusive时间。其完整前向含插桩；此表用于定位瓶颈，不代替配对中位数。父级/子级不重复相加。

| 操作 | 秒 | 占完整前向 |
|---|---:|---:|
| attention_core_fused_grid | 6.528257 | 31.13% |
| reuse_and_budget | 4.285285 | 20.43% |
| kernel_local_normalizers | 1.475845 | 7.04% |
| attention_wrapper | 0.967277 | 4.61% |
| qkv_rope_prepare | 0.501152 | 2.39% |
| cluster_center_graph | 0.460277 | 2.19% |
| kernel_local_store_local | 0.455691 | 2.17% |
| cluster_local_assignment | 0.433487 | 2.07% |
| residual_fine_probe | 0.362953 | 1.73% |
| coarse_observation | 0.241426 | 1.15% |
| global_attention | 0.218354 | 1.04% |
| cluster_full_assignment | 0.170216 | 0.81% |
| attention_grid_normalize_reduce | 0.135997 | 0.65% |
| m5c_coarse_observation | 0.106320 | 0.51% |
| pooling | 0.084763 | 0.40% |
| local_checks_and_coarse_store | 0.049475 | 0.24% |
| execution_and_evidence | 0.044260 | 0.21% |
| history_summary | 0.042700 | 0.20% |
| output_projection | 0.030602 | 0.15% |
| value_update | 0.027875 | 0.13% |
| kernel_local_uniform | 0.019080 | 0.09% |
| local_eps_compatibility | 0.017215 | 0.08% |
| local_normalizers | 0.017085 | 0.08% |
| cluster_repair_empty | 0.015362 | 0.07% |
| kernel_local_compat | 0.012731 | 0.06% |
| cluster_centers_from_labels | 0.010784 | 0.05% |
| m5c_evidence_and_host_copy | 0.004793 | 0.02% |
| local_probability_headmean_cache | 0.002480 | 0.01% |
| output_dropout | 0.000030 | 0.00% |
| 全局Attention之外的完整前向剩余部分 | 4.249349 | 20.26% |

完整profile 20.971124s；global Attention 16.721775s；exclusive求和误差 0.000000000000s。Graph内部损失、预算、打包、补偿统一计入reuse_and_budget，不再叠加独立内核时间。

独立CUDA活动诊断（v7）中，Sparse Attention内核约6.168s，融合状态内核约2.113s，历史缓存_store_local约0.457s。25908次_choose预算内核合计约0.267s。此活动诊断与上述v8profile不同批次，不能相加；活动区间并集与内核时长总和同时保存，约1.656ms重叠已显式记录。

Nsight Compute硬件计数器读取被系统拒绝（ERR_NVGPUCTRPERM），没有修改主机权限。报告不声称已测得SM占用率或带宽利用率。

## 交付与复现

- 独立配置：configs/evaluation/reuse_efficiency_v6.json（修复后的分区路径，v7源码）和reuse_efficiency_v8.json（串行staging路径）；默认配置不变。v6失败源码与v7修复源码由快照区分，不能只凭相同配置文件名区分这两轮。
- scripts/benchmark_efficiency_full.py：同输入原实现/候选交替完整前向计时，明确不调用Dense。
- scripts/profile_efficiency_candidate.py：分层事件树exclusive计时；scripts/profile_efficiency_kernels.py：独立CUDA活动汇总。
- scripts/run_efficiency_quality.py：冻结源码SHA256，单次完整pose+recon_unposed质量评测，严格GPU进程监测。
- 原始六指标、逐场景CSV、源码差异快照、SHA256、质量协议审核、完整前向计时及显存、失败诊断保存在outputs/inference_efficiency_20261103。
- 共享union bank、连续跨grid tile、寄存器限制、延迟Value读取、streaming写入等微基准未获得足够端到端收益，未纳入候选默认配置。无完整结果的诊断均标N/A，不当作质量评测。
