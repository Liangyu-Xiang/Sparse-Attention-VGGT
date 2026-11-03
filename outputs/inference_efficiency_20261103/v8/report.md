# 推理优化工程实验报告（2026-11-03）

结论：候选 v8 暂不替换默认实现。完整前向明显加快，但质量门槛未全部通过，且未达到不慢于 Dense 的要求。

同批 heads 128 帧、RTX 5090 D 完整前向：参考 26.9993 s → 候选 19.9810 s，参考/候选 1.351×，耗时减少 25.99%。一次预热、三次完整前向中位数，严格监测同卡进程，无重叠。

Dense、SAF3R 未重跑。相对 Dense 的正式配对加速比为 N/A；历史 Dense heads128 为 12.2160 s（20261029），候选 19.9810 s 仍高于该参考，不能声称达到速度目标。ETH3D质量运行只有一次预热后一次前向，不能当作三次中位数。

## 六项质量指标

AUC 和 F1 为百分比；ACC/Comp/CD 为米。所有行均为本地实测，Dense/SAF3R/参考复用原始结果；候选仅评测一次完整数据集。

| 数据集 | 方法 | 场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Dense 加速比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7scenes | dense | 7 | 23.6689 | 84.6389 | 0.153741 | 0.125616 | 0.139678 | 48.0128 | 1.00× |
| 7scenes | saf3r | 7 | 23.9349 | 84.7131 | 0.143263 | 0.127883 | 0.135573 | 47.3154 | N/A |
| 7scenes | reference | 7 | 22.1234 | 84.4171 | 0.146727 | 0.124312 | 0.135520 | 45.2177 | N/A |
| 7scenes | candidate_v8 | 7 | 21.9102 | 84.2025 | 0.144386 | 0.121930 | 0.133158 | 46.8951 | N/A |
| eth3d | dense | 11 | 29.1835 | 81.8162 | 0.313193 | 0.848854 | 0.581024 | 62.5731 | 1.00× |
| eth3d | saf3r | 11 | 23.3322 | 80.1747 | 0.303106 | 0.857720 | 0.580413 | 62.8304 | N/A |
| eth3d | reference | 11 | 23.9534 | 79.2698 | 0.326859 | 0.875317 | 0.601088 | 56.8682 | N/A |
| eth3d | candidate_v8 | 11 | 21.9385 | 78.2346 | 0.308629 | 0.872234 | 0.590431 | 59.5813 | N/A |

CD=(ACC+Comp)/2，非平方距离。7Scenes 全七场景128帧；ETH3D 全11场景14–76帧，各场景等权宏平均，二者不得混为128帧测速。7Scenes阈值0.05m、下采样0.0078125m；ETH3D阈值0.25m、下采样0.0390625m。SAF3R 为本地 lazy first-correspondence-head Top-4。完整输入列表、checkpoint实际路径和协议一致性见 protocol_audit.json。

## 质量门槛

- 7scenes：六项相对变化 {"auc03": -0.9639319951273873, "auc30": -0.25414706662095776, "acc": -1.595430408200249, "comp": -1.9164391930922675, "cd": -1.742660776928362, "f1": 3.709439072144005}；未通过1%门槛：无。
- eth3d：六项相对变化 {"auc03": -8.411829222674161, "auc30": -1.305919611275419, "acc": -5.577436363073874, "comp": -0.35221449031505037, "cd": -1.7728974920164786, "f1": 4.770848305608033}；未通过1%门槛：auc03, auc30。

这些结果没有用于继续调预算或公式；所有预算、状态、概率估计和分层设置均保持冻结配置。数学等价的融合仍改变浮点归约顺序和 BF16 中间舍入；小规模数值测试通过不能代替最终几何质量验收。

## 互斥时间分解

单独 profile 完整前向 20.9711s，峰值 allocated 17.789GiB。下表只列 exclusive 时间，父子事件不重复相加。总 global attention 16.721775s，互斥项求和误差 0.000000000s。

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
| 全局 Attention 之外的剩余完整前向 | 4.249349 | 20.26% |

graph 内部状态/预算/打包/补偿步骤未再拆开计时，统一归入 reuse_and_budget，避免与 graph replay 重复统计。attention_wrapper 为排序、索引和分配等包装开销的 exclusive 时间。

## 实现变化与边界

- 原配置继续走参考路径。独立 opt-in 配置开启分区稀疏 Attention、原始 Probe 证据顺带归约、融合状态损失与 mask 操作、形状探针索引缓存和 CUDA Graph 执行缓存。
- 每层重新生成当前 Value 聚类、中心/Rep、概率与分配；Graph 缓存执行图，每次刷新当前数据，不冻结上层状态。
- 原始历史概率、特殊 Token、逻辑 Key 顺序、补偿公式、预算和状态集合均保留。分区 softmax 和并行归约的数学公式相同，但浮点执行顺序有差异，候选未获得质量等价认证。
- 正式 Attention 不重算第二次选中边 QK；在第一次 QK 中保存 Probe Logit 与 grid 局部归约。历史缓存仍为 head-mean、原压缩格式。
- 共享 union bank、跨 grid 连续打包、改变寄存器上限、延迟 Value 读取等已做微基准；没有端到端有效收益的变种未纳入 v8。
- Nsight Compute 硬件计数器不可用（ERR_NVGPUCTRPERM），未修改主机权限。使用 CUDA Event 和 kernel 编译信息；不可伪称已测 SM/带宽利用率。

## 显存与测量限制

独立 candidate profile 峰值 allocated 17.789 GiB；配对计时中两个路径共享进程且 CUDA Graph 静态池持续存活，峰值不能作为两个独立进程的显存节省结论。逐次 peak 已保存在原始 JSON。质量运行逐场景独立进程的峰值见 per_scene.csv；reserved 与 allocated 不同。

完整数据与验证原始 JSON 另存本目录；源快照及 SHA256 保存在原实验目录。未经通过质量门槛和速度验收，不应将本候选标为正式替换。
