# Direct Probability History 精细 GPU Profiling

参考分支 `codex/direct-probability-history-20261104`，远程/本地均为 `d10fd3e6`。正式算法、概率、四状态、16轮预算、mask和Attention Kernel全部保持原样。唯一正式入口改动为默认关闭的`--profile-detail`；其余为诊断脚本。`static_validation.json`验证saf3r/configs源文件与参考字节一致，并验证删除插桩with后各函数AST与原函数相同。

仅7Scenes heads、128帧，RTX5090D物理GPU6，四Probe、20%保留，原checkpoint/预处理不变。基线一次预热+三次正式前向：[15.007992336992174, 15.153267511166632, 15.112961857579648]，中位数 **15.112962s**；新CLI未启用插桩的三次中位数 **15.120932s**。GPU严格单进程，launcher无CUDA上下文，监控每0.5秒检测外部PID；相关成功pass的monitor均无重叠。未运行Dense/SAF3R，配对Dense加速比N/A。

下表是低开销CUDA Events阶段计时，整前向 **15.352257s**，相对同脚本未插桩中位数开销 **1.53%**。A含共享模型的调度残差，B只含AV，C包括本方法额外操作。三项互斥且相加等于该次前向；不能把它们冒充未插桩15.12s的严格分解。

| category | gpu_phase_seconds | includes_children |
| --- | --- | --- |
| A | 4.918914 | False |
| B | 2.729628 | False |
| C | 7.703715 | False |

最细逐语句/GPU Graph时间戳pass为30.695831s，开销103.00%；初次编译/带标记Graph构建32.872505s。其operations/per_window/per_layer表只用于子阶段定位，不能用总量排名代表正常速度。单独不含标记Kernel的算子trace时间跨度15.919343s、GPU kernel self总和15.232113s；GPU self不含CPU等待，与阶段耗时和wall不是同一视图，不跨pass加总。

| module | phase | self | mean_window_seconds | count | memory | extra | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 共享 VGGT 计算 | 4.918914 | N/A | N/A | N/A | N/A：硬件计数器无权限 | 否 | A：共享模型工作及其未插桩调度间隙 |
| Direct Sparse AV | 2.729628 | 2.727465 | 0.001723 | 1584 | N/A：硬件计数器无权限 | B 核心 | 读K/V、QK、Online Softmax、AV、输出和LSE融合 |
| History Probability Export | 0.886249 | 0.885981 | 0.000560 | 1584 | N/A：硬件计数器无权限 | 是 C3 | 独立四Probe QK，融合LSE读取、Debias、LogMeanExp、输出 |
| Value / Cluster 更新 | 1.089881 | N/A | 0.045412 | N/A | N/A：硬件计数器无权限 | 是 C1 | 包含下列邻簇图与局部分配，不与其重复相加 |
| Coarse Observation | 0.165668 | 0.012379 | 0.000105 | N/A | N/A：硬件计数器无权限 | 是 C2 | _m5c仅归一化kernel，完整阶段还含FP32 QK、gather等 |
| History Calibration / Cache | 0.400021 | 0.181248 | 0.000264 | 4554 | N/A：硬件计数器无权限 | 是 C4 | 只L0–22写缓存；不含后面单列的Probe→Grid摘要 |
| Probe→Grid 摘要 | 0.414349 | N/A | 0.000262 | N/A | N/A：硬件计数器无权限 | 是 C4 | 与缓存校准是不同用途、不同调用，24层均执行 |
| Prior / M5c Beta | 0.386293 | 0.029691 | 0.000254 | N/A | N/A：硬件计数器无权限 | 是 C5 | _predict仅覆盖历史边回填，阶段还含完整prior及pi |
| Drop / Rep / History / Exact Loss | N/A | 2.097527 | 0.001382 | 9252 | N/A：硬件计数器无权限 | 是 C6 | 真实融合总耗时，组件微基准不可相加 |
| History Gate | N/A | 0.119036 | 0.000078 | 1520 | N/A：硬件计数器无权限 | 是 C6 | 包含历史资格、Rep重合、eligible_count |
| Budget | N/A | 0.283358 | 0.000187 | 50160 | N/A：硬件计数器无权限 | 是 C7 | 16轮+最终选择；与Graph Replay重叠 |
| Mask Packing | N/A | 0.378540 | 0.000249 | 6080 | N/A：硬件计数器无权限 | 是 C8 | flags+全Patch prefix+emit+validate的kernel下界；未含所有周边小算子 |
| Mass Compensation | N/A | 0.254143 | 0.000167 | 3040 | N/A：硬件计数器无权限 | 是 C9 | Owner读取/概率转移/atomic与weight融合；验证另有小算子 |
| Diagnostics / Graph Replay | 3.587362 | N/A | 0.002363 | N/A | N/A：硬件计数器无权限 | 是 C10 | Replay阶段包含C6–C9，不参与顶层加总；纯Diagnostics没有无污染独立self时间 |

上表Replay包含Loss/Budget/Packing/Transfer/Diagnostics，不能再加一次。Value更新包含邻簇图与局部分配。Graph首次初始化独立于稳定replay，细项见graph_overhead.csv；细粒度原记录为重标记Graph pass，完整低开销窗口输入copy/常量copy/replay/record copy见coarse_phase_per_layer.csv。

额外成本较高的十项如下（阶段/Kernel Self混合视图已明确标记，各项不包含其他表中父模块；未将全部CPU调度强行归属某一算法操作）：

| operation | seconds | kind | source | reuse | fusion | risk |
| --- | --- | --- | --- | --- | --- | --- |
| 融合四状态损失 | 2.097527 | GPU kernel self | saf3r/triton/reuse_fused_states.py::_states | 同一层距离/Value跨Group共享已在全局内存中存在，可研究CTA内多Group复用 | 已融合Drop/Rep/History/Owner/Exact；进一步需保持归约/owner同分规则 | 中：浮点与离散同分敏感 |
| 四Probe概率导出 | 0.886249 | GPU phase | saf3r/triton/reuse_direct_sparse_attention.py::_probe | 与正式AV重复QK；LSE已复用 | 可研究从AV导出必要统计；不能改变log域Debias/Head归约 | 高：寄存器/访存/分母及误差 |
| 32邻簇图 | 0.449932 | GPU phase | saf3r/patch/adaptive_value_clustering.py::center_graph | 当前层中心两轮更新，不能直接冻结跨层图 | 分块TopK/访存组织 | 中：距离精度和同分 |
| 局部分配 | 0.421100 | GPU phase | saf3r/patch/adaptive_value_clustering.py::local_assignment | 候选中心跨token共享 | 距离/argmin/索引融合 | 中：相等距离标签保持 |
| Probe→Grid摘要 | 0.414349 | GPU phase | saf3r/patch/reuse_probability_history.py::observe_execution | 两次score指数、grid索引与cache校准存在共享机会 | 可把weighted/unweighted摘要与分段统计融合 | 中：atomic和求和次序 |
| 缓存校准 | 0.400021 | GPU phase | saf3r/patch/reuse_probability_history.py::save_evidence; saf3r/triton/reuse_probability_history.py::store | grid count/selected max/sum与摘要共享索引 | 归约与FP16 log写回 | 中：log域稳定性、量化 |
| Prior与Beta | 0.386293 | GPU phase | saf3r/patch/reuse_probability_history.py::build_prior | current.mean和ratio/Beta在Python与_predict均计算 | prior填充、校正、pi汇总；不可用簇均值代替token prior | 中：浮点次序和状态选择 |
| Mask Packing基础kernels | 0.378540 | GPU kernel self lower bound | saf3r/patch/reuse_inference_ops.py::pack | flag/prefix/计数/索引可减少全Patch读写 | prefix与emit需跨CTA协调 | 中：顺序、padding、预算 |
| Exact候选筛选 | 0.352906 | GPU phase | saf3r/patch/reuse_residual_exploration.py::ReuseResidualExploration.forward | 保留已有共享coarse QK；不要再计算一套 | 稀疏MM、topk、allowed写回 | 高：不得取消候选资格改变算法 |
| Budget两类kernel | 0.283358 | GPU kernel self | saf3r/triton/reuse_compact.py::budget/_choose/_dual | Cost的0/1/population结构可研究隐式表达 | 需要全Group归约；不能少轮迭代 | 中高：binary search及tie break必须一致 |

状态Kernel同时计算Drop加权Value矩、Rep距离、History最近Owner和四状态目标。成员距离由prepare_geometry每层预计算一次，**已经跨Group共享**；各Group仍重复读取同一距离/Value并应用不同prior与资格。独立组件由同一Triton源码经无关输出删除生成，使用编译器DCE；包含Python发射间隙、不同输出开销和失去融合复用，不能相加，也不能用“某组件比融合更慢”推导它在融合中占比。

| component | seconds | diagnostic_elapsed_seconds | max_abs_error | changed_states | equivalent_speed_comparison_valid |
| --- | --- | --- | --- | --- | --- |
| Fused State Loss | 0.0002789439857006073 | N/A | N/A | N/A | N/A |
| Drop Moment/Loss | 0.0004211519956588745 | N/A | 0.0 | N/A | N/A |
| Rep Loss | 0.0004185920059680939 | N/A | 0.0 | N/A | N/A |
| History Distance Gather | 0.000420960009098053 | N/A | N/A | N/A | N/A |
| History Nearest Owner | 0.00041996800899505617 | N/A | N/A | N/A | N/A |
| History Loss | 0.0004211519956588745 | N/A | 0.0 | N/A | N/A |
| History Eligible Mass | 0.00043011200428009035 | N/A | N/A | N/A | N/A |
| History Value Contribution and Remaining Moment | 0.0003384000062942505 | N/A | N/A | N/A | N/A |
| Exact Eligibility | 0.0003257279992103577 | N/A | N/A | N/A | N/A |
| Shared Cluster Geometry (existing) | N/A | N/A | N/A | N/A | N/A |
| Cluster-Grid sufficient stats plus exact arbitrary prior residual | N/A | 0.000782368004322052 | 2.9103830456733704e-11 | 1 | False |

对照B中原方案已共享geometry；额外Cluster–Grid充分统计量严格保留任意token prior残差。其Drop/Rep误差约2.91e-11，但出现1个状态变化，故等价速度比较N/A；独立算子耗时保留，不作为可用替代。对照C的仅资格质量、Value contribution/剩余moment均只在微基准运行，没有替换History loss或回写模型。

对照A固定同一QKV/mask/bias，取L12首4组，完整K/V bank。4Probe和1Probe分别使用同一组第一Probe对应的真值；Mean Query与四Probe均值比较，目标不同，不能直接按误差列排名。导出的是`log(mean_h exp(raw_logit-LSE_compensated))`，不是最终Dense概率；概率误差列对比理想Dense分母，包含分母差异，不等同于完成Grid校准后的估计器质量。最终条件概率另行和Dense Grid条件概率比较。没有修改正式四Probe。

| variant | seconds | probe_qk_edges | output_bytes | probability_relative_l1 | scope |
| --- | --- | --- | --- | --- | --- |
| Direct AV only | 0.0004859200119972229 | 0 | 270336 | N/A | Independent microbenchmark;4 fixed groups; full K/V bank; no writeback |
| 4 Probe export | 8.979199826717377e-05 | 6822144 | 1705536 | 0.5344148278236389 | QK+LSE read+logmeanexp+store are fused; allocations included in wrapper event interval |
| 1 Probe export | 8.691199868917465e-05 | 1705536 | 426384 | 0.514671802520752 | QK+LSE read+logmeanexp+store are fused; allocations included in wrapper event interval |
| Mean Query independent QK | 8.931200206279755e-05 | 1705536 | 426384 | 0.7596635818481445 | Mean Q and its own sparse LSE prepared outside timing; compares to four-probe Dense mean; approximation only |
| Direct AV +4 Probe | 0.0005256320238113403 | 6822144 | 1975872 | N/A | Two kernels and wrapper allocation; same QKV/mask/bias |
| Legacy all32 history observation | 0.00036793598532676696 | 53266432 | N/A | N/A | Actual legacy observation implementation; no model writeback |
| 4 Probe Grid conditional and cache | 0.0003856639862060547 | 0 | 832288 | 1.2410880327224731 | Same-input conditional diagnostic; score exponent/log introduces extra roundoff; model cache unchanged |

4Probe底层仍执行16行Tensor Core tile；1Probe同样16行，因此微基准几乎不加速。四Probe额外QK的logical有效边数见qk_counts.json；与正式AV逻辑边数约为1/8关系，补齐后的实际dot-lane估算约4倍于Probe逻辑容量。该估算不是硬件性能计数。LSE已经从正式AV复用，导出后校准/缓存不再计算QK。Coarse pooled QK也被两种归一化域共享。

| logical_valid_probe_edges_from_saved_layers | logical_capacity_probe_edges | scheduled_dot_lane_slots_estimate | probe_output_bytes | export_launches |
| --- | --- | --- | --- | --- |
| 170302377024 | 170444445696 | 684412895232 | 42611111424 | 1584 |

正式实现仍在L23计算Probe导出、Probe→Grid摘要和history summary，而最后一层没有后继消费者；这是明确的可移除工作候选，本次没有删除。`debiased_log_patch_probability`是切掉special前缀的view，随后contiguous复制；Special Probability本身也未被当前历史路径消费。这些可以优先做活跃性与布局优化，不能错误删掉AV分母中的Special Token。

该独立trace的_choose/_dual分别比1518次正常重放预期多34/32次，表明warmed输入仍触发2次Graph首次预运行；这些实际开销计入前向，未假装全为缓存命中。

Budget每轮读取Objective和Cost、计算value+lambda*cost、argmin与budget cost在_choose内融合，_dual做总cost与lambda更新。没有17个独立状态求解器；是16轮+最终选择。预算统计：

| iterations | total_kernel_self_seconds | mean_16_round_kernel_seconds_per_window | mean_round_kernel_seconds | state_tables_bytes_per_full64_group_choose | reads_over23_layers17_passes_estimate |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.283358 | 0.000177 | 0.000011 | 53266432 | 1355068067712 |

字节数是逻辑load估算，L2可能命中，不能当作DRAM实测。budget_iterations.csv给出每轮带标记的choose/dual阶段；Final选择在operations.csv单列。初始化、最终count sum有周边小算子，不包含在上述两类kernel self之内。

CPU runtime有2197次cudaStreamSynchronize，CPU累计约2.434s；382606次cudaLaunchKernel累计约4.503s，1518次cudaGraphLaunch约0.369s。它们与GPU执行重叠，不能再加到前向时间，也不能声称消除同步就一定节省2.434秒。显存事件包括张量allocation/free和CPU allocation，不等于每次都调用cudaMalloc；真正API在cpu_runtime_summary.csv分开。cache均在GPU，无pinned spill/prefetch传输，CPU缓存copy为0；其他D2D/H2D/D2H payload见memory_transfers.csv。

Nsight Systems未安装；Nsight Compute返回ERR_NVGPUCTRPERM。Hardware occupancy、DRAM throughput、实际kernel读写字节全部N/A。kernel_resources.csv仅报告Triton编译器资源：AV128 registers/thread、0 spills、45568B shared；Probe114 registers/thread、0 spills、18432B shared。不把API计算的occupancy或编译资源冒充硬件实测。

建议下一阶段优先级：①融合状态Kernel内跨Group共享Value/距离并降低访存和launch开销，保持资格和Owner同分规则；②合并Probe→Grid摘要、缓存分段归约和Prior周边重复索引/均值/Beta，并删除经活跃性证明的L23历史工作；③优化Probe导出数据布局和Special未使用输出，之后再评估与AV融合，严格核对log域Debias、最终LSE和跨Head归约。以上仅为建议，本次未改动正式实现。

本次不评测几何质量，AUC@3/AUC@30/ACC/Comp/CD/F1均N/A（Profiling only）；不应称为新质量评测。初始AST位置错误、trace网络写入中断及独立组件生成失败的日志均保留在profiles目录；成功采集文件和对应monitor独立列出。

采集恢复说明：生产Kernel trace在GPU采样和本机JSON导出完成后，因复制到网络文件系统过慢而中断复制进程；完整本机JSON随后通过流式解析恢复，持久gzip也已保存。故trace进程monitor退出码为-15，不能称整个进程成功。标记阶段采集也在阶段数据完成后遇到独立组件生成错误，组件已由单独v2成功采集补齐。这些退出码和阶段有效性详见collection_attempts.json。
