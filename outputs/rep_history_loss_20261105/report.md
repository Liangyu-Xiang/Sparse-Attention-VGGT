# 基于Rep Loss的History Loss：最终实验报告

只验证文档的两种公式，保留原始实现、独立配置与输出。A为完整union-support版本；B明确是|H|=1的受限消融，其余簇保持原逻辑。未引入新状态系统、额外Fine QK、Query-dependent全量Value距离、M5c或聚类改动、额外预算、Attention kernel改动。

## 实现与配置

- saf3r/patch/rep_history_loss.py：新模块，复用原共享Cluster×Grid Rep统计与概率残差；仅遍历历史边累计p_h*d_h。A新增O(Group×Cluster)的修正/union gate与计数权重，B只修正singleton。
- saf3r/patch/reuse_residual_exploration.py、reuse_window_graph.py：添加默认关闭的配置与dispatch。原始shared loss/kernel/packing、M5c、Value clustering及Attention kernel文件未修改，protected_source_files的SHA审计通过。
- full/configs/baseline.json、A.json、B.json：同一原始布局修复基线，仅A/B配置rep_history_loss；特殊Query沿用用户给出的camera-only基线，不混入all-special修正。

A：L_H=max(0,L_Rep−Σ_H p_h||V_h−V_r||²)。S=H∪{r}，c_H=|S|；非Rep历史Key权重1，Rep权重=|C|−(|S|−1)。H={Rep}时与Rep同Loss、Cost、Mask、权重；H=C时均为单位权重。该权重实现“其他成员由Rep表示”的计数代理，并不保证真实attention概率质量或softmax输出误差完全匹配。

B：仅|H|=1时L_H=max(0,L_Rep+(pi_C−2p_h)||V_h−V_r||²)，cost1，原H support与population补偿不变；多历史Key不套用单Key公式。

## 正确性与协议

7项初始CUDA测试通过；补充原始基线与诊断路径等价后5项测试通过。覆盖H={Rep}原始Loss恒等（不把inf门控当loss）、Cost/Mask/权重、A非负/单调/完整保留、B概率主导、无重复Key、packed count和cost一致、不超预算、可用输出有限、相同支持/权重的Attention输出一致，以及动态先验与历史的CUDA Graph重放。全部完整场景另通过预算/帧表/权重/配置/协议/源码/GPU隔离审计。

ETH3D完整11场景、377帧/方法；每场景上限128帧，uniform seed0；沿用既有terrace/meadow排除设置，不新增过滤。pose+recon_unposed均评测。阈值0.25m、downsample/voxel0.0390625m、TSDF trunc0.2m、mesh1M点；原unposed对齐/掩码/深度处理不变。AUC/F1为百分数，距离为米，CD=(ACC+Comp)/2非平方距离。均为场景等权宏平均。未使用7Scenes测试真值调参。

## 小规模验证（三场景，不是全数据集）

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 14.764243 | 78.158209 | 0.308703 | 0.508498 | 0.408601 | 55.402579 |
| A | 19.398096 | 77.820655 | 0.292947 | 0.454225 | 0.373586 | 58.641635 |
| B | 16.929832 | 79.601345 | 0.319433 | 0.478276 | 0.398855 | 57.617242 |

## 完整ETH3D比较（本批新测）

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 18.507119 | 76.752822 | 0.323607 | 0.871068 | 0.597338 | 56.077487 |
| A | 21.787389 | 77.980775 | 0.320746 | 0.856027 | 0.588386 | 55.556054 |
| B | 19.398037 | 77.158642 | 0.355413 | 0.907583 | 0.631498 | 54.610242 |

用户提供的既有参考单独列示，不将新测分数平移/缩放到历史参考。

| Reference | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| historical layout-fixed | 18.510638 | 76.749655 | 0.323066 | 0.871509 | 0.597288 | 56.077724 |

## 正常完整前向耗时

7Scenes heads128，非全7场景宏平均；GPU6 RTX4090，全部CPU质量评分结束后，同批每种方法2暖机+5重复，完整前向中位数，包含初始化/reset、Value clustering、routing/loss/packing、history、重排与特殊Query补算；不包含模型/图像磁盘加载。统计开关关闭，未使用下面的profiling耗时代替正常时间。

| Method | Median s | Dense speedup | Time / Baseline | Time / SAF3R | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| baseline | 9.347404 | 1.8332x | 1.0000 | 1.0791 | 15.3732 |
| A | 9.430948 | 1.8169x | 1.0089 | 1.0887 | 15.3684 |
| B | 9.334895 | 1.8356x | 0.9987 | 1.0777 | 15.3721 |
| dense | 17.135328 | 1.0000x | 1.8332 | 1.9782 | 14.3586 |
| saf3r | 8.662230 | 1.9782x | 0.9267 | 1.0000 | 14.6143 |

Dense与SAF3R只计时，六项几何质量为N/A；SAF3R使用原版Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO Top-K或Top-128。未混用论文/H100结果。

## Mask构建耗时（独立生产路径CUDA事件）

Quiet production path; two CUDA-event profile forwards after2warm+5repeat normal timing on same GPU/model; pipeline includes clustering/shared pooling/coarse estimation/prior+gate+loss+solver+pack+weights; distinct from uninstrumented production latency

| Method | Mask pipeline ms | Prior/gate/loss/budget/pack/weights ms | Raw pipeline samples ms |
|---|---:|---:|---|
| baseline | 576.8997 | 269.6880 | [582.5407669618726, 571.2587217185646] |
| A | 578.7480 | 284.5039 | [578.7604470327497, 578.7354550771415] |
| B | 579.8531 | 282.1328 | [583.8624958209693, 575.8437141217291] |

以上Mask pipeline包含Value更新/聚类、共享统计、coarse估计和选择，不含稀疏Attention与历史写出；selection子项不含聚类。CUDA事件可能包含主机提交空隙，不等同纯kernel算力时间。两次额外profile均在正常计时之后、同模型/GPU/input上进行，未改变生产公式或增加状态统计；暖机后Graph创建/淘汰为0。

带概率coverage统计的profile、并行CPU评分时的早期profile分别保留在profiles/和profiles_production/，它们是诊断/初步观测，不替代quiet生产路径计时。

## 各层状态与History probability coverage

完整72行状态占比、原H与实际support均值、包含Rep/仅Rep比例、保留概率质量/coverage及各状态实际预算见state_layers.md、state_layers.csv和state_layers.json。明确区分H和A新增的Rep，不将union support算成原始历史证据。L0是原生seed初始化，会计项N/A；实际密度仍提供。状态诊断范围为heads128的两个profile，不伪装成ETH3D宏平均。

## 同Q/K/V对Dense输出误差

pipes,14 frames; layers6/11/13/19, first window group1,32 fixed ordinary Queries,all heads/full Key bank; same current QKV, M5c priors, clusters and history from baseline trajectory. Dense used only after mask selection for diagnostic errors; not full-model quality, not global optimality, not deployment QK.

| Layer | Method | Post-Wo relative L2 % | Post-Wo cosine |
|---|---|---:|---:|
| 6 | baseline | 11.037447 | 0.99440783 |
| 6 | A | 11.524457 | 0.99385118 |
| 6 | B | 11.540587 | 0.99403977 |
| 11 | baseline | 19.415131 | 0.98130643 |
| 11 | A | 19.415502 | 0.98137605 |
| 11 | B | 19.431943 | 0.98129731 |
| 13 | baseline | 19.009700 | 0.98228180 |
| 13 | A | 21.310368 | 0.97761089 |
| 13 | B | 19.026923 | 0.98226571 |
| 19 | baseline | 8.775964 | 0.99615967 |
| 19 | A | 8.775964 | 0.99615967 |
| 19 | B | 8.775964 | 0.99615967 |

抽样结果不显示A/B在所有层更接近Dense；A在部分层误差更大。该诊断不作为唯一选择目标，也不用于调整本轮参数。

## 解释与限制

A修正了Rep/History表示误差定义不一致及support缺Rep的问题，但残余成员仍以Rep的Key/Value和计数权重代替：其logit与真实成员不相同，不能保证保留真实概率质量，更不等于全局softmax Attention输出误差。共享head-mean/RMS Value代理和M5c估计误差仍存在。

B来自对其余成员Value接近Rep的近似；一般的单Key替代精确失真还包含其他Value与(V_h−V_r)的交叉项。公式没有建模这些项，且本轮多历史Key原逻辑完全未修复，因此收益或退化不应外推为完整History损失修复。

端到端AUC/F1/CD与固定QKV的少量局部Attention误差不是同一目标；后续层会改变QKV与状态轨迹。只以本批完整六指标、正常速度和逐场景退化判断，不把History使用率或代理损失降低当作成功证据。

完整逐场景六指标与冷前向记录见full/report.md和full/per_scene.csv；ETH3D冷耗时包含JIT，不能作为暖机速度比较。只报告7Scenes速度，不使用7Scenes真值调参。本报告保留所有候选与负面结果，未自动替换当前方案。

## 逐场景变化与结论

| Scene | A ΔAUC@3 pp | A ΔF1 pp | A ΔCD m | B ΔAUC@3 pp | B ΔF1 pp | B ΔCD m |
|---|---:|---:|---:|---:|---:|---:|
| facade | +4.818713 | +1.054333 | +0.002019 | +2.140351 | -2.418602 | +0.042310 |
| terrains | +0.967867 | +0.436771 | +0.007808 | -1.587302 | +11.095466 | -0.052638 |
| delivery_area | +4.273504 | -5.430158 | +0.086960 | +0.555556 | -6.907078 | +0.246598 |
| electro | +8.636977 | -3.422487 | +0.055543 | -1.259559 | -10.771731 | +0.127777 |
| courtyard | +11.996207 | +8.291658 | -0.030160 | +1.422475 | +1.917750 | -0.012186 |
| playground | +0.806452 | -0.779381 | -0.022418 | +1.411290 | +5.886044 | -0.007803 |
| kicker | +1.290323 | +0.175436 | -0.007833 | +5.161290 | -0.358269 | -0.003339 |
| office | -1.743590 | +0.157397 | -0.011687 | -0.615385 | -0.467613 | +0.011469 |
| relief_2 | -0.350877 | -13.097823 | -0.017306 | +1.052632 | -14.830529 | +0.099706 |
| relief | +4.288499 | +4.673597 | -0.108925 | -2.144250 | +1.874666 | -0.066876 |
| pipes | +1.098901 | +2.204890 | -0.052467 | +3.663004 | -1.159806 | -0.009249 |

A的AUC@3在9/11场景提高，主要增益见courtyard、electro、facade、relief和delivery_area；F1整体略降主要受relief_2、delivery_area和electro拖累。relief_2同时出现F1降低而CD改善，说明阈值内命中率与均值距离不能互相替代。B的CD退化主要见delivery_area、electro和relief_2。

heads128诊断中，L1–L23等层平均History簇占比约为基线0.993%、A8.447%、B2.032%；A的Exact簇占比由约3.804%降至2.544%。这些只是该诊断输入上的预算重新分配证据，不是ETH3D宏平均或精度因果证明。A被选History的原H估计概率coverage约58.12%，低于基线约78.49%；插入Rep后的实际support另行记录，不能混作原历史证据。

A改善AUC和平均CD，但F1约下降0.52个百分点，正常完整前向约慢0.89%；只能称为有收益的部分修正，尚未同时满足恢复F1与完全不牺牲速度。B虽然AUC略升，但CD和F1均变差，不建议作为本轮改进方案。

A同时改变了公式所必需的support、cost和权重，本轮只能评价这组一致性修正，不能分离损失项与各项执行修正的独立贡献。所有参数均在小规模/完整评测前冻结，未根据这些结果重新调整。各方法选中簇集合不同，coverage差异不能直接解释为概率估计质量变坏。

本批中位数相对SAF3R：A慢约8.87%，B慢约7.77%，均在10%范围内；A相对原基线慢约0.89%，不满足严格的完全无速度代价要求。

Mask表仅累计已标注的阶段；未独立包裹的reset/layout初始化、QKV投影等不能从该表推出，均包含在正常完整前向耗时中。两个测试批次包含重叠案例，不将其宣称为12个独立测试。
