# A＋全特殊token：cluster数20%消融

以A＋全特殊token修正为基础，唯一改动cluster_fraction从0.12到0.20，即floor(0.20×patch token数)个簇。两个版本每帧camera及4个register作为Query均对全部Key计算、作为Key始终常驻。A的Loss、H∪{Rep} Support、Cost、权重、M5c、Value Clustering算法、Query Group、逐层Key预算及Attention kernel均保持不变；簇数增加不等于提高Key保留预算。

仍复用先稀疏observer、再用全量SDPA覆盖特殊Query输出的实现，历史流仍来自稀疏observer；未物理剔除稀疏路径的特殊Query重复计算，其代价完整计入正常耗时。

ETH3D既有协议11/11场景、377帧，新方案全部本批重跑；12%簇数对照质量复用上一批完整结果，不重跑未改变的控制精度。上限128帧uniform seed0；terrace/meadow依原协议排除，无新增过滤。相同精确帧表、权重SHA、预处理、评测协议与冻结源码全部审计通过。

AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2非平方距离；宏平均为11场景等权平均。

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| cluster12 | 22.429048 | 78.868713 | 0.317107 | 0.864698 | 0.590902 | 55.604616 |
| A_all_specials_cluster20 | 20.486889 | 78.410848 | 0.323785 | 0.853332 | 0.588559 | 59.348578 |

相对12%簇数：AUC@3 -1.9422pp，AUC@30 -0.4579pp，F1 +3.7440pp，CD -0.002344m。

## 逐场景结果

| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cluster12 | facade | 76 | 32.690058 | 89.231579 | 0.279266 | 0.271280 | 0.275273 | 64.258788 |
| A_all_specials_cluster20 | facade | 76 | 35.040936 | 90.122807 | 0.264337 | 0.273411 | 0.268874 | 64.454099 |
| cluster12 | terrains | 42 | 8.555943 | 62.458382 | 0.300543 | 0.244710 | 0.272627 | 64.012429 |
| A_all_specials_cluster20 | terrains | 42 | 9.639954 | 59.996129 | 0.312858 | 0.260463 | 0.286661 | 64.478630 |
| cluster12 | delivery_area | 40 | 19.401709 | 83.824786 | 0.642957 | 1.517095 | 1.080026 | 13.466539 |
| A_all_specials_cluster20 | delivery_area | 40 | 14.529915 | 77.923077 | 0.485458 | 1.239400 | 0.862429 | 29.286352 |
| cluster12 | electro | 39 | 41.835358 | 92.982456 | 0.184058 | 4.719783 | 2.451920 | 64.994389 |
| A_all_specials_cluster20 | electro | 39 | 37.112011 | 92.887989 | 0.198577 | 4.726994 | 2.462785 | 66.406685 |
| cluster12 | courtyard | 38 | 29.350403 | 86.785206 | 0.272785 | 0.179884 | 0.226335 | 60.960111 |
| A_all_specials_cluster20 | courtyard | 38 | 17.828355 | 84.945472 | 0.214617 | 0.156579 | 0.185598 | 74.811607 |
| cluster12 | playground | 32 | 9.005376 | 67.029570 | 0.390563 | 0.945485 | 0.668024 | 37.924350 |
| A_all_specials_cluster20 | playground | 32 | 11.559140 | 71.142473 | 0.333987 | 0.901887 | 0.617937 | 51.152911 |
| cluster12 | kicker | 31 | 42.222222 | 93.089606 | 0.063555 | 0.059269 | 0.061412 | 99.234099 |
| A_all_specials_cluster20 | kicker | 31 | 44.014337 | 93.835125 | 0.076647 | 0.072006 | 0.074326 | 98.833634 |
| cluster12 | office | 26 | 21.128205 | 78.769231 | 0.075746 | 0.074868 | 0.075307 | 99.448049 |
| A_all_specials_cluster20 | office | 26 | 20.615385 | 78.471795 | 0.059716 | 0.058607 | 0.059161 | 99.604542 |
| cluster12 | relief_2 | 20 | 4.561404 | 49.947368 | 0.369531 | 0.534806 | 0.452169 | 29.506660 |
| A_all_specials_cluster20 | relief_2 | 20 | 2.807018 | 52.684211 | 0.620013 | 0.566390 | 0.593201 | 31.371046 |
| cluster12 | relief | 19 | 13.060429 | 78.089669 | 0.705874 | 0.755312 | 0.730593 | 2.772644 |
| A_all_specials_cluster20 | relief | 19 | 11.695906 | 74.502924 | 0.761588 | 0.784105 | 0.772846 | 1.574211 |
| cluster12 | pipes | 14 | 24.908425 | 85.347985 | 0.203294 | 0.209185 | 0.206239 | 75.072716 |
| A_all_specials_cluster20 | pipes | 14 | 20.512821 | 86.007326 | 0.233838 | 0.346811 | 0.290324 | 70.860641 |

## 运行协议

pose+recon_unposed均评测；F1 threshold=0.25m，downsample/voxel=0.0390625m，TSDF trunc=0.2m，mesh=1000000点；原unposed对齐、掩码和深度处理不变。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；cluster fraction0.20（对照0.12），Query group256。未使用7Scenes真值调参。

ETH3D以下为0暖机、1次完整冷前向，包含JIT/缓存初始化；不是暖机速度比较。相对Dense速度为N/A，原因是只在7Scenes测量速度。

| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |
|---|---:|---:|---:|---:|
| facade | 6 | 12.498244 | 11.2304 | 0.164279 |
| terrains | 4 | 9.334846 | 9.7491 | 0.164353 |
| delivery_area | 5 | 9.447371 | 9.6675 | 0.164312 |
| electro | 5 | 9.540750 | 9.6066 | 0.164296 |
| courtyard | 4 | 9.752155 | 9.5569 | 0.164325 |
| playground | 6 | 10.768323 | 9.4008 | 0.164329 |
| kicker | 5 | 8.785615 | 9.3381 | 0.164311 |
| office | 4 | 8.543060 | 9.1278 | 0.164312 |
| relief_2 | 6 | 8.514334 | 8.8718 | 0.164365 |
| relief | 5 | 8.684845 | 8.8396 | 0.164359 |
| pipes | 4 | 7.741047 | 8.6206 | 0.164342 |

## 同批正常完整前向速度

7Scenes heads128，非全7场景平均；GPU6 RTX4090，全部质量任务结束后2暖机+5重复，中位数。包含每次初始化/reset、clustering、loss/selection/packing、history、重排、全特殊Query补算及SAF3R Top-K reset；不含模型/图像磁盘加载。

| Method | Median s | Dense speedup | Time / cluster12 | Time / SAF3R | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| cluster12 | 9.492108 | 1.8049x | 1.0000 | 1.1014 | 15.3680 |
| A_all_specials_cluster20 | 9.793269 | 1.7494x | 1.0317 | 1.1364 | 15.4952 |
| dense | 17.132507 | 1.0000x | 1.8049 | 1.9880 | 14.3586 |
| saf3r | 8.618074 | 1.9880x | 0.9079 | 1.0000 | 14.6143 |

12%簇数对照、Dense、原SAF3R只重跑计时，沿用用户此前明确授权。计时组六项几何质量为N/A，原因是仅做RGB计时。SAF3R为Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json；非论文DINO Top-K或Top-128，未混用论文/H100数据。

## 独立Mask分阶段计时

| Method | Mask pipeline ms | Prior/selection/budget/pack/weights ms |
|---|---:|---:|
| cluster12 | 572.6337 | 283.8511 |
| A_all_specials_cluster20 | 838.1092 | 403.6363 |

在正常计时后，同模型/输入上额外两次CUDA事件profile。Mask累计仅覆盖已标注的Value更新/聚类、共享统计、coarse估计、prior/gate/loss/solver/pack/weights，不包含未独立包裹的reset/layout初始化、QKV投影、Attention及history执行；这些全部计入正常完整前向。事件可包含主机提交间隙；额外profile秒数不代替正常速度。

## 固定预算下的簇数压力

以下来自heads128最后一次正常计时的逐层记录，不是ETH3D宏平均。浅/深层为L1–L5与L18–L23，排除不使用四状态损失的L0 seed；中间层为L6–L17。Drop按簇数计数，不等于被丢弃真实概率质量或token数。

| Method | Fine clusters | Shallow/deep Drop % | Middle Drop % | Actual unique density % |
|---|---:|---:|---:|---:|
| cluster12 | 15912 | 20.3313 | 0.5111 | 16.2806 |
| A_all_specials_cluster20 | 26521 | 52.1877 | 6.0986 | 16.3632 |

簇数增大而Key预算保持不变，使低保留预算层的簇覆盖压力增大；更高的Drop比例是一个可能影响位姿的机制，但这些统计本身不能证明ETH3D AUC下降的全部原因。Mask阶段随簇数增加而变慢，包含更多簇统计与四状态预算处理。20%版本F1更高、AUC更低，表现为重建和位姿指标取舍，不是全面质量提升。

20%版本比12%版慢3.1728%，比原SAF3R慢13.6364%；本批中位数下满足不牺牲速度：False，满足SAF3R+10%目标：False。本次结果不支持直接将20%作为当前改进版。

该实现是否改善精度应同时检查AUC、F1与CD，不能仅看特殊Query已全量。单次完整质量评测存在已知数值波动，不宣称统计显著性。本次未自动修改其他方案或历史冻结配置。
