# History Loss A＋全特殊token双向保护

以冻结方案A为基础，唯一改动dense_special_queries从1到5：每帧camera及4个register作为Query均对全部Key计算；它们作为Key始终常驻。A的Loss、H∪{Rep} Support、Cost、权重、M5c、Value Clustering、Query Group、逐层预算及Attention kernel完全不变。

仍复用先稀疏observer、再用全量SDPA覆盖特殊Query输出的实现，历史流仍来自稀疏observer；未物理剔除稀疏路径的特殊Query重复计算，其代价完整计入正常耗时。

ETH3D既有协议11/11场景、377帧，新方案全部本批重跑；A对照质量复用上一批完整结果，不重跑未改变的控制精度。上限128帧uniform seed0；terrace/meadow依原协议排除，无新增过滤。相同精确帧表、权重SHA、预处理、评测协议与冻结源码全部审计通过。

AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2非平方距离；宏平均为11场景等权平均。

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| A_camera_dense | 21.787389 | 77.980775 | 0.320746 | 0.856027 | 0.588386 | 55.556054 |
| A_all_specials | 22.429048 | 78.868713 | 0.317107 | 0.864698 | 0.590902 | 55.604616 |

相对A：AUC@3 +0.6417pp，AUC@30 +0.8879pp，F1 +0.0486pp，CD +0.002516m。

## 逐场景结果

| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_camera_dense | facade | 76 | 29.239766 | 88.368421 | 0.266906 | 0.264199 | 0.265552 | 63.956421 |
| A_all_specials | facade | 76 | 32.690058 | 89.231579 | 0.279266 | 0.271280 | 0.275273 | 64.258788 |
| A_camera_dense | terrains | 42 | 8.323655 | 61.974448 | 0.349901 | 0.245832 | 0.297867 | 61.877087 |
| A_all_specials | terrains | 42 | 8.555943 | 62.458382 | 0.300543 | 0.244710 | 0.272627 | 64.012429 |
| A_camera_dense | delivery_area | 40 | 18.162393 | 81.700855 | 0.665107 | 1.497164 | 1.081135 | 13.087305 |
| A_all_specials | delivery_area | 40 | 19.401709 | 83.824786 | 0.642957 | 1.517095 | 1.080026 | 13.466539 |
| A_camera_dense | electro | 39 | 41.205578 | 93.022942 | 0.204344 | 4.705581 | 2.454962 | 64.127513 |
| A_all_specials | electro | 39 | 41.835358 | 92.982456 | 0.184058 | 4.719783 | 2.451920 | 64.994389 |
| A_camera_dense | courtyard | 38 | 28.639165 | 83.304884 | 0.273376 | 0.193396 | 0.233386 | 64.982052 |
| A_all_specials | courtyard | 38 | 29.350403 | 86.785206 | 0.272785 | 0.179884 | 0.226335 | 60.960111 |
| A_camera_dense | playground | 32 | 9.408602 | 65.248656 | 0.418114 | 0.969044 | 0.693579 | 30.267604 |
| A_all_specials | playground | 32 | 9.005376 | 67.029570 | 0.390563 | 0.945485 | 0.668024 | 37.924350 |
| A_camera_dense | kicker | 31 | 44.372760 | 93.612903 | 0.068479 | 0.064752 | 0.066615 | 99.217462 |
| A_all_specials | kicker | 31 | 42.222222 | 93.089606 | 0.063555 | 0.059269 | 0.061412 | 99.234099 |
| A_camera_dense | office | 26 | 20.923077 | 75.917949 | 0.078027 | 0.075774 | 0.076900 | 98.975659 |
| A_all_specials | office | 26 | 21.128205 | 78.769231 | 0.075746 | 0.074868 | 0.075307 | 99.448049 |
| A_camera_dense | relief_2 | 20 | 4.035088 | 51.035088 | 0.387480 | 0.514426 | 0.450953 | 27.880222 |
| A_all_specials | relief_2 | 20 | 4.561404 | 49.947368 | 0.369531 | 0.534806 | 0.452169 | 29.506660 |
| A_camera_dense | relief | 19 | 15.204678 | 78.693957 | 0.629121 | 0.685891 | 0.657506 | 6.070017 |
| A_all_specials | relief | 19 | 13.060429 | 78.089669 | 0.705874 | 0.755312 | 0.730593 | 2.772644 |
| A_camera_dense | pipes | 14 | 20.146520 | 84.908425 | 0.187351 | 0.200236 | 0.193793 | 80.675249 |
| A_all_specials | pipes | 14 | 24.908425 | 85.347985 | 0.203294 | 0.209185 | 0.206239 | 75.072716 |

## 运行协议

pose+recon_unposed均评测；F1 threshold=0.25m，downsample/voxel=0.0390625m，TSDF trunc=0.2m，mesh=1000000点；原unposed对齐、掩码和深度处理不变。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；cluster fraction0.12，Query group256。未使用7Scenes真值调参。

ETH3D以下为0暖机、1次完整冷前向，包含JIT/缓存初始化；不是暖机速度比较。相对Dense速度为N/A，原因是只在7Scenes测量速度。

| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |
|---|---:|---:|---:|---:|
| facade | 5 | 9.113102 | 11.1647 | 0.163845 |
| terrains | 6 | 6.233337 | 9.7182 | 0.164090 |
| delivery_area | 4 | 6.544067 | 9.6374 | 0.164028 |
| electro | 6 | 6.596068 | 9.5799 | 0.163970 |
| courtyard | 4 | 6.521082 | 9.5349 | 0.164070 |
| playground | 5 | 6.799299 | 9.3577 | 0.164082 |
| kicker | 6 | 6.494347 | 9.2977 | 0.164038 |
| office | 4 | 5.854490 | 9.0957 | 0.164072 |
| relief_2 | 5 | 5.462040 | 8.8549 | 0.164162 |
| relief | 4 | 5.901434 | 8.8237 | 0.164125 |
| pipes | 6 | 4.133378 | 8.6133 | 0.164144 |

## 同批正常完整前向速度

7Scenes heads128，非全7场景平均；GPU6 RTX4090，全部质量任务结束后2暖机+5重复，中位数。包含每次初始化/reset、clustering、loss/selection/packing、history、重排、全特殊Query补算及SAF3R Top-K reset；不含模型/图像磁盘加载。

| Method | Median s | Dense speedup | Time / A | Time / SAF3R | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| A | 9.424213 | 1.8164x | 1.0000 | 1.0983 | 15.3684 |
| A_all_specials | 9.488022 | 1.8042x | 1.0068 | 1.1057 | 15.3684 |
| dense | 17.118153 | 1.0000x | 1.8164 | 1.9949 | 14.3586 |
| saf3r | 8.580765 | 1.9949x | 0.9105 | 1.0000 | 14.6143 |

A对照、Dense、原SAF3R只重跑计时，沿用用户此前明确授权。计时组六项几何质量为N/A，原因是仅做RGB计时。SAF3R为Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json；非论文DINO Top-K或Top-128，未混用论文/H100数据。

## 独立Mask分阶段计时

| Method | Mask pipeline ms | Prior/selection/budget/pack/weights ms |
|---|---:|---:|
| A | 575.2131 | 283.7498 |
| A_all_specials | 574.0072 | 283.8756 |

在正常计时后，同模型/输入上额外两次CUDA事件profile。Mask累计仅覆盖已标注的Value更新/聚类、共享统计、coarse估计、prior/gate/loss/solver/pack/weights，不包含未独立包裹的reset/layout初始化、QKV投影、Attention及history执行；这些全部计入正常完整前向。事件可包含主机提交间隙；额外profile秒数不代替正常速度。

该实现是否改善精度应同时检查AUC、F1与CD，不能仅看特殊Query已全量。单次完整质量评测存在已知数值波动，不宣称统计显著性。本次未自动修改其他方案或历史冻结配置。
