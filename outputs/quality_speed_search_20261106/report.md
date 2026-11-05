# History A：独立超参数搜索与降低保留率后的完整评测

先完成独立RGB搜索与验证，再冻结两个候选，最后各评测一次7Scenes全七场景128帧及ETH3D既有11场景14–76帧。六项指标等场景宏平均；AUC/F1为百分数，距离米，CD=(ACC+Comp)/2，未平方。未使用7Scenes/ETH3D测试真值选参数，不按测试分数选择“最佳重复”。

## 实现与候选

保留History A、5个特殊Query全Key输出、每层当前Value更新、History概率公式、16轮预算求解及稀疏Attention。改动：12%→10%簇；head_mean→每Head4维固定正交草图（16×4=64维），减少跨Head抵消；Exact资格25%→100%，所有簇在原预算内竞争。精度优先中层21%，速度优先19%，浅/深各10%；平均保留上限15.5%/14.5%，原版16%。没有Dense层回退。

删除只在全Exact资格路径中未使用的Cluster-Grid overlap、残差需求/候选筛选、patch-only粗归一化及被完整覆盖的fallback计数与缓存。8帧四次真实前向，掩码、相机、深度、置信度均逐值相同。此工程变更默认关闭、显式启用；独立prior_residual补偿变种未入选，最终仍采用population补偿。

## 六指标宏平均

| 数据集 | 方法 | 场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense_historical | 7 | 23.66892 | 84.63887 | 0.15374 | 0.12562 | 0.13968 | 48.01277 |
| 7scenes | saf3r_historical | 7 | 23.93490 | 84.71310 | 0.14326 | 0.12788 | 0.13557 | 47.31542 |
| eth3d | dense_historical | 11 | 29.18346 | 81.81621 | 0.31319 | 0.84885 | 0.58102 | 62.57314 |
| eth3d | saf3r_historical | 11 | 23.33220 | 80.17474 | 0.30311 | 0.85772 | 0.58041 | 62.83039 |
| 7scenes | current_A12_previous_batch | 7 | 22.48020 | 84.31577 | 0.14985 | 0.12390 | 0.13688 | 48.26063 |
| eth3d | current_A12_previous_batch | 11 | 22.64454 | 78.53236 | 0.35894 | 0.91667 | 0.63780 | 52.08344 |
| 7scenes | p3_sketch10_mid21 | 7 | 23.23713 | 84.45274 | 0.15084 | 0.12152 | 0.13618 | 47.72462 |
| eth3d | p3_sketch10_mid21 | 11 | 22.01080 | 76.80262 | 0.34559 | 0.89792 | 0.62176 | 53.35156 |
| 7scenes | p3_sketch10_mid19 | 7 | 22.55109 | 84.12132 | 0.15056 | 0.12540 | 0.13798 | 46.35968 |
| eth3d | p3_sketch10_mid19 | 11 | 20.36250 | 75.42039 | 0.34775 | 0.92807 | 0.63791 | 55.93631 |

Dense/SAF3R标为historical的质量复用combined_reuse_20261031，未重跑；旧控制文件无checkpoint SHA，不能追认严格字节级匹配。当前A12参考来自history_A_special5_cluster12_speed_20261106，与本轮输入/协议/SHA逐场景核对一致。官方tracker-fixed SHA b08a43...20969，两个当前候选都保持同一文件。

## 完整前向、显存和冷启动

每场景GPU独占，1预热+3完整前向取中位数，再等场景平均。包含全部初始化/reset、聚类、历史、路由、打包、特殊Query修正及预测头；不含磁盘加载与后续CPU几何评估。表中旧A12速度是上一批历史参考，不能当作同期配对；Dense作为定义上的基线为1.00×；候选的正式Dense配对加速比N/A。独立RGB验证中每候选与A12在同卡同进程顺序配对，速度均不增加。

| 数据集 | 候选 | 前向 s | 旧A12批次 s | 时间比 新/旧 | 热峰值allocated GiB | 平均冷前向 s | 含冷启动峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | p3_sketch10_mid21 | 5.994382 | 6.090130 | 0.9843 | 14.817 | 19.560 | 28.454 |
| eth3d | p3_sketch10_mid21 | 1.244971 | 1.264160 | 0.9848 | 10.979 | 24.651 | 26.162 |
| 7scenes | p3_sketch10_mid19 | 5.890366 | 6.090130 | 0.9672 | 14.813 | 19.694 | 28.449 |
| eth3d | p3_sketch10_mid19 | 1.237918 | 1.264160 | 0.9792 | 10.976 | 22.666 | 26.162 |

SAF3R已有上一批计时参考为7Scenes6.705480秒、ETH3D1.603858秒；配置为lazy first-correspondence-head Top4/frame，不是论文DINO Top-K。本轮未重跑SAF3R。不要将历史速度直接标为本轮配对加速比。

## 相对当前A12的变化与目标验收

下列为原始相对变化：AUC/F1正数为提升，ACC/Comp/CD负数为提升。之前点云变化<2%的检查单独保留；不能把局部误差代理改进当作六项几何指标全部通过。

| 数据集 | 候选 | auc03变化 % | auc30变化 % | acc变化 % | comp变化 % | cd变化 % | f1变化 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | p3_sketch10_mid21 | +3.367 | +0.162 | +0.658 | -1.922 | -0.510 | -1.111 |
| eth3d | p3_sketch10_mid21 | -2.799 | -2.203 | -3.718 | -2.045 | -2.516 | +2.435 |
| 7scenes | p3_sketch10_mid19 | +0.315 | -0.231 | +0.472 | +1.208 | +0.805 | -3.939 |
| eth3d | p3_sketch10_mid19 | -10.078 | -3.963 | -3.116 | +1.244 | +0.017 | +7.398 |

```json
{
  "p3_sketch10_mid21_7scenes": {
    "quality_relative_changes_percent": {
      "auc03": 3.367126215110372,
      "auc30": 0.16245486110388896,
      "acc": 0.6583031025535924,
      "comp": -1.9220039562928948,
      "cd": -0.5095606155159471,
      "f1": -1.1106692500510218
    },
    "no_metric_worse_than_current": false,
    "pointcloud_change_within_prior2pct": true,
    "time_below_previous_batch": true,
    "paired_target_speed_comparison": null,
    "paired_speed_reason": "Existing current-method target timings are previous-batch references; no unchanged controls rerun. Independent RGB paired validation passed.",
    "independent_rgb_speed_pass": true,
    "independent_rgb_fidelity_pass": true
  },
  "p3_sketch10_mid21_eth3d": {
    "quality_relative_changes_percent": {
      "auc03": -2.7986137148848678,
      "auc30": -2.2025892998912777,
      "acc": -3.718244333540466,
      "comp": -2.0450132354926986,
      "cd": -2.5158370840449695,
      "f1": 2.4347937337681813
    },
    "no_metric_worse_than_current": false,
    "pointcloud_change_within_prior2pct": false,
    "time_below_previous_batch": true,
    "paired_target_speed_comparison": null,
    "paired_speed_reason": "Existing current-method target timings are previous-batch references; no unchanged controls rerun. Independent RGB paired validation passed.",
    "independent_rgb_speed_pass": true,
    "independent_rgb_fidelity_pass": true
  },
  "p3_sketch10_mid19_7scenes": {
    "quality_relative_changes_percent": {
      "auc03": 0.3153423158114288,
      "auc30": -0.2306191976063965,
      "acc": 0.47238419198947224,
      "comp": 1.2078161935895837,
      "cd": 0.8052454880244975,
      "f1": -3.9389419773051326
    },
    "no_metric_worse_than_current": false,
    "pointcloud_change_within_prior2pct": false,
    "time_below_previous_batch": true,
    "paired_target_speed_comparison": null,
    "paired_speed_reason": "Existing current-method target timings are previous-batch references; no unchanged controls rerun. Independent RGB paired validation passed.",
    "independent_rgb_speed_pass": true,
    "independent_rgb_fidelity_pass": true
  },
  "p3_sketch10_mid19_eth3d": {
    "quality_relative_changes_percent": {
      "auc03": -10.077664591657243,
      "auc30": -3.962659955133996,
      "acc": -3.115900588879572,
      "comp": 1.2441803225323067,
      "cd": 0.017314483673991354,
      "f1": 7.3975066743775075
    },
    "no_metric_worse_than_current": false,
    "pointcloud_change_within_prior2pct": false,
    "time_below_previous_batch": true,
    "paired_target_speed_comparison": null,
    "paired_speed_reason": "Existing current-method target timings are previous-batch references; no unchanged controls rerun. Independent RGB paired validation passed.",
    "independent_rgb_speed_pass": true,
    "independent_rgb_fidelity_pass": true
  }
}
```

## 搜索、诊断及限制

第一阶段60组，第二阶段26组（含对照与重复结构的因子对照），第三阶段7组（含对照）预算细化，均为有限搜索，不声称全局最优。独立校准为ScanNet scene0000_00及TUM sitting halfsphere各64RGB；验证为与最终seq01不同的chessseq03现有24RGB，以及TUM walking halfsphere128RGB。只读取RGB；全六项几何字段N/A并给出原因。精度代理为32个确定性Patch Query、当前稀疏轨迹的24层Wo投影误差，对比同QKV完整Key Attention；它不包含Dense模型完整轨迹，不保证最终几何质量。

候选必须在两个验证序列上同时更低误差、同卡热中位耗时不增加；预先冻结最低误差主候选与最快改进候选，不按最终测试分数换推荐身份。每个候选只做一次全数据集质量评测。54种Kernel分块参数在两个保留形状上作无真值速度/精度检查，保留原64Q/128K/4warp/2stage组合，没有把kernel微基准当端到端速度。

GPU7在一次验证结束时出现NVML lost，整批排除并在GPU0重做；其余单卡独占监控保留。首次新工作目录缺少数据软链接，在图像加载前失败、六指标N/A，补齐并先核验所有冻结RGB后才正式运行。另一份ScanNet JPEG损坏在校准预检中排除并记录。所有失败原样保留。

关键Key分析是独立RGB上的同预算单Key反事实，不参与冻结后的选参。每组只考察8条高概率遗漏边，不能声称所有Key的全局oracle；真实Dense概率仅用于诊断。Probe0估计与Probe0真概率单独对齐，另外保留四Probe组均值，防止混淆概率定义。详见key_impact_analysis.json/key_impact_aligned。

## 逐场景六指标

| 数据集 | 候选 | 场景 | 帧 | AUC3 % | AUC30 % | ACC m | Comp m | CD m | F1 % | 前向 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | p3_sketch10_mid21 | heads | 128 | 49.65141 | 93.59129 | 0.03334 | 0.03301 | 0.03317 | 83.58206 | 6.002234 |
| 7scenes | p3_sketch10_mid21 | chess | 128 | 20.01312 | 88.28699 | 0.14512 | 0.08993 | 0.11753 | 34.79324 | 5.977731 |
| 7scenes | p3_sketch10_mid21 | stairs | 128 | 6.45095 | 80.67954 | 0.23326 | 0.44588 | 0.33957 | 15.27107 | 5.979485 |
| 7scenes | p3_sketch10_mid21 | fire | 128 | 53.46129 | 94.12853 | 0.05048 | 0.05362 | 0.05205 | 67.57478 | 6.010128 |
| 7scenes | p3_sketch10_mid21 | office | 128 | 14.00098 | 77.48524 | 0.04839 | 0.05093 | 0.04966 | 59.49136 | 5.992724 |
| 7scenes | p3_sketch10_mid21 | pumpkin | 128 | 10.44948 | 77.61647 | 0.46768 | 0.09927 | 0.28347 | 24.33280 | 5.991258 |
| 7scenes | p3_sketch10_mid21 | redkitchen | 128 | 8.63271 | 79.38115 | 0.07760 | 0.07802 | 0.07781 | 49.02702 | 6.007118 |
| eth3d | p3_sketch10_mid21 | courtyard | 38 | 24.32432 | 84.68468 | 0.30547 | 0.21240 | 0.25893 | 54.88579 | 1.365405 |
| eth3d | p3_sketch10_mid21 | delivery_area | 40 | 17.39316 | 80.70513 | 0.69505 | 1.50209 | 1.09857 | 11.07905 | 1.400121 |
| eth3d | p3_sketch10_mid21 | electro | 39 | 39.18129 | 92.69456 | 0.24830 | 5.06163 | 2.65496 | 61.28422 | 1.393598 |
| eth3d | p3_sketch10_mid21 | facade | 76 | 32.94737 | 89.37895 | 0.24836 | 0.23427 | 0.24131 | 64.47718 | 2.818408 |
| eth3d | p3_sketch10_mid21 | kicker | 31 | 44.65950 | 93.46237 | 0.07057 | 0.06935 | 0.06996 | 98.92996 | 1.110156 |
| eth3d | p3_sketch10_mid21 | office | 26 | 21.64103 | 79.29231 | 0.09011 | 0.08669 | 0.08840 | 99.06175 | 0.940537 |
| eth3d | p3_sketch10_mid21 | pipes | 14 | 24.90842 | 86.15385 | 0.25476 | 0.26996 | 0.26236 | 71.44005 | 0.533921 |
| eth3d | p3_sketch10_mid21 | playground | 32 | 8.46774 | 65.50403 | 0.42092 | 0.92382 | 0.67237 | 36.15653 | 1.121644 |
| eth3d | p3_sketch10_mid21 | relief | 19 | 14.23002 | 77.68031 | 0.60456 | 0.62697 | 0.61577 | 14.23524 | 0.745695 |
| eth3d | p3_sketch10_mid21 | relief_2 | 20 | 5.96491 | 42.17544 | 0.46920 | 0.55621 | 0.51270 | 17.96651 | 0.781133 |
| eth3d | p3_sketch10_mid21 | terrains | 42 | 8.40108 | 53.09717 | 0.39422 | 0.33376 | 0.36399 | 57.35091 | 1.484057 |
| 7scenes | p3_sketch10_mid19 | heads | 128 | 46.98983 | 93.18365 | 0.03473 | 0.03483 | 0.03478 | 81.31842 | 5.901098 |
| 7scenes | p3_sketch10_mid19 | chess | 128 | 20.26739 | 88.45883 | 0.14380 | 0.08659 | 0.11520 | 36.37283 | 5.910801 |
| 7scenes | p3_sketch10_mid19 | stairs | 128 | 4.79413 | 79.20522 | 0.23265 | 0.46621 | 0.34943 | 14.50805 | 5.885148 |
| 7scenes | p3_sketch10_mid19 | fire | 128 | 52.68209 | 93.76804 | 0.05080 | 0.05447 | 0.05264 | 65.05470 | 5.866938 |
| 7scenes | p3_sketch10_mid19 | office | 128 | 13.89436 | 76.90822 | 0.05359 | 0.05755 | 0.05557 | 55.46492 | 5.908309 |
| 7scenes | p3_sketch10_mid19 | pumpkin | 128 | 10.47408 | 77.60663 | 0.45942 | 0.10151 | 0.28046 | 22.94434 | 5.875388 |
| 7scenes | p3_sketch10_mid19 | redkitchen | 128 | 8.75574 | 79.71867 | 0.07893 | 0.07664 | 0.07778 | 48.85447 | 5.884883 |
| eth3d | p3_sketch10_mid19 | courtyard | 38 | 23.70792 | 82.89711 | 0.17164 | 0.13680 | 0.15422 | 83.68454 | 1.337670 |
| eth3d | p3_sketch10_mid19 | delivery_area | 40 | 17.73504 | 81.95299 | 0.73617 | 1.56134 | 1.14876 | 12.14035 | 1.410844 |
| eth3d | p3_sketch10_mid19 | electro | 39 | 38.55151 | 93.09042 | 0.20626 | 5.11229 | 2.65928 | 62.50683 | 1.366157 |
| eth3d | p3_sketch10_mid19 | facade | 76 | 31.06433 | 88.77895 | 0.30938 | 0.32152 | 0.31545 | 51.22929 | 2.804093 |
| eth3d | p3_sketch10_mid19 | kicker | 31 | 45.30466 | 93.06810 | 0.06451 | 0.05855 | 0.06153 | 99.27221 | 1.098558 |
| eth3d | p3_sketch10_mid19 | office | 26 | 19.17949 | 78.65641 | 0.08201 | 0.07696 | 0.07948 | 99.36935 | 0.942389 |
| eth3d | p3_sketch10_mid19 | pipes | 14 | 19.41392 | 84.46886 | 0.20333 | 0.24052 | 0.22192 | 81.52642 | 0.538750 |
| eth3d | p3_sketch10_mid19 | playground | 32 | 8.33333 | 59.38844 | 0.41708 | 0.95134 | 0.68421 | 32.86962 | 1.114299 |
| eth3d | p3_sketch10_mid19 | relief | 19 | 9.94152 | 74.63938 | 0.76093 | 0.82111 | 0.79102 | 1.74237 | 0.745995 |
| eth3d | p3_sketch10_mid19 | relief_2 | 20 | 4.56140 | 43.43860 | 0.42998 | 0.54641 | 0.48819 | 31.40358 | 0.762208 |
| eth3d | p3_sketch10_mid19 | terrains | 42 | 6.19435 | 49.24506 | 0.44400 | 0.38196 | 0.41298 | 59.55489 | 1.496137 |

原始JSON、逐层预算/状态、峰值allocated/reserved、输入帧表、源码哈希及GPU监控均归档。

## 具体哪些Key被找回

在独立校准序列预选的1536条高概率遗漏边中，917条同边数替换降低当前层Wo输出误差，其中876条来自仍有Rep的簇，799条上一层不在历史集合。新主候选实际保留其中288条（31.41%）；这不是全局Key集合的覆盖率，也不是最终几何收益。

对这些正收益候选，Probe0真实概率/同Probe估计概率的中位数为8.33；这是偏向重要遗漏边的条件统计，不能冒充全部边的估计误差。Head-mean Value差异能量/逐Head差异能量中位数为0.0604，反映跨Head抵消。critical_keys.csv保存具体层、组、原帧与Patch坐标、真实/估计概率、替换收益及新Mask是否保留。

完整分析见key_impact_report.md；候选池只有每组8条，离线单边收益不相加，轨迹不同也不能单独归因最终F1。
