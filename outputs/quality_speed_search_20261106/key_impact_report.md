# 关键Key：独立RGB上的固定边数反事实

{
  "current": {
    "candidate_edges": 1536,
    "positive_single_edge_swaps": 917,
    "positive_approximate_cluster": 876,
    "positive_dropped_cluster": 41,
    "positive_never_previous_history": 799,
    "median_probe0_true_over_prior": 8.334571626537624,
    "fraction_probe0_underestimated_gt10": 0.4713302752293578,
    "median_head_mean_distance_over_headwise_distance": 0.060396059666890725,
    "mean_local_relative_mse": 0.010035470341487477
  },
  "p3_sketch10_mid21": {
    "candidate_edges": 1536,
    "positive_single_edge_swaps": 989,
    "positive_approximate_cluster": 987,
    "positive_dropped_cluster": 2,
    "positive_never_previous_history": 862,
    "median_probe0_true_over_prior": 8.294145194322947,
    "fraction_probe0_underestimated_gt10": 0.4627492130115425,
    "median_head_mean_distance_over_headwise_distance": 0.061486735802803914,
    "mean_local_relative_mse": 0.008301824385853251
  }
}

{
  "old_positive_candidate_edges": 917,
  "retained_by_new": 288,
  "fraction": 0.31406761177753545,
  "single_edge_gain_weighted_coverage": 0.26638495977096693,
  "interpretation": "Conditional coverage of8-per-group candidate pool;single-edge gains are not additive joint output improvement. Trajectories differ, so this is support overlap, not an isolated causal estimate of final geometry."
}

此分析固定ScanNet64帧、24层、8组×4Patch Query；每组仅考察8条高概率遗漏Key。真实概率是同一稀疏轨迹QKV下的完整Key Softmax，不是独立Dense模型轨迹。估计prior针对Probe0，因此只与Probe0真实概率对齐；四Probe均值另存。

单Key替换保持边数不变，精确重算当前补偿加权Softmax输出及Wo投影误差。它可能超出原四状态可表达的Mask；收益是当前层局部的，不能当作最终相机/F1因果收益，也不能累加当作多Key共同收益。

head_mean距离小于逐Head距离，说明跨Head抵消可能掩盖Value差异；新Head草图保留Head分离方向。遗漏原因区分为：整个簇被Drop，或簇仍有Rep但漏掉具体Key。后者说明仅增加簇数/只确保每簇一Rep不足。无历史边的新峰值也不可能仅靠历史概率校准找回。

critical_keys.csv记录层、Query组、原帧与Patch坐标、所属簇/Rep、是否上一层历史Key、Probe0真概率与估计、固定预算替换收益及新方案是否实际保留。所有六项几何指标N/A，未读取GT。
