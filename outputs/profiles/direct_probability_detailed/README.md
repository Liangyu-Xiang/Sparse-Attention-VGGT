本次仅增加可选精细profiling，不改变正式算法。参考分支/提交：codex/direct-probability-history-20261104，d10fd3e6。见report.md、audit.json、static_validation.json。

所有用户要求的原始文件均已写入共享项目目录：
/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/profiles/direct_probability_detailed/

本提交保留代码、报告、小型汇总和原始文件SHA256清单。40MiB逐window、62MiB allocation、25MiB runtime事件、1.3GiB raw trace及114MiB gzip trace等大文件不重复进入Git。raw_artifact_manifest.json记录完整路径、大小、SHA256。原始profiler_trace.json指向本机完整JSON；profiler_trace.json.gz为共享文件系统中的持久副本，可解压后用Chrome/Perfetto打开。

纯速度仅用heads128序列，1预热+3次完整前向中位数。所有成功GPU采集在同一物理GPU6单进程独占，监控脚本无CUDA上下文。Dense与SAF3R未运行，配对Dense速度N/A。没有重新评测几何质量，六指标为N/A且说明原因。

耗时有三个独立视图：低开销CUDA Events互斥A/B/C、重标记Graph的逐语句高开销阶段、无marker的生产Kernel trace self。不能跨视图加总。特别是Graph Replay包含状态、预算、mask和补偿；组件微基准不可相加。

可复现入口：scripts/evaluate_reuse_exploration.py --profile-detail（默认关闭）。按报告要求分开稳定时间、细粒度标记和无marker算子trace时，使用scripts/profile_direct_probability_coarse.py、scripts/profile_direct_detail_stages.py、scripts/trace_direct_probability.py及scripts/profile_direct_probability_components.py。命令参数均为--config、--input（基线input.json）、--output-dir。用scripts/run_gpu_lease.py --single-process --strict-contexts启动并固定同一物理GPU；不要同时运行多个GPU模型程序。

原始trace很大，应先导出本机/tmp，再压缩到共享路径。scripts/summarize_direct_trace.py以流式JSON解析生成kernel/runtime/allocation表，避免整体载入。
