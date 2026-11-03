"""Combine all frozen variants without selecting a test-score winner."""
import argparse,csv,json,shutil
from pathlib import Path
K=['auc03','auc30','acc','comp','cd','f1']
def main():
 p=argparse.ArgumentParser();p.add_argument('--outputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=True,parents=True)
 reports={rev:json.loads((a.outputs/f'efficiency_final_{rev}_20261103/summary.json').read_text()) for rev in ['v7','v8']}
 macros=[v for v in reports['v8']['macro'] if v['method']!='candidate_v8']+[v for v in reports['v7']['macro'] if v['method']=='candidate_v7']+[v for v in reports['v8']['macro'] if v['method']=='candidate_v8']
 lines=['# 推理 GPU 优化与完整验收报告（2026-11-03）','','本轮未达到“质量不降低且不慢于 Dense”的要求。原实现继续作为默认路径；优化代码以独立配置提供，未将失败候选替换为正式方法。','','## 完整前向速度与显存','', '同一 heads 序列128帧、RTX 5090 D；每个批次交替执行原实现与候选，各一次预热、三次完整前向取中位数。GPU进程独占监测未发现同卡重叠。两个候选属于不同计时批次，不能把它们的绝对差当作严格配对消融。','', '| 批次/候选 | 同批原实现 s | 候选 s | 原实现/候选 | 独立 profile 峰值 allocated GiB | Dense 配对加速比 |','|---|---:|---:|---:|---:|---:|']
 for rev in reports:
  v=reports[rev];t=v['timing'];lines.append(f"| {rev} | {t['reference_seconds']:.5f} | {t['candidate_seconds']:.5f} | {t['speedup_vs_reference']:.3f}× | {v['profile']['peak_allocated_gib']:.3f} | N/A |")
 lines+=['','按用户要求，Dense与SAF3R均未重跑。历史 Dense heads128 完整前向为12.2160s，来自20261029批次；两条候选均高于这个历史参考，不能声称达到不慢于Dense的目标。历史值不是本轮配对测量，相对Dense的正式加速比为N/A。Dense作为自身参照为1.00×。','', '独立profile含事件插桩，其完整耗时不是三次中位数。配对进程的Graph静态池在切换方法后继续存活，原始峰值保留在timing.json中，不能拿共享进程峰值证明独立显存节省。','', '## 六项质量，全部场景等权宏平均','', 'AUC/F1为百分比，ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离。Dense、SAF3R、reference均复用已有完整质量结果，仅候选重新评测。7Scenes全七场景128帧；ETH3D全11场景14–76帧。','', '| 数据集 | 方法 | 场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for ds in ['7scenes','eth3d']:
  for v in [v for v in macros if v['dataset']==ds]:lines.append(f"| {ds} | {v['method']} | {v['scenes']} | {v['auc03']*100:.4f} | {v['auc30']*100:.4f} | {v['acc']:.6f} | {v['comp']:.6f} | {v['cd']:.6f} | {v['f1']*100:.4f} |")
 lines+=['','7Scenes重建阈值0.05m/下采样0.0078125m；ETH3D阈值0.25m/下采样0.0390625m。SAF3R为本地lazy first-correspondence-head Top-4，不标为论文DINO配置。所有候选与reference的输入路径、帧索引、预处理形状、checkpoint实际路径和完整质量协议逐场景核对一致，见各版本protocol_audit.json。','', '## 验收结论','']
 for rev in reports:
  for v in reports[rev]['changes']:
   lines.append(f"- {rev} / {v['dataset']}：未通过1%退化门槛的指标为 {', '.join(v['failed_one_percent_gate']) or '无'}；六项相对变化百分比：{json.dumps(v['relative_change_percent'],ensure_ascii=False)}。")
 lines+=['','v8的ETH3D重建四项均优于reference，但位姿AUC@3/AUC@30未通过，因此整体仍不合格。1%门槛也不是数学上零质量变化的证明。没有根据测试真值调整预算、状态、阈值或概率模型，也没有挑选最好重复结果。','', '## 实现与正确性','', '- 预算固定20%，状态集合、损失、历史概率估计和概率质量转移公式保持原设定；每层重新处理当前Value、聚类/Rep、历史概率及分配。','- CUDA Graph仅缓存执行图和缓冲区，每层刷新几何数据，每个窗口刷新prior、历史Key、Exact资格；缓存容量和shape桶仅影响执行调度。','- 合并状态损失、gate、支持打包和补偿访存；固定探针索引按输入形状准备一次；范围与预算检查仍执行，合并到每层读取。','- v7在一次选中边QK中同时收集Probe Logit和grid局部归约，避免历史生成再次扫描raw logits求分母，并对Attention作分区softmax合并。FP32小测试误差低于1e-5；BF16 Attention相对L2约0.0004–0.0011，但完整质量仍失败。','- v8保留原串行Attention计算顺序，只将num_stages设为1；FP32、BF16以及真实张量尺寸的输出、grid概率、Probe Logit逐位一致。v8历史分母采用参考扫描内核，因此比v7多约1.5秒诊断开销。','- 新增融合状态内核曾错误假设Exact资格掩码连续。真实路径的掩码来自转置稀疏矩阵乘法并保留列布局，导致v6读取错误。已在内核入口转为连续布局，增加非连续掩码测试；v6失败结果完整保留。','- 修复后ETH3D relief真实序列抽查每层首窗口：23个状态分配窗口的保留Key集合与参考一致，Graph与即时计算集合一致；此范围之外没有宣称逐位等价。','- v7/v8剩余的完整质量差异尚未被唯一定位。v8恢复原Attention顺序后重建指标改善，但位姿仍失败，不能把全部退化归因于分区Attention。融合归约和coarse归一化仍有浮点差异，原实现也含非确定性原子归约；这些是待隔离因素，不是豁免质量门槛的理由。','', '## 互斥耗时','', '以下为v8单独profile，只列exclusive时间。其完整前向含插桩；此表用于定位瓶颈，不代替配对中位数。父级/子级不重复相加。','', '| 操作 | 秒 | 占完整前向 |','|---|---:|---:|']
 profile=json.loads((a.outputs/'efficiency_profile_v8_20261103/profile.json').read_text());total=0
 for n,v in sorted(profile['groups'].items(),key=lambda kv:-kv[1]['exclusive_seconds']):
  t=v['exclusive_seconds'];total+=t;lines.append(f"| {n} | {t:.6f} | {t/profile['wall_seconds']*100:.2f}% |")
 other=profile['wall_seconds']-profile['global_attention_seconds'];lines.append(f"| 全局Attention之外的完整前向剩余部分 | {other:.6f} | {other/profile['wall_seconds']*100:.2f}% |")
 lines+=['',f"完整profile {profile['wall_seconds']:.6f}s；global Attention {profile['global_attention_seconds']:.6f}s；exclusive求和误差 {profile['sum_error_seconds']:.12f}s。Graph内部损失、预算、打包、补偿统一计入reuse_and_budget，不再叠加独立内核时间。",'', '独立CUDA活动诊断（v7）中，Sparse Attention内核约6.168s，融合状态内核约2.113s，历史缓存_store_local约0.457s。25908次_choose预算内核合计约0.267s。此活动诊断与上述v8profile不同批次，不能相加；活动区间并集与内核时长总和同时保存，约1.656ms重叠已显式记录。','', 'Nsight Compute硬件计数器读取被系统拒绝（ERR_NVGPUCTRPERM），没有修改主机权限。报告不声称已测得SM占用率或带宽利用率。','', '## 交付与复现','', '- 独立配置：configs/evaluation/reuse_efficiency_v6.json（修复后的分区路径，v7源码）和reuse_efficiency_v8.json（串行staging路径）；默认配置不变。v6失败源码与v7修复源码由快照区分，不能只凭相同配置文件名区分这两轮。','- scripts/benchmark_efficiency_full.py：同输入原实现/候选交替完整前向计时，明确不调用Dense。','- scripts/profile_efficiency_candidate.py：分层事件树exclusive计时；scripts/profile_efficiency_kernels.py：独立CUDA活动汇总。','- scripts/run_efficiency_quality.py：冻结源码SHA256，单次完整pose+recon_unposed质量评测，严格GPU进程监测。','- 原始六指标、逐场景CSV、源码差异快照、SHA256、质量协议审核、完整前向计时及显存、失败诊断保存在outputs/inference_efficiency_20261103。','- 共享union bank、连续跨grid tile、寄存器限制、延迟Value读取、streaming写入等微基准未获得足够端到端收益，未纳入候选默认配置。无完整结果的诊断均标N/A，不当作质量评测。']
 (a.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
 (a.output_dir/'summary.json').write_text(json.dumps(dict(complete=True,accepted=False,quality_and_speed_goal_achieved=False,macro=macros,variants=reports),indent=2)+'\n')
 combined=[]
 for rev in ['v7','v8']:
  data=list(csv.DictReader((a.outputs/f'efficiency_final_{rev}_20261103/per_scene.csv').open()));combined += [v for v in data if rev=='v8' or v['method']=='candidate_v7']
 with (a.output_dir/'per_scene.csv').open('w') as f:
  w=csv.DictWriter(f,list(combined[0]),lineterminator='\n');w.writeheader();w.writerows(combined)
 print(a.output_dir/'report.md')
if __name__=='__main__':main()
