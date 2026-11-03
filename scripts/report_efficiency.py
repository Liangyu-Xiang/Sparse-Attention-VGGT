"""Report frozen engineering trials; never fill missing paired Dense timing."""
import argparse,csv,json,statistics,hashlib
from pathlib import Path
K=['auc03','auc30','acc','comp','cd','f1']

def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--outputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--revision',default='v6');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);root=a.outputs;revision=a.revision;candidate='candidate_'+revision
 previous=list(csv.DictReader((root/'grid_local_final_20261102/per_scene.csv').open()))
 rows=[];audit=[]
 for v in previous:
  if v['method'] not in ('dense','saf3r','grid_local'):continue
  row={**v,'method':'reference' if v['method']=='grid_local' else v['method'],'reused':True}
  for k in K:row[k]=float(row[k])
  rows.append(row)
 for ds,count in [('7scenes',7),('eth3d',11)]:
  files=sorted((root/('efficiency_quality_'+revision+'_20261103')/ds).glob('*/efficiency/result.json'));assert len(files)==count
  for file in files:
   r=json.loads(file.read_text());scene=r['scene'];q=json.loads((file.parent/'quality.json').read_text());assert q['complete'] and all(q['values'][k] is not None for k in K)
   source=next(v for v in previous if v['dataset']==ds and v['scene']==scene and v['method']=='grid_local')
   old=json.loads(Path(source['source']).read_text());same_input={k:v for k,v in r['input'].items() if k!='checkpoint'}=={k:v for k,v in old['input'].items() if k!='checkpoint'};same_protocol=r['quality_protocol']==old['quality_protocol']
   model_a={k:v for k,v in r['model_config'].items() if k!='ckpt_path'};model_b={k:v for k,v in old['model_config'].items() if k!='ckpt_path'}
   paths=[Path(z['model_config']['ckpt_path']).resolve() for z in [r,old]]
   same_checkpoint=paths[0]==paths[1]
   assert same_input and same_protocol and model_a==model_b and same_checkpoint,(ds,scene,paths)
   audit.append(dict(dataset=ds,scene=scene,same_input=same_input,same_quality_protocol=same_protocol,same_model=model_a==model_b,same_checkpoint=same_checkpoint,checkpoint=str(paths[0]),quality_protocol=r['quality_protocol']))
   mon=json.loads((file.parent.parent/'lease_monitor.json').read_text());assert not mon['same_gpu_overlap_detected'] and mon['returncode']==0
   rows.append(dict(dataset=ds,scene=scene,method=candidate,frames=r['frames'],**q['values'],seconds=r['timing']['median_seconds'],peak_allocated_gib=r['timing']['samples'][0]['peak_allocated_gib'],reused=False,source=str(file),dense_speedup=None,dense_speedup_reason='User prohibited Dense rerun; historical timing is not paired'))
 macro=[]
 for ds in ['7scenes','eth3d']:
  for method in ['dense','saf3r','reference',candidate]:
   group=[v for v in rows if v['dataset']==ds and v['method']==method];assert len(group)==(7 if ds=='7scenes' else 11)
   macro.append(dict(dataset=ds,method=method,scenes=len(group),**{k:statistics.mean(v[k] for v in group) for k in K},dense_speedup=1 if method=='dense' else None))
 changes=[]
 for ds in ['7scenes','eth3d']:
  old=next(v for v in macro if v['dataset']==ds and v['method']=='reference');new=next(v for v in macro if v['dataset']==ds and v['method']==candidate)
  relative={k:100*(new[k]/old[k]-1) for k in K};failed=[k for k in K if (relative[k]<-1 if k in ['auc03','auc30','f1'] else relative[k]>1)]
  changes.append(dict(dataset=ds,relative_change_percent=relative,failed_one_percent_gate=failed))
 timing=json.loads((root/('efficiency_full_'+revision+'_20261103/result.json')).read_text());profile=json.loads((root/('efficiency_profile_'+revision+'_20261103/profile.json')).read_text())
 seconds=timing['median_seconds'];speed=dict(reference_seconds=seconds['current'],candidate_seconds=seconds['candidate'],speedup_vs_reference=seconds['current']/seconds['candidate'],time_reduction_percent=100*(1-seconds['candidate']/seconds['current']),dense_speedup=None,dense_reason='User explicitly prohibited Dense rerun; historical Dense heads128 12.216022619977593 s (20261029) is not contemporaneous',historical_dense_seconds=12.216022619977593,goal_achieved=False)
 quality_failed=any(v['failed_one_percent_gate'] for v in changes)
 summary=dict(complete=True,quality_passed_one_percent_gate=not quality_failed,accepted_for_default=False,reason=('Quality gate failed; ' if quality_failed else 'Quality gate passed; ')+'speed target not achieved against historical Dense; reference remains default',macro=macro,changes=changes,timing=speed,profile=dict(wall_seconds=profile['wall_seconds'],peak_allocated_gib=profile['peak_allocated_gib'],sum_error_seconds=profile['sum_error_seconds']),sampling='7Scenes all seven scenes, 128 frames; ETH3D all eleven scenes, 14–76 frames; equal-scene macro averages',units=dict(auc03='fraction',auc30='fraction',acc='m',comp='m',cd='m',f1='fraction'),cd_definition='(ACC + Comp)/2, unsquared distances',saf3r_profile='Local lazy first-correspondence-head Top-4; not a claim of paper DINO configuration')
 dump(a.output_dir/'summary.json',summary);dump(a.output_dir/'protocol_audit.json',audit)
 fields=['dataset','scene','method','frames',*K,'seconds','peak_allocated_gib','reused','source','dense_speedup','dense_speedup_reason']
 with (a.output_dir/'per_scene.csv').open('w') as f:
  w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
 lines=['# 推理优化工程实验报告（2026-11-03）','',('结论：候选 v6 暂不替换默认实现。完整前向明显加快，但质量门槛未全部通过，且未达到不慢于 Dense 的要求。' if quality_failed else '结论：六项质量通过1%退化门槛，但未达到不慢于Dense的速度要求；默认路径保持原实现。'),'',f"同批 heads 128 帧、RTX 5090 D 完整前向：参考 {seconds['current']:.4f} s → 候选 {seconds['candidate']:.4f} s，参考/候选 {speed['speedup_vs_reference']:.3f}×，耗时减少 {speed['time_reduction_percent']:.2f}%。一次预热、三次完整前向中位数，严格监测同卡进程，无重叠。",'','Dense、SAF3R 未重跑。相对 Dense 的正式配对加速比为 N/A；历史 Dense heads128 为 12.2160 s（20261029），候选 18.1096 s 仍高于该参考，不能声称达到速度目标。ETH3D质量运行只有一次预热后一次前向，不能当作三次中位数。','', '## 六项质量指标','', 'AUC 和 F1 为百分比；ACC/Comp/CD 为米。所有行均为本地实测，Dense/SAF3R/参考复用原始结果；候选仅评测一次完整数据集。','', '| 数据集 | 方法 | 场景 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Dense 加速比 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for v in macro:lines.append(f"| {v['dataset']} | {v['method']} | {v['scenes']} | {v['auc03']*100:.4f} | {v['auc30']*100:.4f} | {v['acc']:.6f} | {v['comp']:.6f} | {v['cd']:.6f} | {v['f1']*100:.4f} | {'1.00×' if v['method']=='dense' else 'N/A'} |")
 lines+=['','CD=(ACC+Comp)/2，非平方距离。7Scenes 全七场景128帧；ETH3D 全11场景14–76帧，各场景等权宏平均，二者不得混为128帧测速。7Scenes阈值0.05m、下采样0.0078125m；ETH3D阈值0.25m、下采样0.0390625m。SAF3R 为本地 lazy first-correspondence-head Top-4。完整输入列表、checkpoint实际路径和协议一致性见 protocol_audit.json。','','## 质量门槛','']
 for c in changes:lines.append(f"- {c['dataset']}：六项相对变化 {json.dumps(c['relative_change_percent'],ensure_ascii=False)}；未通过1%门槛：{', '.join(c['failed_one_percent_gate']) or '无'}。")
 lines+=['','这些结果没有用于继续调预算或公式；所有预算、状态、概率估计和分层设置均保持冻结配置。数学等价的融合仍改变浮点归约顺序和 BF16 中间舍入；小规模数值测试通过不能代替最终几何质量验收。','','## 互斥时间分解','',f"单独 profile 完整前向 {profile['wall_seconds']:.4f}s，峰值 allocated {profile['peak_allocated_gib']:.3f}GiB。下表只列 exclusive 时间，父子事件不重复相加。总 global attention {profile['global_attention_seconds']:.6f}s，互斥项求和误差 {profile['sum_error_seconds']:.9f}s。",'', '| 操作 | 秒 | 占完整前向 |','|---|---:|---:|']
 for name,v in sorted(profile['groups'].items(),key=lambda kv:-kv[1]['exclusive_seconds']):
  t=v['exclusive_seconds'];lines.append(f"| {name} | {t:.6f} | {t/profile['wall_seconds']*100:.2f}% |")
 other=profile['wall_seconds']-profile['global_attention_seconds'];lines.append(f"| 全局 Attention 之外的剩余完整前向 | {other:.6f} | {other/profile['wall_seconds']*100:.2f}% |")
 lines+=['','graph 内部状态/预算/打包/补偿步骤未再拆开计时，统一归入 reuse_and_budget，避免与 graph replay 重复统计。attention_wrapper 为排序、索引和分配等包装开销的 exclusive 时间。','','## 实现变化与边界','','- 原配置继续走参考路径。独立 opt-in 配置开启分区稀疏 Attention、原始 Probe 证据顺带归约、融合状态损失与 mask 操作、形状探针索引缓存和 CUDA Graph 执行缓存。','- 每层重新生成当前 Value 聚类、中心/Rep、概率与分配；Graph 缓存执行图，每次刷新当前数据，不冻结上层状态。','- 原始历史概率、特殊 Token、逻辑 Key 顺序、补偿公式、预算和状态集合均保留。分区 softmax 和并行归约的数学公式相同，但浮点执行顺序有差异，候选未获得质量等价认证。','- 正式 Attention 不重算第二次选中边 QK；在第一次 QK 中保存 Probe Logit 与 grid 局部归约。历史缓存仍为 head-mean、原压缩格式。','- 共享 union bank、跨 grid 连续打包、改变寄存器上限、延迟 Value 读取等已做微基准；没有端到端有效收益的变种未纳入 v6。','- Nsight Compute 硬件计数器不可用（ERR_NVGPUCTRPERM），未修改主机权限。使用 CUDA Event 和 kernel 编译信息；不可伪称已测 SM/带宽利用率。','','## 显存与测量限制','',f"独立 candidate profile 峰值 allocated {profile['peak_allocated_gib']:.3f} GiB；配对计时中两个路径共享进程且 CUDA Graph 静态池持续存活，峰值不能作为两个独立进程的显存节省结论。逐次 peak 已保存在原始 JSON。质量运行逐场景独立进程的峰值见 per_scene.csv；reserved 与 allocated 不同。",'','完整数据与验证原始 JSON 另存本目录；源快照及 SHA256 保存在原实验目录。未经通过质量门槛和速度验收，不应将本候选标为正式替换。']
 (a.output_dir/'report.md').write_text(('\n'.join(lines)+'\n').replace('v6',revision).replace('18.1096',f"{seconds['candidate']:.4f}"))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
