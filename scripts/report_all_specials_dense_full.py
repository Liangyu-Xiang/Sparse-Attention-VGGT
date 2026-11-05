"""Full ETH3D quality and matched heads128 timings for all-special Query protection."""
import csv,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/eth3d_all_specials_dense_full_20261105'
BASE=ROOT/'outputs/eth3d_layout_fixed_full_20261105'
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def report():
 from scripts.run_eth3d_layout_fix_evaluation import verify
 from scripts.quality_metrics import quality_markdown
 manifest=read(OUT/'frozen_manifest.json');variant='all_specials_dense';plan=dict(manifest['configs'][variant],source_sha256=manifest['source_sha256']);verify(plan)
 config=read(Path(plan['config']));base_config=read(BASE/'frozen_config.json')
 assert config==base_config|dict(dense_special_queries=5),'Only camera-only -> all-special Queries is allowed'
 candidate=OUT/'quality'/variant;rows=[];audits=[];runtimes=[];grouped={'layout_fixed_camera_dense':[],'all_specials_dense':[]}
 queue=read(OUT/'queue.json');assert not queue['pending'] and not read(OUT/'execution.json')['failures']
 for scene in manifest['scenes']:
  before=read(BASE/'runs'/scene/'result.json');after=read(candidate/'runs'/scene/'result.json')
  assert before['quality']['complete'] and after['quality']['complete']
  checks={k:before['input'][k]==after['input'][k] for k in ['image_paths','frame_indices','shape','checkpoint_sha256']}
  checks['same_quality_protocol']=before['quality_protocol']==after['quality_protocol']
  checks['unchanged_evaluation']=all(after['source_sha256'].get(k)==v for k,v in before['source_sha256'].items() if k.startswith('evaluation/'))
  checks['frozen_source']=all(after['source_sha256'].get(k)==v for k,v in manifest['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
  checks['frozen_config']=all(after['config'].get(k)==v for k,v in config.items())
  claim=next(c for c in queue['claims'] if c['scene']==scene);gpu=claim['gpu'];monitor=read(OUT/f'worker{gpu}_gpu.json')
  checks['permitted_gpu']=gpu in [4,5,6];checks['same_hardware_uuid']=after['hardware']['gpu_uuid'].removeprefix('GPU-')==monitor['gpu_uuid'].removeprefix('GPU-')
  checks['isolated_gpu']=monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'];checks['forward_complete']=claim['status']=='done' and claim['returncode']==0
  layers=read(candidate/'runs'/scene/'layers.json');assert len(layers)==24
  checks['all_special_queries_dense_every_layer']=all(l['dense_special_query_count']==5*after['frames'] for l in layers)
  checks['all_special_keys_retained_every_layer']=all(l['semantic_special_keys']==5*after['frames'] and l['special_keys']>=l['semantic_special_keys'] for l in layers)
  checks['native_budget_count_correct']=max(l.get('native_cost_count_error_mean',0) for l in layers)==0
  assert all(checks.values()),(scene,checks);audits.append(dict(scene=scene,checks=checks))
  for name,d in [('layout_fixed_camera_dense',before),('all_specials_dense',after)]:
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
   grouped[name].append(v);rows.append(dict(scene=scene,method=name,frames=d['frames'],**v))
  runtimes.append(dict(scene=scene,gpu=gpu,frames=after['frames'],cold_full_forward_seconds=after['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in after['timing']['samples']),actual_unique_key_density_mean=statistics.mean(l['actual_unique_density'] for l in layers),warmup=0,repeats=1,speedup_vs_dense=None,speedup_unavailable='ETH3D quality only; no paired warmed Dense ETH3D timing'))
 macro={name:{k:statistics.mean(v[k] for v in values) for k in FIELDS} for name,values in grouped.items()};delta={k:macro[variant][k]-macro['layout_fixed_camera_dense'][k] for k in FIELDS}
 assert len(audits)==11 and sum(r['frames'] for r in runtimes)==377
 quiet=(OUT/'speed_heads128_quiet/execution.json').exists()
 speed_dir=OUT/('speed_heads128_quiet' if quiet else 'speed_heads128')
 monitor_path=OUT/('speed_quiet_gpu.json' if quiet else 'speed_gpu.json')
 while not (speed_dir/'execution.json').exists():
  if monitor_path.exists():
   monitor=read(monitor_path)
   if monitor['returncode']!=0:raise RuntimeError('Timing failed; retained quality results and timing logs')
  time.sleep(2)
 assert read(speed_dir/'execution.json')['complete']
 monitor=read(monitor_path) if monitor_path.exists() else None
 while monitor is None:
  time.sleep(.5)
  if monitor_path.exists():monitor=read(monitor_path)
 assert monitor['gpu']==6 and monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
 times={name:read(speed_dir/name/'timing_result.json') for name in ['all_specials_dense','layout_fixed','dense','saf3r']};dense=times['dense'];speed_rows=[]
 assert times['saf3r']['saf3r']['topk']==4 and times['saf3r']['saf3r']['lazy_dino_topk']
 for name,d in times.items():
  assert d['input']==dense['input'] and d['hardware']==dense['hardware']
  assert d['timing']['warmup']==1 and d['timing']['repeats']==3 and not d['timing']['capture']
  assert all(d['source_sha256'].get(k)==v for k,v in manifest['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
  seconds=d['timing']['median_seconds'];speed=dense['timing']['median_seconds']/seconds
  d['quality']['speedup_vs_vggt']=speed;d['quality']['speedup_unavailable']=None
  d['paired_timing']=dict(dense=str(speed_dir/'dense/timing_result.json'),same_inputs_hardware_protocol=True)
  save(speed_dir/name/'timing_result.json',d);save(speed_dir/name/'quality.json',d['quality']);(speed_dir/name/'quality.md').write_text(quality_markdown(d['quality']))
  speed_rows.append(dict(method=name,seconds=seconds,speedup_vs_dense=speed,time_over_layout_fixed=seconds/times['layout_fixed']['timing']['median_seconds'],time_over_saf3r=seconds/times['saf3r']['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),actual_unique_key_density_mean=(statistics.mean(l['actual_unique_density'] for l in read(speed_dir/name/'layers.json')) if name in ['all_specials_dense','layout_fixed'] else None),timing=d['timing'],hardware=d['hardware'],quality=d['quality']))
 relative=times[variant]['timing']['median_seconds']/times['layout_fixed']['timing']['median_seconds'];saf_ratio=times[variant]['timing']['median_seconds']/times['saf3r']['timing']['median_seconds']
 speed=dict(batch='after_cpu_quality' if quiet else 'concurrent_cpu_quality',initial_batch_reference='speed_heads128/paired_summary.json',scope='7Scenes heads128 only, not all7-scene macro; GPU6 RTX4090,1 warmup+3 repeats, median complete forward including reset/clustering/routing/history/special-query dense correction and SAF3R Top-K reset',rows=speed_rows,time_ratio_vs_layout_fixed=relative,time_ratio_vs_saf3r=saf_ratio,no_speed_sacrifice=relative<=1.,within_saf3r_10percent=saf_ratio<=1.1,timing_variability={n:dict(min_seconds=min(s['seconds'] for s in d['timing']['samples']),max_seconds=max(s['seconds'] for s in d['timing']['samples'])) for n,d in times.items()},control_policy='Prior explicit user authorization for timing-only Dense/SAF3R/layout-fixed controls; no unchanged control quality reruns')
 summary=dict(dataset='eth3d',complete=True,scenes=11,total_frames=377,method=variant,macro=macro,delta_vs_layout_fixed=delta,per_scene=rows,audits=audits,runtime=runtimes,paired_7scenes_heads128_speed=speed,units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),speedup_vs_dense=None,speedup_unavailable='No paired warmed Dense ETH3D timing; matched7Scenes speed stored separately',adopted=False,intervention='dense_special_queries1->5 only; special Keys already retained, all camera/register Query outputs now use every Key; sparse observer remains the history stream')
 save(OUT/'summary.json',summary);save(speed_dir/'paired_summary.json',speed)
 with (OUT/'per_scene.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['scene','method','frames']+FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# 全部特殊token双向保护：完整ETH3D质量与7Scenes速度','',
 '11/11个既有评测场景，共377帧；本批全部重新推理和评分。每场景上限128帧，uniform seed0；既有协议排除terrace/meadow，本轮不新增过滤。对照质量复用完整布局修复版结果。','',
 '唯一方法变更：dense_special_queries从1改为5。每帧1个camera和4个register token始终作为Key保留，且现在作为Query均对全部Key计算。普通Query仍对全部特殊Key与原方法选出的patch Key计算。保留率、聚类、损失及权重均未调整；未重新启用已淘汰的partial states。','',
 '此实现先运行稀疏observer，再以完整SDPA覆盖所有特殊Query输出；历史流仍来自稀疏observer。它保证最终输出的双向支持关系，但不是底层kernel物理剔除特殊Query；重复计算完整计入速度。','',
 'AUC/F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，非平方距离。以下为11场景等权平均。','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for name,v in macro.items():lines.append('| '+name+' | '+cells(v)+' |')
 lines+=['',f"相对布局修复版：AUC@3 {delta['auc03']*100:+.4f}个百分点，AUC@30 {delta['auc30']*100:+.4f}个百分点，F1 {delta['f1']*100:+.4f}个百分点；CD {delta['cd']:+.6f}m。",'',
 '## 同批完整前向计时','',speed['scope'],'',
 '| Method | Median s | Dense speedup | Time / layout-fixed | Time / SAF3R | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
 for r in speed_rows:lines.append(f"| {r['method']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.4f}x | {r['time_over_layout_fixed']:.4f} | {r['time_over_saf3r']:.4f} | {r['peak_allocated_gib']:.4f} |")
 lines += ['',f"满足不牺牲布局修复版速度：{speed['no_speed_sacrifice']}；满足SAF3R耗时上浮不超过10%：{speed['within_saf3r_10percent']}。",'',
 '首组计时保存在speed_heads128及report_initial_timing.md；当时CPU质量评分并行，SAF3R样本8.388–9.460秒，不能据其中位数直接宣称稳定速度优势。补测在全部CPU质量任务结束后执行，顺序为布局修复版→新版本→Dense→SAF3R，仍为同GPU、1暖机+3完整前向，未修改任何方法参数。若补测已完成，本报告主表使用该组；首组结果完整保留，没有删除或混合样本。','',
 '补测未观察到CPU质量评分进程。计时仍只有每种方法3个暖机后样本，表中速度结论仅代表该输入与本批测量；不将首组的较快中位数当作已经证实的稳定提速。两组结果均保留，未据结果重新调参。','',
 'SAF3R为原版Top-4、lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO Top-K或Top-128。四组计时只有RGB完整前向，六项几何质量为N/A；未混用论文分数和H100时间。控制方法仅计时重跑，依据用户此前明确授权。','',
 '## 逐场景六项质量','',
 '| Scene | Method | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['scene']} | {r['method']} | {r['frames']} | "+cells(r)+' |')
 protocol=read(candidate/'runs'/manifest['scenes'][0]/'result.json')['quality_protocol'];rates=config['layer_retentions']
 lines+=['','## 协议与运行记录','',f"重建阈值{protocol['f1_threshold_m']}m，下采样/voxel {protocol['downsample_m']}m，sdf_trunc {protocol['sdf_trunc_m']}m，mesh采样{protocol['mesh_sample_points']}点。沿用unposed Umeyama/RANSAC scale+pose对齐、random_state42、原mask和深度处理。",'',f"逐层patch保留预算为{rates}，均值{statistics.mean(rates):.4f}；fine cluster fraction={config['cluster_fraction']}。实际密度另存layers.json与下表；全量特殊Query会增加实际计算覆盖率，不伪装为相同等效密度。",'',
 '仅GPU4/5/6；ETH3D每场景0暖机、1次完整前向。以下为包含JIT/缓存建立的冷前向记录，不用于速度比较；ETH3D Dense暖机加速为N/A，因为只在7Scenes测速度。','',
 '| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |','|---|---:|---:|---:|---:|']
 for r in runtimes:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['actual_unique_key_density_mean']:.6f} |")
 lines+=['','所有场景通过相同帧表、权重SHA、预处理、评测协议与源码、冻结配置、GPU范围和隔离审计。每一层均验证5×帧数个特殊Query为全量、全部特殊Key保留，且预算计数无误。','',
 '这是单次完整质量评测；既有实现重复前向存在数值波动，不能把小幅差异当成统计显著收益。完整原始指标、源码快照与计时样本均保存；当前报告不自动替换已保留的方法。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');save(OUT/'progress.json',dict(stage='complete',completed=manifest['scenes'],failures=[]))
 print('ALL_SPECIALS_DENSE_COMPLETE',json.dumps(dict(macro=macro,delta=delta,time_ratio_vs_layout_fixed=relative,time_ratio_vs_saf3r=saf_ratio)),flush=True)
if __name__=='__main__':report()
