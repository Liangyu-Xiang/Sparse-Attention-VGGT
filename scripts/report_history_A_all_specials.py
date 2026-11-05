"""Audit A+all-specials quality and matched speed without rerunning control quality."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_A_all_specials_20261105';BASE=ROOT/'outputs/rep_history_loss_20261105/full/quality/A'
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def report():
 from scripts.run_eth3d_layout_fix_evaluation import verify
 from scripts.quality_metrics import quality_markdown
 m=read(OUT/'frozen_manifest.json');name='A_all_specials';plan=dict(m['configs'][name],source_sha256=m['source_sha256']);verify(plan)
 cfg=read(Path(plan['config']));base_cfg=read(ROOT/'outputs/rep_history_loss_20261105/full/configs/A.json')
 assert cfg==base_cfg|dict(dense_special_queries=5)
 assert m['source_sha256']==read(ROOT/'outputs/rep_history_loss_20261105/full/frozen_manifest.json')['source_sha256'],'Only config changes; all native model/evaluation code must remain frozen'
 q=read(OUT/'queue.json');assert not q['pending'] and not read(OUT/'execution.json')['failures']
 grouped={'A_camera_dense':[],'A_all_specials':[]};rows=[];audits=[];runtime=[]
 for scene in m['scenes']:
  before=read(BASE/'runs'/scene/'result.json');after=read(OUT/'quality'/name/'runs'/scene/'result.json')
  assert before['quality']['complete'] and after['quality']['complete']
  checks={k:before['input'][k]==after['input'][k] for k in ['image_paths','frame_indices','shape','checkpoint_sha256']}
  checks['same_protocol']=before['quality_protocol']==after['quality_protocol'];checks['frozen_source']=all(after['source_sha256'].get(k)==v for k,v in m['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
  checks['frozen_config']=all(after['config'].get(k)==v for k,v in cfg.items())
  claim=next(c for c in q['claims'] if c['scene']==scene);monitor=read(OUT/f"worker{claim['gpu']}_gpu.json")
  checks['gpu_scope_isolation']=monitor['gpu'] in [4,5,6] and monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'];checks['uuid']=after['hardware']['gpu_uuid'].removeprefix('GPU-')==monitor['gpu_uuid'].removeprefix('GPU-')
  layers=read(OUT/'quality'/name/'runs'/scene/'layers.json');assert len(layers)==24
  checks['all_special_queries']=all(l['dense_special_query_count']==5*after['frames'] for l in layers)
  checks['all_special_keys']=all(l['semantic_special_keys']==5*after['frames'] and l['special_keys']>=l['semantic_special_keys'] for l in layers)
  checks['budget_count']=all(l.get('native_cost_count_error_mean',0)==0 for l in layers)
  assert all(checks.values()),(scene,checks);audits.append(dict(scene=scene,checks=checks))
  for method,d in [('A_camera_dense',before),('A_all_specials',after)]:
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
   grouped[method].append(v);rows.append(dict(method=method,scene=scene,frames=d['frames'],**v))
  runtime.append(dict(scene=scene,gpu=monitor['gpu'],frames=after['frames'],cold_full_forward_seconds=after['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in after['timing']['samples']),mean_actual_unique_key_density=statistics.mean(l['actual_unique_density'] for l in layers),speedup_vs_dense=None,speedup_unavailable='ETH3D quality only; no paired warmed Dense ETH3D timing'))
 macro={method:{k:statistics.mean(v[k] for v in values) for k in FIELDS} for method,values in grouped.items()};delta={k:macro[name][k]-macro['A_camera_dense'][k] for k in FIELDS}
 assert len(audits)==11 and sum(r['frames'] for r in runtime)==377
 quality=dict(complete=True,scenes=11,frames=377,macro=macro,delta_vs_A=delta,per_scene=rows,audits=audits,runtime=runtime,units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),speedup_vs_dense=None,speedup_unavailable='No warmed Dense ETH3D timing;7Scenes speed reported separately')
 save(OUT/'quality_summary.json',quality)
 with (OUT/'per_scene.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['method','scene','frames']+FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# History Loss A＋全特殊token双向保护','',
 '以冻结方案A为基础，唯一改动dense_special_queries从1到5：每帧camera及4个register作为Query均对全部Key计算；它们作为Key始终常驻。A的Loss、H∪{Rep} Support、Cost、权重、M5c、Value Clustering、Query Group、逐层预算及Attention kernel完全不变。','',
 '仍复用先稀疏observer、再用全量SDPA覆盖特殊Query输出的实现，历史流仍来自稀疏observer；未物理剔除稀疏路径的特殊Query重复计算，其代价完整计入正常耗时。','',
 'ETH3D既有协议11/11场景、377帧，新方案全部本批重跑；A对照质量复用上一批完整结果，不重跑未改变的控制精度。上限128帧uniform seed0；terrace/meadow依原协议排除，无新增过滤。相同精确帧表、权重SHA、预处理、评测协议与冻结源码全部审计通过。','',
 'AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2非平方距离；宏平均为11场景等权平均。','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for method,v in macro.items():lines.append('| '+method+' | '+cells(v)+' |')
 lines+=['',f"相对A：AUC@3 {delta['auc03']*100:+.4f}pp，AUC@30 {delta['auc30']*100:+.4f}pp，F1 {delta['f1']*100:+.4f}pp，CD {delta['cd']:+.6f}m。",'',
 '## 逐场景结果','', '| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['method']} | {r['scene']} | {r['frames']} | "+cells(r)+' |')
 protocol=read(BASE/'runs'/m['scenes'][0]/'result.json')['quality_protocol']
 lines+=['','## 运行协议','',f"pose+recon_unposed均评测；F1 threshold={protocol['f1_threshold_m']}m，downsample/voxel={protocol['downsample_m']}m，TSDF trunc={protocol['sdf_trunc_m']}m，mesh={protocol['mesh_sample_points']}点；原unposed对齐、掩码和深度处理不变。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；cluster fraction0.12，Query group256。未使用7Scenes真值调参。",'',
 'ETH3D以下为0暖机、1次完整冷前向，包含JIT/缓存初始化；不是暖机速度比较。相对Dense速度为N/A，原因是只在7Scenes测量速度。','', '| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |','|---|---:|---:|---:|---:|']
 for r in runtime:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['mean_actual_unique_key_density']:.6f} |")
 speed_dir=OUT/'speed_heads128';speed=None
 if (speed_dir/'execution.json').exists() and (OUT/'speed_gpu.json').exists():
  mon=read(OUT/'speed_gpu.json');assert mon['gpu']==6 and mon['returncode']==0 and not mon['same_gpu_overlap_detected'];assert read(speed_dir/'execution.json')['complete']
  runs={method:read(speed_dir/method/'timing_result.json') for method in ['A','A_all_specials','dense','saf3r']};dense=runs['dense'];speed=[]
  assert runs['saf3r']['saf3r']['topk']==4 and runs['saf3r']['saf3r']['lazy_dino_topk']
  for method,d in runs.items():
   assert d['input']==dense['input'] and d['hardware']==dense['hardware'];assert d['timing']['warmup']==2 and d['timing']['repeats']==5 and not d['timing']['capture']
   assert all(d['source_sha256'].get(k)==v for k,v in m['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
   sec=d['timing']['median_seconds'];s=dense['timing']['median_seconds']/sec;d['quality'].update(speedup_vs_vggt=s,speedup_unavailable=None)
   save(speed_dir/method/'timing_result.json',d);save(speed_dir/method/'quality.json',d['quality']);(speed_dir/method/'quality.md').write_text(quality_markdown(d['quality']))
   speed.append(dict(method=method,seconds=sec,speedup_vs_dense=s,time_over_A=sec/runs['A']['timing']['median_seconds'],time_over_saf3r=sec/runs['saf3r']['timing']['median_seconds'],peak_allocated_gib=max(v['peak_allocated_gib'] for v in d['timing']['samples']),samples=d['timing']['samples']))
  lines+=['','## 同批正常完整前向速度','',
   '7Scenes heads128，非全7场景平均；GPU6 RTX4090，全部质量任务结束后2暖机+5重复，中位数。包含每次初始化/reset、clustering、loss/selection/packing、history、重排、全特殊Query补算及SAF3R Top-K reset；不含模型/图像磁盘加载。','',
   '| Method | Median s | Dense speedup | Time / A | Time / SAF3R | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
  for r in speed:lines.append(f"| {r['method']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.4f}x | {r['time_over_A']:.4f} | {r['time_over_saf3r']:.4f} | {r['peak_allocated_gib']:.4f} |")
  lines+=['','A对照、Dense、原SAF3R只重跑计时，沿用用户此前明确授权。计时组六项几何质量为N/A，原因是仅做RGB计时。SAF3R为Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json；非论文DINO Top-K或Top-128，未混用论文/H100数据。','',
   '## 独立Mask分阶段计时','', '| Method | Mask pipeline ms | Prior/selection/budget/pack/weights ms |','|---|---:|---:|']
  profiles=[]
  for method in ['A','A_all_specials']:
   d=read(OUT/'profiles_quiet'/f'{method}.json');assert all(s['graph_creations']==0 and s['graph_evictions']==0 for s in d['samples'])
   row=dict(method=method,mask_pipeline_gpu_ms=statistics.median(s['mask_pipeline_gpu_ms'] for s in d['samples']),selection_gpu_ms=statistics.median(s['selection_prior_budget_gpu_ms'] for s in d['samples']),samples=d['samples']);profiles.append(row)
   lines.append(f"| {method} | {row['mask_pipeline_gpu_ms']:.4f} | {row['selection_gpu_ms']:.4f} |")
  lines+=['','在正常计时后，同模型/输入上额外两次CUDA事件profile。Mask累计仅覆盖已标注的Value更新/聚类、共享统计、coarse估计、prior/gate/loss/solver/pack/weights，不包含未独立包裹的reset/layout初始化、QKV投影、Attention及history执行；这些全部计入正常完整前向。事件可包含主机提交间隙；额外profile秒数不代替正常速度。']
  save(OUT/'speed_summary.json',dict(speed=speed,profiles=profiles))
 else:lines+=['','正常速度正在等待全部CPU质量评分结束后测量；目前为N/A，不能用ETH3D冷前向或历史噪声时间替代。']
 lines+=['','该实现是否改善精度应同时检查AUC、F1与CD，不能仅看特殊Query已全量。单次完整质量评测存在已知数值波动，不宣称统计显著性。本次未自动修改其他方案或历史冻结配置。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');save(OUT/'summary.json',dict(quality=quality,speed=speed,complete=speed is not None,adopted=False))
 print('A_ALL_SPECIALS_REPORT',json.dumps(dict(macro=macro,delta=delta,speed=speed)),flush=True)
if __name__=='__main__':report()
