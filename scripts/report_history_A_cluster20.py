"""Audit A+all-specials quality and matched speed without rerunning control quality."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105';BASE=ROOT/'outputs/rep_history_A_all_specials_20261105/quality/A_all_specials'
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def report():
 from scripts.run_eth3d_layout_fix_evaluation import verify
 from scripts.quality_metrics import quality_markdown
 m=read(OUT/'frozen_manifest.json');name='A_all_specials_cluster20';plan=dict(m['configs'][name],source_sha256=m['source_sha256']);verify(plan)
 cfg=read(Path(plan['config']));base_cfg=read(ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_dense.json')
 assert cfg==base_cfg|dict(cluster_fraction=.20)
 assert m['source_sha256']==read(ROOT/'outputs/rep_history_loss_20261105/full/frozen_manifest.json')['source_sha256'],'Only config changes; all native model/evaluation code must remain frozen'
 q=read(OUT/'queue.json');assert not q['pending'] and not read(OUT/'execution.json')['failures']
 grouped={'cluster12':[],'A_all_specials_cluster20':[]};rows=[];audits=[];runtime=[]
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
  expected=int(cfg['cluster_fraction']*(after['frames']*(after['input']['shape'][-2]//14)*(after['input']['shape'][-1]//14)))
  checks['cluster_count20']=all(l['fine_clusters']==expected for l in layers)
  checks['budget_count']=all(l.get('native_cost_count_error_mean',0)==0 for l in layers)
  assert all(checks.values()),(scene,checks);audits.append(dict(scene=scene,checks=checks))
  for method,d in [('cluster12',before),('A_all_specials_cluster20',after)]:
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
   grouped[method].append(v);rows.append(dict(method=method,scene=scene,frames=d['frames'],**v))
  runtime.append(dict(scene=scene,gpu=monitor['gpu'],frames=after['frames'],cold_full_forward_seconds=after['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in after['timing']['samples']),mean_actual_unique_key_density=statistics.mean(l['actual_unique_density'] for l in layers),speedup_vs_dense=None,speedup_unavailable='ETH3D quality only; no paired warmed Dense ETH3D timing'))
 macro={method:{k:statistics.mean(v[k] for v in values) for k in FIELDS} for method,values in grouped.items()};delta={k:macro[name][k]-macro['cluster12'][k] for k in FIELDS}
 assert len(audits)==11 and sum(r['frames'] for r in runtime)==377
 quality=dict(complete=True,scenes=11,frames=377,macro=macro,delta_vs_cluster12=delta,per_scene=rows,audits=audits,runtime=runtime,units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),speedup_vs_dense=None,speedup_unavailable='No warmed Dense ETH3D timing;7Scenes speed reported separately')
 save(OUT/'quality_summary.json',quality)
 with (OUT/'per_scene.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['method','scene','frames']+FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# A＋全特殊token：cluster数20%消融','',
 '以A＋全特殊token修正为基础，唯一改动cluster_fraction从0.12到0.20，即floor(0.20×patch token数)个簇。两个版本每帧camera及4个register作为Query均对全部Key计算、作为Key始终常驻。A的Loss、H∪{Rep} Support、Cost、权重、M5c、Value Clustering算法、Query Group、逐层Key预算及Attention kernel均保持不变；簇数增加不等于提高Key保留预算。','',
 '仍复用先稀疏observer、再用全量SDPA覆盖特殊Query输出的实现，历史流仍来自稀疏observer；未物理剔除稀疏路径的特殊Query重复计算，其代价完整计入正常耗时。','',
 'ETH3D既有协议11/11场景、377帧，新方案全部本批重跑；12%簇数对照质量复用上一批完整结果，不重跑未改变的控制精度。上限128帧uniform seed0；terrace/meadow依原协议排除，无新增过滤。相同精确帧表、权重SHA、预处理、评测协议与冻结源码全部审计通过。','',
 'AUC/F1为百分数，ACC/Comp/CD为米；CD=(ACC+Comp)/2非平方距离；宏平均为11场景等权平均。','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for method,v in macro.items():lines.append('| '+method+' | '+cells(v)+' |')
 lines+=['',f"相对12%簇数：AUC@3 {delta['auc03']*100:+.4f}pp，AUC@30 {delta['auc30']*100:+.4f}pp，F1 {delta['f1']*100:+.4f}pp，CD {delta['cd']:+.6f}m。",'',
 '## 逐场景结果','', '| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['method']} | {r['scene']} | {r['frames']} | "+cells(r)+' |')
 protocol=read(BASE/'runs'/m['scenes'][0]/'result.json')['quality_protocol']
 lines+=['','## 运行协议','',f"pose+recon_unposed均评测；F1 threshold={protocol['f1_threshold_m']}m，downsample/voxel={protocol['downsample_m']}m，TSDF trunc={protocol['sdf_trunc_m']}m，mesh={protocol['mesh_sample_points']}点；原unposed对齐、掩码和深度处理不变。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；cluster fraction0.20（对照0.12），Query group256。未使用7Scenes真值调参。",'',
 'ETH3D以下为0暖机、1次完整冷前向，包含JIT/缓存初始化；不是暖机速度比较。相对Dense速度为N/A，原因是只在7Scenes测量速度。','', '| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |','|---|---:|---:|---:|---:|']
 for r in runtime:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['mean_actual_unique_key_density']:.6f} |")
 speed_dir=OUT/'speed_heads128';speed=None
 if (speed_dir/'execution.json').exists() and (OUT/'speed_gpu.json').exists():
  mon=read(OUT/'speed_gpu.json');assert mon['gpu']==6 and mon['returncode']==0 and not mon['same_gpu_overlap_detected'];assert read(speed_dir/'execution.json')['complete']
  runs={method:read(speed_dir/method/'timing_result.json') for method in ['cluster12','A_all_specials_cluster20','dense','saf3r']};dense=runs['dense'];speed=[]
  assert runs['saf3r']['saf3r']['topk']==4 and runs['saf3r']['saf3r']['lazy_dino_topk']
  for method,d in runs.items():
   assert d['input']==dense['input'] and d['hardware']==dense['hardware'];assert d['timing']['warmup']==2 and d['timing']['repeats']==5 and not d['timing']['capture']
   assert all(d['source_sha256'].get(k)==v for k,v in m['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
   sec=d['timing']['median_seconds'];s=dense['timing']['median_seconds']/sec;d['quality'].update(speedup_vs_vggt=s,speedup_unavailable=None)
   save(speed_dir/method/'timing_result.json',d);save(speed_dir/method/'quality.json',d['quality']);(speed_dir/method/'quality.md').write_text(quality_markdown(d['quality']))
   speed.append(dict(method=method,seconds=sec,speedup_vs_dense=s,time_over_cluster12=sec/runs['cluster12']['timing']['median_seconds'],time_over_saf3r=sec/runs['saf3r']['timing']['median_seconds'],peak_allocated_gib=max(v['peak_allocated_gib'] for v in d['timing']['samples']),samples=d['timing']['samples']))
  lines+=['','## 同批正常完整前向速度','',
   '7Scenes heads128，非全7场景平均；GPU6 RTX4090，全部质量任务结束后2暖机+5重复，中位数。包含每次初始化/reset、clustering、loss/selection/packing、history、重排、全特殊Query补算及SAF3R Top-K reset；不含模型/图像磁盘加载。','',
   '| Method | Median s | Dense speedup | Time / cluster12 | Time / SAF3R | Peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
  for r in speed:lines.append(f"| {r['method']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.4f}x | {r['time_over_cluster12']:.4f} | {r['time_over_saf3r']:.4f} | {r['peak_allocated_gib']:.4f} |")
  lines+=['','12%簇数对照、Dense、原SAF3R只重跑计时，沿用用户此前明确授权。计时组六项几何质量为N/A，原因是仅做RGB计时。SAF3R为Top-4 lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json；非论文DINO Top-K或Top-128，未混用论文/H100数据。','',
   '## 独立Mask分阶段计时','', '| Method | Mask pipeline ms | Prior/selection/budget/pack/weights ms |','|---|---:|---:|']
  profiles=[]
  for method in ['cluster12','A_all_specials_cluster20']:
   d=read(OUT/'profiles_quiet'/f'{method}.json');assert all(s['graph_creations']==0 and s['graph_evictions']==0 for s in d['samples'])
   row=dict(method=method,mask_pipeline_gpu_ms=statistics.median(s['mask_pipeline_gpu_ms'] for s in d['samples']),selection_gpu_ms=statistics.median(s['selection_prior_budget_gpu_ms'] for s in d['samples']),samples=d['samples']);profiles.append(row)
   lines.append(f"| {method} | {row['mask_pipeline_gpu_ms']:.4f} | {row['selection_gpu_ms']:.4f} |")
  lines+=['','在正常计时后，同模型/输入上额外两次CUDA事件profile。Mask累计仅覆盖已标注的Value更新/聚类、共享统计、coarse估计、prior/gate/loss/solver/pack/weights，不包含未独立包裹的reset/layout初始化、QKV投影、Attention及history执行；这些全部计入正常完整前向。事件可包含主机提交间隙；额外profile秒数不代替正常速度。']
  pressure=[]
  for method in ['cluster12','A_all_specials_cluster20']:
   layers=read(speed_dir/method/'layers.json')
   shallow_deep=[1,2,3,4,5,18,19,20,21,22,23]
   mid=list(range(6,18))
   pressure.append(dict(method=method,scope='7Scenes heads128 last normal timed forward; equal-layer means, L0 seed excluded',fine_clusters=layers[0]['fine_clusters'],shallow_deep_drop_fraction=statistics.mean(layers[i]['state_0_clusters_mean']/layers[i]['fine_clusters'] for i in shallow_deep),middle_drop_fraction=statistics.mean(layers[i]['state_0_clusters_mean']/layers[i]['fine_clusters'] for i in mid),actual_unique_key_density_mean=statistics.mean(l['actual_unique_density'] for l in layers)))
  lines+=['','## 固定预算下的簇数压力','',
   '以下来自heads128最后一次正常计时的逐层记录，不是ETH3D宏平均。浅/深层为L1–L5与L18–L23，排除不使用四状态损失的L0 seed；中间层为L6–L17。Drop按簇数计数，不等于被丢弃真实概率质量或token数。','',
   '| Method | Fine clusters | Shallow/deep Drop % | Middle Drop % | Actual unique density % |','|---|---:|---:|---:|---:|']
  for r in pressure:lines.append(f"| {r['method']} | {r['fine_clusters']} | {r['shallow_deep_drop_fraction']*100:.4f} | {r['middle_drop_fraction']*100:.4f} | {r['actual_unique_key_density_mean']*100:.4f} |")
  lines+=['',
   '簇数增大而Key预算保持不变，使低保留预算层的簇覆盖压力增大；更高的Drop比例是一个可能影响位姿的机制，但这些统计本身不能证明ETH3D AUC下降的全部原因。Mask阶段随簇数增加而变慢，包含更多簇统计与四状态预算处理。20%版本F1更高、AUC更低，表现为重建和位姿指标取舍，不是全面质量提升。']
  candidate=next(r for r in speed if r['method']=='A_all_specials_cluster20')
  gates=dict(no_slowdown_vs_cluster12=candidate['time_over_cluster12']<=1.,within_saf3r_10percent=candidate['time_over_saf3r']<=1.1)
  lines+=['',f"20%版本比12%版慢{(candidate['time_over_cluster12']-1)*100:.4f}%，比原SAF3R慢{(candidate['time_over_saf3r']-1)*100:.4f}%；本批中位数下满足不牺牲速度：{gates['no_slowdown_vs_cluster12']}，满足SAF3R+10%目标：{gates['within_saf3r_10percent']}。本次结果不支持直接将20%作为当前改进版。"]
  save(OUT/'speed_summary.json',dict(speed=speed,profiles=profiles,cluster_pressure=pressure,gates=gates))
 else:lines+=['','正常速度正在等待全部CPU质量评分结束后测量；目前为N/A，不能用ETH3D冷前向或历史噪声时间替代。']
 lines+=['','该实现是否改善精度应同时检查AUC、F1与CD，不能仅看特殊Query已全量。单次完整质量评测存在已知数值波动，不宣称统计显著性。本次未自动修改其他方案或历史冻结配置。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');save(OUT/'summary.json',dict(quality=quality,speed=speed,complete=speed is not None,adopted=False))
 print('CLUSTER20_REPORT',json.dumps(dict(macro=macro,delta=delta,speed=speed)),flush=True)
if __name__=='__main__':report()
