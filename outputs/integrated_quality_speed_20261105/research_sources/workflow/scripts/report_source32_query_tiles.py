"""Matched full-forward, attention-stage, memory and six-metric tile comparison."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source32_query_tiles_20261105'
plan=json.loads((OUT/'plan.json').read_text());variants=[v['name'] for v in plan['variants']];scenes=plan['scenes'];fields=('auc03','auc30','acc','comp','cd','f1')
rows=[];missing=[];labels={n:n.replace('bq','BQ').replace('_resource','（资源调整）') for n in variants}
for scene in scenes:
 reference_path=OUT/scene/'bq32/result.json';reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
 for variant in variants:
  p=OUT/scene/variant/'result.json';mp=OUT/f'{scene}_{variant}.monitor.json'
  if not p.exists() or not mp.exists():missing.append(f'{scene}/{variant}');continue
  d=json.loads(p.read_text());monitor=json.loads(mp.read_text());assert monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
  assert d['quality']['complete'],(scene,variant,d['quality'])
  if reference:
   for key in ('input','hardware','quality_protocol','model_config'):assert d[key]==reference[key],(scene,variant,key)
   refmon=json.loads((OUT/f'{scene}_bq32.monitor.json').read_text());assert monitor['gpu_uuid']==refmon['gpu_uuid'],(scene,variant,'physical GPU')
   common_source=set(d['source_sha256'])&set(reference['source_sha256'])
   assert all(d['source_sha256'][k]==reference['source_sha256'][k] for k in common_source if k.startswith('saf3r/') or k=='scripts/evaluate_zdrop_history_states.py'),(scene,variant,'source drift')
   allowed={'attention_block_queries','attention_warps','attention_stages'}
   assert all(d['config'][k]==reference['config'][k] for k in reference['config'] if k not in allowed)
  cfg=d['config'];assert cfg['query_group_size']==256 and cfg['source_width']==32 and cfg['clustering_mode']=='source_local_unanchored' and cfg['source_layout']=='sequential' and cfg['retention']==.2
  assert cfg['history_split_export']==False
  ps=d['attention_profiles'];assert len(ps)==3;idx=sorted(range(3),key=lambda i:ps[i]['wall_seconds'])[1];profile=ps[idx]
  direct=profile['stages_seconds']['sampled_direct_av'];export=profile['stages_seconds']['sampled_probability_export'];total_us=round(profile['wall_seconds']*1e6);direct_us=round(direct*1e6);export_us=round(export*1e6);remainder_us=total_us-direct_us-export_us
  assert remainder_us>0 and direct_us+export_us+remainder_us==total_us
  q=d['quality']['values'];assert abs(q['cd']-(q['acc']+q['comp'])/2)<1e-10
  row=dict(scene=scene,variant=variant,block_queries=cfg['attention_block_queries'],num_warps=cfg['attention_warps'],num_stages=2 if cfg['attention_block_queries']==32 else cfg['attention_stages'],
   full_forward_seconds=d['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),
   relative_to_bq32=reference['timing']['median_seconds']/d['timing']['median_seconds'] if reference else None,
   attention_profile_index=idx,attention_main_seconds=direct_us/1e6,attention_export_seconds=export_us/1e6,profile_other_seconds=remainder_us/1e6,profile_full_wall_seconds=total_us/1e6,
   logical_qk_per_head=profile['logical_qk_per_head'],mean_key_density=profile['mean_key_density'],resources=profile['compiler_resources'],
   normal_graph_creations=[s['graph_counters']['graph_cache_creations'] for s in d['timing']['samples']],
   gpu_uuid=monitor['gpu_uuid'],gpu_index=monitor['gpu'],dense_speedup=None,dense_speedup_reason='Unchanged Dense not rerun; no current paired full-forward timing',quality_complete=True,**q)
  rows.append(row)
means=[]
for variant in variants:
 rs=[r for r in rows if r['variant']==variant]
 if len(rs)!=3:continue
 m=dict(variant=variant,block_queries=rs[0]['block_queries'],num_warps=rs[0]['num_warps'],num_stages=rs[0]['num_stages'],peak_allocated_gib=max(r['peak_allocated_gib'] for r in rs))
 for k in ('full_forward_seconds','attention_main_seconds','attention_export_seconds','profile_other_seconds','profile_full_wall_seconds','logical_qk_per_head','mean_key_density',*fields):m[k]=statistics.mean(r[k] for r in rs)
 base=next((r for r in means if r['variant']=='bq32'),None);m['relative_to_bq32']=base['full_forward_seconds']/m['full_forward_seconds'] if base else 1.
 main_us=round(m['attention_main_seconds']*1e6);export_us=round(m['attention_export_seconds']*1e6);wall_us=round(m['profile_full_wall_seconds']*1e6)
 m.update(profile_attention_main_us=main_us,profile_attention_export_us=export_us,profile_other_us=wall_us-main_us-export_us,profile_wall_us=wall_us)
 means.append(m)
complete=not missing;winner=min(means,key=lambda r:r['full_forward_seconds'])['variant'] if complete else None
summary=dict(complete=complete,valid_quality_runs=len(rows),planned_quality_runs=len(variants)*3,missing=missing,per_scene=rows,three_scene_means=means,
 selected_by_full_forward_time=winner,selection_uses_test_truth=False,quality_caution='One final normal output evaluated per run. Float atomic reductions and discrete routing are non-deterministic; differences are not proof of stable accuracy gain/loss.',
 dense_speedup=None,dense_speedup_reason='Unchanged Dense not rerun under AGENTS.md; no paired timing')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for filename,data in (('per_scene.csv',rows),('three_scene_means.csv',means)):
 with (OUT/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['variant']);w.writeheader();w.writerows(data)
lines=['# Query group256：attention计算块32/64/128实验','',f'有效完整质量评估 {len(rows)}/{len(variants)*3} 次。只测 heads/chess/stairs，各128帧；三序列等权平均，不是七场景宏平均。','',
 '固定连续编号source32无锚点聚类、2次更新、Query group256、Key保留上限20%、历史采样和Graph策略。BQ32/64/128主对照统一4个warp、2级流水线、相同历史导出。资源调整组另外改变warp/流水线，候选由固定输入算子测速选出，没有使用测试集真值。','',
 '同一序列全部配置使用同一物理RTX4090；纳入结果的严格监控无其他GPU任务重叠。相同checkpoint、逐帧列表、预处理、pose/recon_unposed协议。正常前向1次预热+3次取中位数，包含初始化、聚类、选边、history、attention和输出头。','',
 '## 三序列等权均值','', '| 配置 | warp/stage | 正常完整前向s | 相对BQ32 | AV＋inline证据s | 独立导出s | 峰值allocated GiB |','|---|---|---|---|---|---|---|']
for r in means:lines.append(f"| {labels[r['variant']]} | {r['num_warps']}/{r['num_stages']} | {r['full_forward_seconds']:.6f} | {r['relative_to_bq32']:.3f}x | {r['attention_main_seconds']:.6f} | {r['attention_export_seconds']:.6f} | {r['peak_allocated_gib']:.3f} |")
lines+=['','AV与独立导出来自另外3次轻量事件统计前向，选择整个wall时间居中的一次。不要把阶段时间拼接到正常测速总时间；下表才是同一次前向的闭合账本。','', '| 配置 | AV＋inline证据s | 独立导出s | 其余完整前向s | 完整wall＝前三项之和s |','|---|---|---|---|---|']
for r in means:lines.append('| '+labels[r['variant']]+' | '+' | '.join(f"{r[k]/1e6:.6f}" for k in ('profile_attention_main_us','profile_attention_export_us','profile_other_us','profile_wall_us'))+' |')
lines+=['','## 编译资源','', '| 配置 | 最大n_regs | 最大n_spills | 最大shared bytes |','|---|---|---|---|']
for variant in variants:
 rs=[r for r in rows if r['variant']==variant]
 if rs:lines.append('| '+labels[variant]+' | '+' | '.join(str(max(r['resources'][k] for r in rs)) for k in ('registers','spills','shared_bytes'))+' |')
lines+=['','峰值显存为PyTorch allocated，不等于整个GPU进程占用，也不一定完整反映驱动管理的spill存储。','', '## 六项质量','', 'AUC/F1为百分数，距离米；CD=(ACC+Comp)/2非平方距离。F1阈值0.05m，下采样0.0078125m。每次使用末轮正常前向输出评估pose和recon_unposed；不是三次独立质量均值，浮点原子归约会影响离散选边，不能把单次差值认作稳定提升。','', '| 配置 | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | vs Dense |','|---|---|---|---|---|---|---|---|']
for r in means:lines.append('| '+labels[r['variant']]+' | '+' | '.join(f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in fields)+' | N/A |')
lines+=['','Dense速度比均N/A：未重跑未改动Dense，没有本批配对完整前向；SAF3R也未重跑。既有SAF3R参照为lazy首个对应头Top-4，不是论文DINO特征Top-K。','', '## 逐序列完整结果','']
for scene in scenes:
 lines += [f'### {scene}','', '| 配置 | 正常完整前向s | AV＋inline证据s | 峰值GiB | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | vs Dense |','|---|---|---|---|---|---|---|---|---|---|---|']
 for r in rows:
  if r['scene']==scene:lines.append('| '+labels[r['variant']]+' | '+f"{r['full_forward_seconds']:.6f} | {r['attention_main_seconds']:.6f} | {r['peak_allocated_gib']:.3f} | "+' | '.join(f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in fields)+' | N/A |')
lines+=['','## 资源与固定输入诊断','', 'operators/result.json 保存固定4warps/2stages的同输入同mask检查、尾部Query及Key padding检查和实际128帧K/V几何的算子测速。resource_probe/result.json 保存预先确定的warp4/8、stage1/2资源网格，单列为算子诊断，六项质量N/A且原因明确。主循环同时导出history，不能当作纯AV。','']
if complete:lines+=['按正常完整前向均值选择的配置：'+winner+'。选择依据仅为速度及算子正确性/运行完整性，没有根据7Scenes测试真值调参。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=complete,valid_runs=len(rows),planned=len(variants)*3,winner=winner)))
