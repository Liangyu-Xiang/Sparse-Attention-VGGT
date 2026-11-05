"""Audit and report a frozen full ETH3D evaluation of the partial-Key candidate."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/eth3d_partial_states_full_20261105'
BASE=ROOT/'outputs/eth3d_layout_fixed_full_20261105'
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def report():
 from scripts.run_eth3d_layout_fix_evaluation import verify
 manifest=read(OUT/'frozen_manifest.json');variant='states_only_fast';plan=dict(manifest['configs'][variant],source_sha256=manifest['source_sha256']);verify(plan)
 assert len(manifest['scenes'])==11
 candidate=OUT/'quality'/variant;config=read(Path(plan['config']))
 assert config==read(ROOT/'outputs/operator_repair_phase2_20261105/configs/states_only_fast.json')
 reference_sources=read(ROOT/'outputs/operator_repair_phase2_20261105/frozen_manifest.json')['source_sha256']
 assert manifest['source_sha256']==reference_sources,'Model/evaluation must remain identical to final calibration candidate'
 rows=[];audits=[];runtimes=[];grouped={'layout_fixed':[],'partial_states':[]}
 queue=read(OUT/'queue.json');execution=read(OUT/'execution.json');assert not execution['failures'] and not queue['pending']
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
  assert all(checks.values()),(scene,checks)
  audits.append(dict(scene=scene,checks=checks))
  for name,d in [('layout_fixed',before),('partial_states',after)]:
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
   grouped[name].append(v);rows.append(dict(scene=scene,method=name,frames=d['frames'],**v))
  layers=read(candidate/'runs'/scene/'layers.json');assert len(layers)==24
  assert max(l.get('native_cost_count_error_mean',0) for l in layers)==0
  runtimes.append(dict(scene=scene,gpu=gpu,frames=after['frames'],cold_full_forward_seconds=after['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in after['timing']['samples']),actual_unique_key_density_mean=statistics.mean(l['actual_unique_density'] for l in layers),alternate_rep_clusters_mean=statistics.mean(l.get('alternate_rep_clusters_mean',0) for l in layers),partial_pair_clusters_mean=statistics.mean(l.get('partial_pair_clusters_mean',0) for l in layers),warmup=0,repeats=1,speedup_vs_dense=None,speedup_unavailable='ETH3D quality only; speed requested on7Scenes, no paired warmed Dense ETH3D timing'))
 macro={name:{k:statistics.mean(v[k] for v in values) for k in FIELDS} for name,values in grouped.items()}
 delta={k:macro['partial_states'][k]-macro['layout_fixed'][k] for k in FIELDS}
 previous=read(ROOT/'outputs/operator_repair_phase2_20261105/summary.json')
 speed_rows=[r for r in previous['paired_heads128_speed'] if r['variant'] in ['states_only_fast','layout_fixed','dense','saf3r_top4_lazy_correspondence']]
 speed=dict(scope='Previously measured phase2 batch, same frozen candidate source/config;7Scenes heads128 only, GPU6 RTX4090,1 warmup+3 repeats, warmed complete-forward median. This ETH3D batch does not rerun timings or supply an all7Scenes timing macro.',source='outputs/operator_repair_phase2_20261105/summary.json',rows=speed_rows)
 total_frames=sum(r['frames'] for r in runtimes);assert total_frames==377
 summary=dict(dataset='eth3d',complete=True,scenes=11,total_frames=total_frames,max_frames=128,sampling='uniform seed0; all frames where fewer than128 eligible images',scope='Existing filtered ETH3D11-scene benchmark; terrace/meadow excluded by unchanged repository protocol; not a claim of evaluating those two scenes',method=variant,macro=macro,delta_vs_layout_fixed=delta,per_scene=rows,audits=audits,runtime=runtimes,previous_7scenes_speed_reference=speed,units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),speedup_vs_dense=None,speedup_unavailable='No paired warmed Dense ETH3D timing; prior7Scenes speed stored separately',adopted=False,adoption_note='Full quality experiment only; previously observed4.22% slower than layout-fixed candidate remains a speed regression')
 save(OUT/'summary.json',summary)
 with (OUT/'per_scene.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['scene','method','frames']+FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=['# 扩展状态＋缓存优化：完整 ETH3D 评测','',
 '11/11 个既有评测场景，共377帧；所有候选场景均在本批重新推理和评测，包括此前的三个校准场景。沿用既有ETH3D过滤协议：terrace/meadow按原设置排除，不新增过滤。','',
 '固定此前 states_only_fast 的代码、配置与逐层预算，不再调参。新增 Alternate/Pair 是在整体预算下供各簇选择的可选状态，不要求每簇保留一个Key。沿用原状态损失；本次不启用 normalized_moment 或概率质量权重校准。','',
 '与当前布局修复版对比，复用其完整历史质量结果，不重跑未变化的对照。全部输入帧表、checkpoint SHA、预处理、Pose协议、unposed融合/对齐、阈值以及评测源码均通过审计。','',
 'AUC/F1为百分数；ACC/Comp/CD为米；CD=(ACC+Comp)/2，使用非平方距离；宏平均为11场景等权平均。','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for name,values in macro.items():lines.append('| '+name+' | '+cells(values)+' |')
 lines+=['',f"相对布局修复版：AUC@3 {delta['auc03']*100:+.4f}个百分点，AUC@30 {delta['auc30']*100:+.4f}个百分点，F1 {delta['f1']*100:+.4f}个百分点；CD {delta['cd']:+.6f}m。",'',
 '## 逐场景六指标','',
 '| Scene | Method | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['scene']} | {r['method']} | {r['frames']} | "+cells(r)+' |')
 protocol=read(candidate/'runs'/manifest['scenes'][0]/'result.json')['quality_protocol']
 rates=config['layer_retentions']
 lines+=['','## 协议、预算与运行记录','',f"重建阈值{protocol['f1_threshold_m']}m，下采样/voxel {protocol['downsample_m']}m，sdf_trunc {protocol['sdf_trunc_m']}m，mesh采样{protocol['mesh_sample_points']}点。沿用仓库unposed Umeyama/RANSAC scale+pose alignment，random_state42、既有mask与深度处理。",'',f"逐层保留率上限为{rates}，均值{statistics.mean(rates):.4f}；fine cluster fraction={config['cluster_fraction']}。这是预算上限，实际密度另存layers.json和下表。该方法不使用SAF3R Top-K/profile。",'',
 '仅使用GPU4/5/6；动态场景队列与CPU重建评分并行。ETH3D每场景0暖机、1次完整前向；以下为冷前向记录，包含JIT/缓存建立，不用于速度比较。ETH3D相对Dense暖机加速比全部为N/A，原因是没有该数据集的成对暖机计时。','',
 '| Scene | GPU | Cold forward s | Peak allocated GiB | Actual unique Key density |','|---|---:|---:|---:|---:|']
 for r in runtimes:lines.append(f"| {r['scene']} | {r['gpu']} | {r['cold_full_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['actual_unique_key_density_mean']:.6f} |")
 lines+=['','## 既有7Scenes速度记录（本轮未重测）','',speed['scope'],'',
 '| Method | Warmed median s | Dense speedup | Time / layout-fixed |','|---|---:|---:|---:|']
 for r in speed_rows:lines.append(f"| {r['variant']} | {r['seconds']:.6f} | {r['speedup_vs_dense']:.4f}x | {r['relative_time_to_layout_fixed']:.4f} |")
 lines+=['','SAF3R对照为原版Top-4、lazy first-correspondence head，profile=eth3d-train-fltr-calib_cmpmse.json，并非论文DINO特征Top-K或Top-128。速度对照只有计时，六项几何质量为N/A；未混用论文公开分数或H100计时。','',
 '本候选此前比布局修复版慢约4.22%，不满足“不牺牲速度”；完整质量评测不会自动替换当前方案。重复前向存在数值波动，单次质量差异不等于统计显著增益。所有原始六项指标、输入清单、源码快照、各场景配置和GPU监控均保存。']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 save(OUT/'progress.json',dict(stage='complete',completed=manifest['scenes'],failures=[]))
 print('FULL_ETH3D_PARTIAL_STATES_COMPLETE',json.dumps(dict(macro=macro,delta=delta)),flush=True)
if __name__=='__main__':report()
