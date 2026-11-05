"""Frozen single-factor quality tables and input/protocol audits for Rep-History A/B."""
import csv,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def report_quality(out):
 from scripts.run_rep_history_campaign import plan,BASE
 from scripts.run_eth3d_layout_fix_evaluation import verify
 m=read(out/'frozen_manifest.json');q=read(out/'queue.json');assert not read(out/'execution.json')['failures'] and not q['pending']
 rows=[];audits=[];runtime=[];macro={}
 for method in m['variants']:
  verify(plan(m,method));group=[]
  for scene in m['scenes']:
   folder=out/'quality'/method/'runs'/scene;d=read(folder/'result.json');base=read(BASE/'runs'/scene/'result.json')
   assert d['quality']['complete']
   checks={k:d['input'][k]==base['input'][k] for k in ['image_paths','frame_indices','shape','checkpoint_sha256']}
   checks['quality_protocol']=d['quality_protocol']==base['quality_protocol'];checks['frozen_source']=all(d['source_sha256'].get(k)==v for k,v in m['source_sha256'].items() if k.startswith(('saf3r/','evaluation/')))
   checks['frozen_config']=all(d['config'].get(k)==v for k,v in read(Path(m['configs'][method]['config'])).items())
   claim=next(c for c in q['claims'] if c['variant']==method and c['scene']==scene);monitor=read(out/f"worker{claim['gpu']}_gpu.json")
   checks['gpu_scope_isolation']=monitor['gpu'] in [4,5,6] and not monitor['same_gpu_overlap_detected'] and monitor['returncode']==0
   checks['gpu_uuid']=d['hardware']['gpu_uuid'].removeprefix('GPU-')==monitor['gpu_uuid'].removeprefix('GPU-')
   layers=read(folder/'layers.json');checks['budget_counts']=all(l.get('native_cost_count_error_mean',0)==0 for l in layers)
   assert all(checks.values()),(method,scene,checks);audits.append(dict(method=method,scene=scene,checks=checks))
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
   group.append(v);rows.append(dict(method=method,scene=scene,frames=d['frames'],**v))
   runtime.append(dict(method=method,scene=scene,frames=d['frames'],gpu=monitor['gpu'],cold_forward_seconds=d['timing']['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),mean_actual_key_density=statistics.mean(l['actual_unique_density'] for l in layers),speedup_vs_dense=None,speedup_unavailable='ETH3D quality-only; warmed speed measured separately on7Scenes'))
  macro[method]={k:statistics.mean(v[k] for v in group) for k in FIELDS}
 reference=m['base_reference']['macro'];delta={method:{k:v[k]-macro['baseline'][k] for k in FIELDS} for method,v in macro.items()}
 summary=dict(complete=True,scenes=len(m['scenes']),macro=macro,delta_vs_fresh_baseline=delta,provided_reference_macro=reference,baseline_drift_vs_reference={k:macro['baseline'][k]-reference[k] for k in FIELDS} if len(m['scenes'])==11 else None,per_scene=rows,audits=audits,runtime=runtime,units=dict(auc03='fraction',auc30='fraction',acc='metres',comp='metres',cd='metres',f1='fraction'),speedup_vs_dense=None,speedup_unavailable='ETH3D quality only; see matched7Scenes timing report')
 save(out/'summary.json',summary)
 with (out/'per_scene.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=['method','scene','frames']+FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
 def cells(v):return ' | '.join(f"{v[k]*(100 if k in ['auc03','auc30','f1'] else 1):.6f}" for k in FIELDS)
 lines=[f"# Rep-based History Loss：ETH3D {len(m['scenes'])}场景评测",'',
 '固定布局修复后的原始四状态基线，仅验证用户给出的A/B。未引入新状态、Fine QK、每Query Group全量Value距离、M5c改动、聚类改动、预算改动或Attention kernel改动。特殊Query保留文档基线的camera-only设置，不混入上轮all-special修正。','',
 'A：max(0,L_Rep−sum_H p_h||V_h−V_rep||²)，实际support=H∪{rep}，cost去重；历史非rep权重1，rep权重=population−(support_count−1)。这是Value表示失真代理，不是softmax输出误差。','',
 'B：仅|H|=1时max(0,L_Rep+(pi_C−2p_h)||V_h−V_rep||²)；多历史Key保持原损失；support和cluster_population权重不变。','',
 '本批Baseline/A/B均重新推理与pose+recon_unposed评分。相同checkpoint、精确帧表、预处理与协议均审计通过。AUC/F1为百分数，ACC/Comp/CD为米，CD=(ACC+Comp)/2非平方距离；宏平均为场景等权平均。','',
 '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|']
 for method,v in macro.items():lines.append('| '+method+' | '+cells(v)+' |')
 if len(m['scenes'])==11:
  lines+=['','用户提供/既有历史参考（不是本批新测Baseline）：','', '| Reference | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---:|---:|---:|---:|---:|---:|','| historical layout-fixed | '+cells(reference)+' |', '', '本批对照变化以新测Baseline为准，单独保留旧基线复现漂移，不平移或缩放分数。既有ETH3D协议共11场景、377帧，terrace/meadow仍按旧协议排除。']
 else:lines+=['','这是三个校准场景的小规模验证，不是完整11场景结果。']
 lines+=['','## 逐场景六指标','', '| Method | Scene | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['method']} | {r['scene']} | {r['frames']} | "+cells(r)+' |')
 protocol=read(out/'quality/baseline/runs'/m['scenes'][0]/'result.json')['quality_protocol']
 lines+=['',f"重建阈值{protocol['f1_threshold_m']}m，下采样/voxel {protocol['downsample_m']}m，TSDF trunc {protocol['sdf_trunc_m']}m，mesh采样{protocol['mesh_sample_points']}点；沿用原unposed融合/对齐、mask和深度处理。逐层patch预算0.10×6、0.22×12、0.10×6，均值0.16；fine cluster fraction0.12，Query group256。方法不使用SAF3R Top-K/profile。",'',
 '## 独立运行记录','',
 '以下为0暖机、1次完整冷前向，包含JIT/缓存建立；不作为速度比较。所有ETH3D相对Dense暖机加速为N/A，因为用户仅要求7Scenes速度。','',
 '| Method | Scene | GPU | Cold forward s | Peak allocated GiB | Actual Key density |','|---|---|---:|---:|---:|---:|']
 for r in runtime:lines.append(f"| {r['method']} | {r['scene']} | {r['gpu']} | {r['cold_forward_seconds']:.6f} | {r['peak_allocated_gib']:.4f} | {r['mean_actual_key_density']:.6f} |")
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('QUALITY_CAMPAIGN_COMPLETE',out.name,json.dumps(macro),flush=True)
if __name__=='__main__':report_quality(Path(sys.argv[1]))
