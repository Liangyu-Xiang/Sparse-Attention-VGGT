"""Four-method quality comparison with explicit historical timing provenance."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/no_mass_compensation_quality_v2_20261102'
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def normalize(x):
 if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
 if isinstance(x,list):return [normalize(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):return str(Path(x).resolve())
 return x
def main():
 m=json.loads((OUT/'manifest.json').read_text());snap=Path(m['source_root'])
 unchanged=all(hashlib.sha256((snap/n).read_bytes()).hexdigest()==h for n,h in m['source_sha256'].items())
 rows=[];audits=[];missing=[];macro={};text=['# 取消质量转移与补偿权重：单因素消融','', '仅评估取消质量转移与补偿的方案；ready_previous是上一轮已评测的head-mean/single-QK版本，Dense和SAF3R也复用已有结果。仅跳过transfer并将所有层的选中Key权重设为1；预算、状态损失、History筛选、聚类及概率估计保持不变。不按GT调参。', '', '每场景一次预热、三次完整前向取中位数。当前方法无同批Dense，Dense相对加速比N/A；历史对照时间只作标明批次的参考，不能据此计算新版配对加速比。', '']
 for ds,scenes in m['datasets'].items():
  for scene in scenes:
   refs=m['references'][ds][scene];controls={k:json.loads(Path(p).read_text()) for k,p in refs.items()}
   path=OUT/ds/scene/'no_mass_compensation/result.json'
   new=json.loads(path.read_text()) if path.exists() else None
   if new:
    monitor=json.loads((path.parent.parent/'lease_monitor.json').read_text());ls=json.loads((path.parent/'layers.json').read_text())
    checks={'source_unchanged':unchanged,'quality_complete':new['quality']['complete'],'clean_gpu':not monitor['same_gpu_overlap_detected'] and monitor['returncode']==0,'all24':len(ls)==24,'zero_second_qk':all(l['evidence_extra_qk_per_head']==0 for l in ls),'headmean_history':all(not l['history_estimator_head_aware'] for l in ls),'no_cpu_probability_cache':all(l['history_cpu_bytes']==0 for l in ls),'config_frozen':new['config']==m['config'],'compensation_disabled':new['config']['disable_population_compensation'] and all(l.get('mass_compensation_enabled') is False for l in ls),'retention_cap':all(l['actual_unique_density']<=.2+1e-7 for l in ls),'timing_protocol':new['timing']['warmup']==1 and new['timing']['repeats']==3}
    old_config={**controls['ready_previous']['config'],'disable_population_compensation':True}
    checks['single_config_change']=old_config==new['config']
    for k,d in controls.items():
     checks[k+'_input']=normalize(d['input'])==normalize(new['input'])
     checks[k+'_protocol']=d['quality_protocol']==new['quality_protocol']
     # SAF3R has an extra patch module; compare the core model and checkpoint.
     oldcfg={k:v for k,v in d['model_config'].items() if k not in ['patch_module','sparse_config_path']}
     checks[k+'_model']=normalize(oldcfg)==normalize(new['model_config'])
    audits.append(dict(dataset=ds,scene=scene,checks=checks,passed=all(checks.values())))
   else:missing.append(dict(dataset=ds,scene=scene,reason='result.json absent; inspect status/log'))
   data={**controls,**({'no_mass_compensation':new} if new else {})}
   for method,d in data.items():
    t=d['timing'];samples=t['samples'];current=method=='no_mass_compensation'
    rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],quality_complete=d['quality']['complete'],**d['quality']['values'],median_seconds=t['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in samples),peak_reserved_gib=max(s['peak_reserved_gib'] for s in samples),quality_reused=not current,timing_batch=OUT.name if current else str(Path(refs[method]).relative_to(ROOT)).split('/')[1],speedup_vs_dense=1. if method=='dense' else None,speedup_reason='Dense is the reference' if method=='dense' else 'No contemporaneous paired Dense in this evaluation; historical controls reused',source=str(path) if current else refs[method]))
  macro[ds]={}
  for method in ['dense','saf3r','ready_previous','no_mass_compensation']:
   rr=[r for r in rows if r['dataset']==ds and r['method']==method]
   if len(rr)!=len(scenes) or not all(r['quality_complete'] for r in rr):continue
   macro[ds][method]={k:statistics.mean(r[k] for r in rr) for k in FIELDS}
   macro[ds][method].update(mean_scene_median_seconds=statistics.mean(r['median_seconds'] for r in rr),peak_allocated_gib=max(r['peak_allocated_gib'] for r in rr),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rr),scenes=len(rr),speedup_vs_dense=1. if method=='dense' else None)
  text += [f'## {ds}：{len(scenes)}场景等权宏平均','', '|方法|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|平均场景前向 s|峰值allocated GiB|Dense加速比|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
  for method,d in macro[ds].items():
   v=[f'{d[k]*(100 if k in ["auc03","auc30","f1"] else 1):.5f}' for k in FIELDS]
   text.append('|'+ '|'.join([method]+v+[f'{d["mean_scene_median_seconds"]:.4f}',f'{d["peak_allocated_gib"]:.3f}','1.00×' if method=='dense' else 'N/A'])+'|')
  text+=['', 'Dense和SAF3R时间来自历史批次；当前方法时间来自本批，不能视作同批速度比较。', '']
  if 'no_mass_compensation' in macro[ds]:
   cur=macro[ds]['no_mass_compensation'];text+=['|新版相对对照|AUC@3 变化 pp|AUC@30 变化 pp|ACC 相对变化 %|Comp 相对变化 %|CD 相对变化 %|F1 变化 pp|','|---|---:|---:|---:|---:|---:|---:|']
   for ref in ['dense','saf3r','ready_previous']:
    d=macro[ds][ref];values=[(cur[k]-d[k])*100 if k in ['auc03','auc30','f1'] else (cur[k]/d[k]-1)*100 for k in FIELDS]
    text.append('|'+ref+'|'+'|'.join(f'{v:+.4f}' for v in values)+'|')
   text+=['','AUC/F1越高越好；距离越低越好。pp为百分点，距离为相对百分比变化。','']
 complete=not missing and len(audits)==18 and all(a['passed'] for a in audits)
 summary=dict(complete=complete,macro=macro,audits=audits,missing=missing,source_unchanged=unchanged,controls_executed=0,new_quality_runs=len(audits),no_quality_retuning=True)
 save(OUT/'summary.json',summary)
 if rows:
  with (OUT/'per_scene.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  mr=[dict(dataset=ds,method=k,**d) for ds,dd in macro.items() for k,d in dd.items()]
  with (OUT/'macro.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(mr[0]));w.writeheader();w.writerows(mr)
 text+=['SAF3R为本地lazy first-correspondence Top-4；不是论文DINO Top-K。具体profile、输入帧、重建阈值与降采样见原始result.json。7Scenes：F1阈值0.05 m、降采样0.0078125 m；ETH3D：0.25 m、0.0390625 m。CD=(ACC+Comp)/2，未平方距离。', '', f'完整性审计通过：{complete}；源码未变：{unchanged}。逐场景指标、时间批次和显存见per_scene.csv；原始预测评估指标见各场景result.json。']
 (OUT/'report.md').write_text('\n'.join(text)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
