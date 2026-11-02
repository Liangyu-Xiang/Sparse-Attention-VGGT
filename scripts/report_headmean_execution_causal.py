"""Report all six metrics and conditional effects for the fixed 2x2 design."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/headmean_execution_causal_20261102';FIELDS=['auc03','auc30','acc','comp','cd','f1'];ORDER=['perhead_two_qk','headmean_two_qk','perhead_one_qk','headmean_one_qk']
def normalize(x):
 if isinstance(x,dict):return {k:normalize(v) for k,v in x.items()}
 if isinstance(x,list):return [normalize(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):return str(Path(x).resolve())
 return x
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 m=json.loads((OUT/'manifest.json').read_text());rows=[];audits=[];missing=[];macro={}
 unchanged=all(hashlib.sha256((Path(m['sources'][v])/n).read_bytes()).hexdigest()==h for v,dd in m['source_sha256'].items() for n,h in dd.items())
 for scene in m['scenes']:
  data={v:json.loads(Path(p).read_text()) for v,p in m['references'][scene].items()}
  for variant in ['headmean_two_qk','perhead_one_qk']:
   path=OUT/'eth3d'/scene/variant/'result.json'
   if not path.exists():missing.append(dict(scene=scene,variant=variant));continue
   d=json.loads(path.read_text());data[variant]=d
   monitor=json.loads((path.parent.parent/(variant+'_monitor.json')).read_text());layers=json.loads((path.parent/'layers.json').read_text());parent=data['perhead_two_qk'];expected_config={**parent['config'],'evidence_mode':'second_qk' if variant=='headmean_two_qk' else 'fused_grid'}
   checks=dict(source_unchanged=unchanged,quality_complete=d['quality']['complete'],clean_gpu=monitor['returncode']==0 and not monitor['same_gpu_overlap_detected'],same_inputs=normalize(parent['input'])==normalize(d['input']),same_protocol=parent['quality_protocol']==d['quality_protocol'],same_model=normalize(parent['model_config'])==normalize(d['model_config']),config=expected_config==d['config'],all24=len(layers)==24,budget=all(x['actual_unique_density']<=.2+1e-7 for x in layers),estimator=all(x['history_estimator_head_aware']==(variant=='perhead_one_qk') for x in layers),execution=all((x['evidence_extra_qk_per_head']==0)==(variant=='perhead_one_qk') for x in layers))
   audits.append(dict(scene=scene,variant=variant,checks=checks,passed=all(checks.values())))
  for variant,d in data.items():
   t=d['timing'];rows.append(dict(scene=scene,variant=variant,frames=d['frames'],quality_complete=d['quality']['complete'],**d['quality']['values'],seconds=t['median_seconds'],peak_allocated_gib=max(x['peak_allocated_gib'] for x in t['samples']),quality_reused=variant in m['references'][scene],dense_speedup=None,timing_batch='historical_reused' if variant in m['references'][scene] else OUT.name))
 for variant in ORDER:
  rr=[r for r in rows if r['variant']==variant]
  if len(rr)==11 and all(r['quality_complete'] for r in rr):macro[variant]={**{k:statistics.mean(r[k] for r in rr) for k in FIELDS},'seconds':statistics.mean(r['seconds'] for r in rr),'peak_allocated_gib':max(r['peak_allocated_gib'] for r in rr)}
 complete=not missing and len(audits)==22 and all(a['passed'] for a in audits)
 contrasts={}
 if len(macro)==4:
  for name,pos,neg in [('headmean_on_two_qk','headmean_two_qk','perhead_two_qk'),('single_qk_on_perhead','perhead_one_qk','perhead_two_qk'),('headmean_on_single_qk','headmean_one_qk','perhead_one_qk'),('single_qk_on_headmean','headmean_one_qk','headmean_two_qk')]:contrasts[name]={k:macro[pos][k]-macro[neg][k] for k in FIELDS}
  contrasts['interaction']={k:macro['headmean_one_qk'][k]-macro['headmean_two_qk'][k]-macro['perhead_one_qk'][k]+macro['perhead_two_qk'][k] for k in FIELDS}
 save(OUT/'summary.json',dict(complete=complete,macro=macro,contrasts=contrasts,audits=audits,missing=missing,source_unchanged=unchanged,new_quality_runs=len(audits),unchanged_controls_executed=0,no_quality_retuning=True,dense_speedup=None))
 if rows:
  with (OUT/'per_scene.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=['# ETH3D head-mean与单次QK的独立归因','', '固定2×2：估计器per-head/head-mean × 原两次QK/按grid分组的单次QK。A、D复用已有11场景完整质量；只执行B、C，各11场景，一次质量评测。不混入取消补偿。输入、模型、协议、20%预算、状态目标不变；没有按GT调参。', '', '|方案|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|平均前向 s|峰值GiB|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for v in ORDER:
  if v in macro:
   d=macro[v];text.append('|'+v+'|'+'|'.join(f'{d[k]*(100 if k in ["auc03","auc30","f1"] else 1):.5f}' for k in FIELDS)+f'|{d["seconds"]:.5f}|{d["peak_allocated_gib"]:.3f}|')
 text+=['','等权11场景宏平均；全部可用14–76帧。距离单位米；CD=(ACC+Comp)/2未平方距离；F1阈值0.25m、降采样0.0390625m。历史与本轮耗时不是同批配对比较；Dense加速比N/A，未重跑Dense。', '', '|条件效应：正项减负项|AUC@3 pp|AUC@30 pp|ACC m|Comp m|CD m|F1 pp|','|---|---:|---:|---:|---:|---:|---:|']
 for name,d in contrasts.items():text.append('|'+name+'|'+'|'.join(f'{d[k]*(100 if k in ["auc03","auc30","f1"] else 1):+.6f}' for k in FIELDS)+'|')
 text += ['', 'interaction = D-B-C+A；不将两项同时改变造成的差异全归给其中一项。AUC/F1越高越好，距离越低越好。', '', f'完整性检查通过：{complete}；新运行{len(audits)}个场景；不变对照运行0次。原始逐场景指标、配置和层记录见各variant/result.json及layers.json，源码哈希和冻结日期见manifest.json。']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(complete=complete,macro=macro,contrasts=contrasts,failed_audits=[a for a in audits if not a['passed']]),indent=2))
if __name__=='__main__':main()
