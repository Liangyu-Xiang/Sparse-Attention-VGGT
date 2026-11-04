import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/dense_p0_v2_comparison_20261104';rows=[];layers=[];summary=[]
for method in ['dense','p0']:
 d=json.loads((OUT/method/'result.json').read_text());g=d['profile']['groups'];core=g['dense_global_core' if method=='dense' else 'sparse_core_fused_evidence']['inclusive_seconds'];global_s=g['global_attention']['inclusive_seconds']
 summary.append(dict(method=method,median_seconds=d['median_seconds'],dense_speedup=d['dense_speedup'],samples=d['samples'],profile_seconds=d['profile']['wall_seconds'],core_seconds=core,global_attention_seconds=global_s,outside_global_seconds=d['profile']['wall_seconds']-global_s))
 events=d['profile']['events'];by={x['id']:x for x in events}
 def inside(row,ancestor):
  while row['parent'] is not None:
   row=by[row['parent']]
   if row['id']==ancestor:return True
  return False
 for layer,glob in enumerate(x for x in events if x['operation']=='global_attention'):
  children=[x for x in events if inside(x,glob['id'])];core_s=sum(x['exclusive_seconds'] for x in children if x['operation'] in ['dense_global_core','sparse_core_fused_evidence']);common=sum(x['exclusive_seconds'] for x in children if x['operation'] in ['global_qkv','global_qk_norm','global_rope','global_output_projection'])
  layers.append(dict(method=method,layer=layer,global_s=glob['inclusive_seconds'],core_s=core_s,common_qkv_norm_rope_proj_s=common,other_global_s=glob['inclusive_seconds']-common-core_s))
(OUT/'summary.json').write_text(json.dumps(dict(complete=True,results=summary,equivalence_accepted=False,reason='Corrected P0 still fails ALL-window routing at L1/window61;not promoted'),indent=2)+'\n')
with (OUT/'per_layer.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(layers[0]));w.writeheader();w.writerows(layers)
text=['# Corrected P0 versus original Dense, paired heads128','', 'Same GPU7,exact128 frames,checkpoint,preprocessing,BF16,TF32 disabled,onewarmup+three complete forwards. CPU-only lock and single CUDA process through full subprocess exit. Profiles are separate,not a partition of medians. No quality measured by these timing diagnostics; all six fields N/A with reasons in raw results and diagnostic tables. Full geometry quality is in ../v8_p0_quality_v3_20261104/report.md.','', '|method|warmed median s|Dense speedup|profile core s|profile global Attention s|profile outside global s|','|---|---|---|---|---|---|']
for r in summary:text.append('|'+r['method']+'|'+'|'.join(f"{r[k]:.6f}" for k in ['median_seconds','dense_speedup','core_seconds','global_attention_seconds','outside_global_seconds'])+'|')
text+=['','Full24-layer rows are in per_layer.csv. Sparse core includes fused probe/Grid evidence, not sorting/reduction wrapper. Common backbone/frame Attention/FFNs/heads remain outside global Attention. Corrected P0 fails the later all-window equivalence check and is slower than original Dense. No equivalent-execution optimization success is claimed. Source is frozen by ../v8_p0_quality_v3_20261104/source_sha256.json and archive. Its method differs from initial P0 in ../dense_sparse_comparison_20261104.']
(OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary,indent=2))
