"""CPU-only aggregation of frozen camera diagnostics; no model reruns or tuning."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/camera_token_comparison_20261105'
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvsave(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def report():
 import torch
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from scripts.camera_token_diagnostic import similarity
 torch.set_num_threads(2)
 plan=read(OUT/'frozen_plan.json');scenes=plan['scenes'];methods=['ours','saf3r'];rows=[];cf=[];local=[];att=[];heads=[];direct=[];end=[];runtime=[]
 for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 for g in [4,5,6]:
  mon=read(OUT/f'worker{g}_gpu.json');assert mon['returncode']==0 and not mon['same_gpu_overlap_detected']
 for scene in scenes:
  ds={m:read(OUT/'full'/scene/m/'result.json') for m in ['dense']+methods};ins={m:read(OUT/'full'/scene/m/'input.json') for m in ds}
  for m,d in ds.items():
   assert d['complete'] and len(d['local_operator_checks'])==48
   for k in ['checkpoint_sha256','image_paths','frame_indices','shape']:assert ins[m][k]==ins['dense'][k],(scene,m,k)
   assert ins[m]['checkpoint_sha256']==plan['checkpoint_sha256']
   runtime.append(dict(scene=scene,method=m,instrumented_seconds=d['instrumented_seconds'],peak_allocated_gib=d['peak_allocated_gib'],warmed_full_forward_seconds=None,dense_speedup=None,reason='Instrumented diagnostics; no paired production timing'))
   for r in d['local_operator_checks']:
    assert r['reconstruction']['relative_l2']<=.03
    local.append(dict(scene=scene,method=m,kind=r['kind'],layer=r['layer']+1,reconstruction_l2=r['reconstruction']['relative_l2'],same_qkv_l2=r['actual_vs_dense_same_qkv']['relative_l2']))
  for m in methods:
   for r in ds[m]['features']:
    for token,v in r.items():
     if isinstance(v,dict) and 'relative_l2' in v:rows.append(dict(scene=scene,method=m,stage=r['stage'],token=token,**{k:v[k] for k in ['cosine','relative_l2','norm_ratio']}))
   for r in ds[m]['attention']:
    att.append(dict(scene=scene,method=m,kind=r['kind'],layer=r['layer']+1,**r['against_dense'],local_same_qkv_l2=r['local_same_qkv']['relative_l2']))
    for h in range(16):heads.append(dict(scene=scene,method=m,kind=r['kind'],layer=r['layer']+1,head=h,**{k:float(np.mean(v[h])) for k,v in r['per_head_camera'].items()}))
   for r in ds[m]['counterfactuals']:
    for case,v in r['against_dense'].items():cf.append(dict(scene=scene,method=m,kind=r['kind'],layer=r['layer']+1,case=case,cosine=v['cosine'],relative_l2=v['relative_l2']))
  tensors={m:torch.load(OUT/'full'/scene/m/'camera_tensors.pt',map_location='cpu',weights_only=True,mmap=True) for m in ds}
  ts={m:t['camera_features_and_attention_summaries'] for m,t in tensors.items()}
  for stage,v in ts['ours'].items():
   if isinstance(v,dict) and 'camera' in v:
    a=v['camera'];b=ts['saf3r'][stage]['camera'];s=similarity(a,b,False);direct.append(dict(scene=scene,stage=stage,cosine=s['cosine'],relative_l2=s['relative_l2'],reference='saf3r'))
  for m in methods:
   for kind in ['frame','global']:
    for i in range(24):
     stage=f'{kind}_{i:02d}_attention_pre_Wo';a=ts[m][stage]['camera'].float().reshape(-1,16,64);b=ts['dense'][stage]['camera'].float().reshape_as(a)
     for h in range(16):
      s=similarity(a[:,h],b[:,h],False)
      # Separate table: per-head feature output, distinct from attention PMF.
      end.append(dict(scene=scene,method=m,kind=kind,layer=i+1,head=h,cosine=s['cosine'],relative_l2=s['relative_l2']))
  if scene=='pipes':
   fig,axs=plt.subplots(3,4,figsize=(13,8),layout='constrained')
   for ri,m in enumerate(['dense']+methods):
    for ci,i in enumerate([0,7,15,23]):
     v=ts[m][f'global_{i:02d}_attention_summary']['frame_value_mass'].float().mean(0).numpy();im=axs[ri,ci].imshow(v,vmin=0,vmax=1,cmap='viridis');axs[ri,ci].set_title(f'{m}, global {i+1}');axs[ri,ci].set_xlabel('Value source frame');axs[ri,ci].set_ylabel('Camera query frame')
   fig.colorbar(im,ax=axs,shrink=.6,label='attention mass (head mean)');fig.savefig(OUT/'frame_attention_pipes.png',dpi=160);plt.close(fig)
  del tensors,ts
  print('aggregated',scene,flush=True)
 def macro(data,keys,metrics):
  groups={}
  for r in data:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
  return [dict(zip(keys,key),**{k:float(np.mean([r[k] for r in group])) for k in metrics},scenes=len(group)) for key,group in groups.items()]
 fm=macro(rows,['method','stage','token'],['cosine','relative_l2','norm_ratio']);am=macro(att,['method','kind','layer'],['cosine','relative_l2','js_nats','total_variation','local_same_qkv_l2']);cm=macro(cf,['method','kind','layer','case'],['cosine','relative_l2']);hm=macro(heads,['method','kind','layer','head'],['cosine','relative_l2','js_nats','total_variation']);em=macro(end,['method','kind','layer','head'],['cosine','relative_l2'])
 for name,data in [('features_per_scene',rows),('features_macro',fm),('attention_per_scene',att),('attention_macro',am),('attention_per_head',hm),('counterfactual_per_scene',cf),('counterfactual_macro',cm),('local_operator_checks',local),('head_output_macro',em),('ours_vs_saf3r_features',direct),('instrumented_resources',runtime)]:csvsave(OUT/(name+'.csv'),data)
 save(OUT/'summary.json',dict(scope=plan['sampling'],scene_count=len(scenes),frames=sum(len(read(OUT/'full'/s/'dense'/'input.json')['image_paths']) for s in scenes),feature_macro=fm,attention_macro=am,counterfactual_macro=cm,local_operator_max={m:{k:max(r[k] for r in local if r['method']==m) for k in ['reconstruction_l2','same_qkv_l2']} for m in ['dense']+methods},quality_reason='Attention/features diagnostic only; no pose GT scoring or point-cloud reconstruction',production_timing_reason='Instrumentation adds probability reconstruction, CPU transfer and counterfactuals; not a production timing comparison'))
 def feature(m,stage,metric,token='camera'):return next(r[metric] for r in fm if r['method']==m and r['stage']==stage and r['token']==token)
 colors={'ours':'#d35400','saf3r':'#2471a3'}
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for row,kind in enumerate(['frame','global']):
  for m in methods:
   for col,metric in enumerate(['cosine','relative_l2']):axs[row,col].plot(range(1,25),[feature(m,f'{kind}_{i:02d}_block_output',metric) for i in range(24)],label=m,color=colors[m]);axs[row,col].set_title(f'{kind}: camera block output {metric} vs Dense');axs[row,col].set_xlabel('Layer');axs[row,col].grid(alpha=.2);axs[row,col].legend()
 fig.savefig(OUT/'camera_feature_layers.png',dpi=170);fig.savefig(OUT/'camera_feature_layers.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for col,metric in enumerate(['cosine','relative_l2','js_nats']):
  for m in methods:axs[col].plot(range(1,25),[next(r[metric] for r in am if r['method']==m and r['kind']=='global' and r['layer']==i) for i in range(1,25)],label=m,color=colors[m])
  axs[col].set_title(f'Global camera attention {metric} vs Dense');axs[col].set_xlabel('Layer');axs[col].grid(alpha=.2);axs[col].legend()
 fig.savefig(OUT/'camera_attention_layers.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
 for j,m in enumerate(methods):
  for row,(data,metric,title) in enumerate([(hm,'js_nats','Attention JS (nats)'),(em,'relative_l2','Pre-Wo output relative L2')]):
   mat=np.array([[next(r[metric] for r in data if r['method']==m and r['kind']=='global' and r['layer']==i and r['head']==h) for h in range(16)] for i in range(1,25)])
   im=axs[row,j].imshow(mat,aspect='auto',origin='lower');axs[row,j].set_title(f'{m}: {title}');axs[row,j].set_xlabel('Head (0-based)');axs[row,j].set_ylabel('Global layer (row+1)');fig.colorbar(im,ax=axs[row,j])
 fig.savefig(OUT/'camera_head_heatmaps.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for token in ['camera_query','camera_key','camera_value','patch_key','patch_value']:axs[0].plot(range(1,25),[feature('ours',f'global_{i:02d}_qkv_drift','relative_l2',token) for i in range(24)],label=token)
 for case in ['current_Q_only','current_K_only','current_V_only','current_QK_dense_V','own_dense_all_qkv_drift']:axs[1].plot(range(1,25),[next(r['relative_l2'] for r in cm if r['method']=='ours' and r['kind']=='global' and r['layer']==i and r['case']==case) for i in range(1,25)],label=case)
 for ax,title in zip(axs,['Ours: Q/K/V drift vs Dense','Ours: conditional attention-output error vs Dense']):ax.set_title(title);ax.set_xlabel('Global layer');ax.set_ylabel('Relative L2');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.savefig(OUT/'qkv_counterfactuals.png',dpi=170);plt.close(fig)
 stages=['camera_head_input','camera_head_normalized_input']+[f'camera_head_iteration{i}_block{j}' for i in range(4) for j in range(4)]
 fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
 for m in methods:ax.plot(range(len(stages)),[feature(m,s,'relative_l2') for s in stages],marker='.',label=m,color=colors[m])
 ax.set_xticks(range(len(stages)),['input','norm']+[f'{i+1}/{j+1}' for i in range(4) for j in range(4)],rotation=45);ax.set_xlabel('Camera head iteration/block');ax.set_ylabel('Relative L2 vs Dense');ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'camera_pose_head.png',dpi=170);plt.close(fig)
 # Human-readable complete global-layer table, all numbers equal-scene means.
 lines=['# Camera token diagnosis: A + all-special correction + cluster_fraction=0.20','','Scope: ETH3D filtered11, 377 exact matched frames; 24 frame + 24 global blocks, all16 heads and camera rows. Layer numbers below are 1-based. Dense is the feature reference (cosine1, relativeL2=0). No production code/config change.','','Attention metrics compare effective Value-source token probability distributions. SAF3R broadcast_first uses first-frame scoring Q/K but target-frame Values; score ownership is retained separately in raw tensor summaries. SAF3R is local Top-4 lazy first-correspondence-head, not paper DINO Top-K.','','Cosine = <x,d>/(||x||||d||); relative L2 = ||x-d||/||d||. Feature rows flatten all frame camera vectors per scene, then average scenes equally. Attention rows average head/query metrics within scene, then scenes equally. JS uses natural logarithms. Similarity is diagnostic, not pose accuracy.','','| Global layer | Ours block cos | Ours block relL2 | SAF block cos | SAF block relL2 | Ours attention JS | SAF attention JS |','|---|---:|---:|---:|---:|---:|---:|']
 for i in range(24):
  vals=[feature(m,f'global_{i:02d}_block_output',k) for m in methods for k in ['cosine','relative_l2']]+[next(r['js_nats'] for r in am if r['method']==m and r['kind']=='global' and r['layer']==i+1) for m in methods];lines.append('| '+str(i+1)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
 lines+=['','## Camera head input, per scene','','| Scene | Ours cosine | Ours relL2 | SAF cosine | SAF relL2 |','|---|---:|---:|---:|---:|']
 for scene in scenes:
  vals=[next(r[k] for r in rows if r['scene']==scene and r['method']==m and r['stage']=='camera_head_input' and r['token']=='camera') for m in methods for k in ['cosine','relative_l2']];lines.append('| '+scene+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
 lines+=['| Equal-scene macro | '+' | '.join(f'{feature(m,"camera_head_input",k):.6f}' for m in methods for k in ['cosine','relative_l2'])+' |','','## Six required quality metrics and speed','','| Method | AUC@3 (fraction) | AUC@30 (fraction) | ACC | Comp | CD | F1 (fraction) | Dense full-forward speedup |','|---|---|---|---|---|---|---|---|']
 for m in ['Dense','Ours','SAF3R']:lines.append(f'| {m} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
 lines+=['','All six unavailable: this is a camera attention/features diagnostic, without GT pose scoring or unposed reconstruction. Per-run quality.json/quality.md record that reason. Instrumented runtime/peak memory are in instrumented_resources.csv; production speedup unavailable because counterfactuals, CPU transfer and full attention reconstruction change execution. Dense=1.00x only defines the production speed reference; no paired warmed production timings were measured here. No accuracy claim follows from cosine alone.','','Raw per-scene JSON and camera_tensors.pt are retained locally. Tracked CSVs cover frame/global layers, all captured stages, Q/K/V, attention heads, same-QKV operator checks and counterfactuals. Direct ours-vs-SAF feature relativeL2 uses SAF as denominator. Counterfactual effects are nonlinear conditional sensitivities, not additive causal percentages.','','Postprocessing recovery: all33 model diagnostics succeeded; initial coordinator ended with missing report module. This CPU reporter was then added and run without rerunning model forwards.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 e=read(OUT/'execution.json');e.update(postprocessing='recovered: reporter added after initial ModuleNotFoundError; no model reruns',report_complete=True);save(OUT/'execution.json',e)
 print('REPORT COMPLETE',flush=True)
if __name__=='__main__':report()
