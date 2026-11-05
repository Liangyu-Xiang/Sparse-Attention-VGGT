"""Audit all runs and report success/failure against pre-frozen acceptance gates."""
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs')
FIELDS=('auc03','auc30','acc','comp','cd','f1');RECON=('acc','comp','cd','f1')
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def csvout(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;plan=read(out/'plan.json');completion=read(out/'completion.json');assert completion['complete']
 audit=[];rows=[];timings=[];source={}
 for ds,scenes in plan['datasets'].items():
  for sc in scenes:
   group={}
   for name in ('reference','saf3r','optimized'):
    reused=ds=='eth3d' and name=='reference'
    reference_reused=bool(plan.get('reuse_control_batch')) and name=='reference'
    folder=(BASE/'eth3d_cluster20_20261105/runs/eth3d'/sc/'integrated_main_f20') if reused else out/'runs'/ds/sc/name
    raw=read(folder/('timing_result.json' if name=='saf3r' else 'result.json'));group[name]=raw
    if name!='saf3r':
     q=raw['quality'];assert q['complete'] and all(math.isfinite(q['values'][k]) for k in FIELDS)
     assert abs(q['values']['cd']-(q['values']['acc']+q['values']['comp'])/2)<1e-9
    if not reused:
     mon=read(out/'monitors'/f'{ds}_{sc}_{name}.json');assert not mon['same_gpu_overlap_detected'] and mon['returncode']==0
     assert mon['max_simultaneous_gpu_applications']==1
     log=(out/'logs'/f'{ds}_{sc}_{name}.log').read_text();assert log.count('FORWARD ')==4
     if name=='saf3r':assert log.count('Computing QK TopK indices')==4
    t=raw['timing'];assert t['warmup']==1 and t['repeats']==3 and len(t['samples'])==3
    assert abs(t['median_seconds']-statistics.median(x['seconds'] for x in t['samples']))<1e-10
    for i,x in enumerate(t['cold_samples']+t['samples']):timings.append(dict(dataset=ds,scene=sc,method=name,reused=reused,iteration=i,cold=i<len(t['cold_samples']),**{k:x[k] for k in ('seconds','peak_allocated_gib','peak_reserved_gib')}))
    if name=='saf3r':
     # Unchanged quality is reused from prior matched VGGT controls, never timing-only N/A replaced in raw data.
     previous=BASE/'combined_reuse_20261031'/ds/sc/'saf3r/result.json'
     if not previous.exists():
      candidates=list((BASE/'combined_reuse_20261031').rglob(f'{sc}/saf3r/result.json'));assert len(candidates)==1,(ds,sc,candidates);previous=candidates[0]
     control=read(previous);q=control['quality'];assert q['complete'];quality_batch=str(previous)
     for key in ('image_paths','frame_indices','shape'):assert control['input'][key]==raw['input'][key],(ds,sc,'historical SAF3R input',key)
     assert Path(control['input']['checkpoint']).resolve()==Path(raw['input']['checkpoint']).resolve()
     for key,value in control['quality_protocol'].items():assert raw['quality_protocol'][key]==value,(ds,sc,'historical SAF3R protocol',key)
    else:q=raw['quality'];quality_batch='eth3d_cluster20_20261105' if reused else ('cluster20_speed_20261105' if reference_reused else out.name)
    vals=q['values'];row=dict(dataset=ds,scene=sc,method=name,frames=raw['frames'],**{k:vals[k] for k in FIELDS},seconds=t['median_seconds'],peak_warm_allocated_gib=max(x['peak_allocated_gib'] for x in t['samples']),peak_warm_reserved_gib=max(x['peak_reserved_gib'] for x in t['samples']),peak_all_allocated_gib=max(x['peak_allocated_gib'] for x in t['samples']+t['cold_samples']),cold_seconds=t['cold_samples'][0]['seconds'],quality_batch=quality_batch,timing_reused=reused or reference_reused,speedup_vs_dense=None,speedup_vs_saf3r=None)
    rows.append(row)
   ref,new,saf=group['reference'],group['optimized'],group['saf3r']
   for key in ('image_paths','frame_indices','shape','checkpoint_sha256'):
    assert ref['input'][key]==new['input'][key]==saf['input'][key],(ds,sc,key)
   assert ref['quality_protocol']==new['quality_protocol']==saf['quality_protocol']
   assert saf['saf3r']['topk']==4 and saf['saf3r']['lazy_dino_topk']
   mon_new=read(out/'monitors'/f'{ds}_{sc}_optimized.json');mon_saf=read(out/'monitors'/f'{ds}_{sc}_saf3r.json');assert mon_new['gpu_uuid']==mon_saf['gpu_uuid']
   for row in rows[-3:]:row['speedup_vs_saf3r']=saf['timing']['median_seconds']/row['seconds'] if row['method']!='reference' or (ds=='7scenes' and not plan.get('reuse_control_batch')) else None
   audit.append(dict(dataset=ds,scene=sc,inputs_equal=True,protocol_equal=True,paired_gpu=mon_new['gpu_uuid'],optimized_faster_than_saf3r=new['timing']['median_seconds']<=saf['timing']['median_seconds']))
 macros=[];checks=[]
 for ds,scenes in plan['datasets'].items():
  for method in ('reference','saf3r','optimized'):
   rr=[r for r in rows if r['dataset']==ds and r['method']==method];assert len(rr)==len(scenes)
   m=dict(dataset=ds,method=method,scenes=len(rr),**{k:statistics.mean(r[k] for r in rr) for k in FIELDS},seconds=statistics.mean(r['seconds'] for r in rr),peak_warm_allocated_gib=max(r['peak_warm_allocated_gib'] for r in rr),peak_warm_reserved_gib=max(r['peak_warm_reserved_gib'] for r in rr),peak_all_allocated_gib=max(r['peak_all_allocated_gib'] for r in rr),mean_cold_seconds=statistics.mean(r['cold_seconds'] for r in rr));macros.append(m)
  ref,new,saf=[next(m for m in macros if m['dataset']==ds and m['method']==k) for k in ('reference','optimized','saf3r')]
  changes={k:100*(new[k]-ref[k])/ref[k] for k in FIELDS};gate=all(abs(changes[k])<2 for k in RECON)
  checks.append(dict(dataset=ds,relative_quality_change_percent=changes,reconstruction_macro_pass=gate,speedup_vs_saf3r=saf['seconds']/new['seconds'],speed_macro_pass=new['seconds']<=saf['seconds'],all_scenes_speed_pass=all(x['optimized_faster_than_saf3r'] for x in audit if x['dataset']==ds),seconds_below9_all128=all(r['seconds']<9 for r in rows if r['dataset']==ds and r['method']=='optimized' and r['frames']==128)))
 changes=[]
 for row in rows:
  if row['method']!='optimized':continue
  ref=next(x for x in rows if x['dataset']==row['dataset'] and x['scene']==row['scene'] and x['method']=='reference')
  c=dict(dataset=row['dataset'],scene=row['scene'],**{k:100*(row[k]-ref[k])/ref[k] for k in FIELDS});c['all_reconstruction_absolute_changes_below2pct']=all(abs(c[k])<2 for k in RECON);changes.append(c)
 bad=[k for k,v in plan['source_sha256'].items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v];assert not bad,bad
 summary=dict(complete=True,macro=macros,acceptance=checks,per_scene_changes=changes,audit=audit,all_scenes_quality_pass=all(x['all_reconstruction_absolute_changes_below2pct'] for x in changes),all_quality_macro_pass=all(x['reconstruction_macro_pass'] for x in checks),all_scenes_speed_pass=all(x['all_scenes_speed_pass'] for x in checks),speedup_vs_original_vggt_dense=None,dense_speedup_reason='No Dense rerun authorized; historical timings are not paired',source_unchanged=True)
 write(out/'summary.json',summary);write(out/'audit.json',dict(passed=True,new_runs=len(completion['runs']),method_records=len(rows),scenes=18,source_unchanged=True,checks=audit));csvout(out/'per_scene.csv',rows);csvout(out/'macro.csv',macros);csvout(out/'quality_changes.csv',changes);csvout(out/'timing.csv',timings)
 lines=['# Cluster20 equivalent engineering: completed evaluation','',f"All{len(completion['runs'])} new runs complete; {len(rows)} method/scene records including reused references. Quality macro gate: {summary['all_quality_macro_pass']}; every-scene paired speed gate: {summary['all_scenes_speed_pass']}; every-scene point-cloud change<2%: {summary['all_scenes_quality_pass']}.",'','VGGT; same .20 patch-cluster fraction,10%/22%/10% attention retention,256-query mask sharing,one current-Value local update,cluster_population weights,16 budget steps. No probability/state/objective changes or test-truth tuning. Engineering configuration is frozen in each plan; first-candidate and conservative variants are reported separately. Precision/self-repeat diagnostics are retained in the first campaign.1 warmup+3 measured complete forwards; median per scene, equal-scene macros. RTX5090D,GPU4/5/6,strict same-GPU scene pairs,all initialization/clustering/history/routing/checks/reordering and prediction heads included. Model loading/image decode/geometry evaluation excluded. Any enabled cuDNN first-call search remains in recorded cold forward; warm and cold memory both shown.','', 'End-to-end implementation comparison: common allocator/RoPE/residual improvements are enabled only for the changed method. This is not evidence that the sparse selector itself is faster than SAF3R, or that every optimized SAF3R implementation is slower. SAF3R timing rerun explicitly authorized; quality reused. SAF3R is lazy first-correspondence Top4 per frame, not paper DINO TopK. Dense not rerun: formal original-Dense speedup N/A.7Scenes128-frame .20-cluster reference quality was newly completed in the first campaign and reused in the conservative campaign. ETH3D reference reuses the earlier complete20% batch. Reused reference timings are historical for the conservative candidate, not paired.','',table(['Dataset','Method','Scenes','AUC@3 %','AUC@30 %','ACC m','Comp m','CD m','F1 %','Forward s','Warm allocated GiB','All-forward peak GiB'],[[m['dataset'],m['method'],m['scenes'],*[f"{m[k]*(100 if k in ('auc03','auc30','f1') else 1):.6f}" for k in FIELDS],f"{m['seconds']:.6f}",f"{m['peak_warm_allocated_gib']:.6f}",f"{m['peak_all_allocated_gib']:.6f}"] for m in macros]),'', 'AUC/F1 displayed as percentages; JSON/CSV fractions. Distances unsquared metres; CD=(ACC+Comp)/2.7Scenes includes seven existing scene/sequence entries at128frames each, not the full official test split.7Scenes F1 threshold/downsample .05/.0078125m; ETH3D .25/.0390625m; unchanged native unposed alignment/fusion and exact input lists/checkpoint hash audited.','',table(['Dataset','ΔACC %','ΔComp %','ΔCD %','ΔF1 %','Quality <2%','vs SAF3R speedup','All scenes faster'],[[c['dataset'],*[f"{c['relative_quality_change_percent'][k]:+.6f}" for k in RECON],c['reconstruction_macro_pass'],f"{c['speedup_vs_saf3r']:.4f}x",c['all_scenes_speed_pass']] for c in checks]),'', 'Relative quality changes are absolute-tested (<2%, not2 percentage points); improvements exceeding2% are also flagged under the requested change criterion. Per-scene deviations retained in quality_changes.csv; macro gates do not establish every-scene equivalence. No threshold relaxation or post-quality parameter changes.','', '## Per-scene quality and timing','',table(['Dataset','Scene','Method','AUC3 %','AUC30 %','ACC m','Comp m','CD m','F1 %','Forward s','SAF3R speedup'],[[r['dataset'],r['scene'],r['method'],*[f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.5f}" for k in FIELDS],f"{r['seconds']:.6f}",'N/A historical' if r['speedup_vs_saf3r'] is None else f"{r['speedup_vs_saf3r']:.4f}x"] for r in rows])]
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
