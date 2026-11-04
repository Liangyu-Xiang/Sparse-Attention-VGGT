"""Synthetic 128-frame clustering only; no full-forward speedup claim."""
import importlib.util,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('source_local',ROOT/'saf3r/patch/source_local_unanchored.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2);torch.manual_seed(20261105)
out=ROOT/'outputs/source_local_unanchored_20261105';rows=[]
with torch.inference_mode():
 x=torch.randn(128*28*37,64,device='cuda')
 for layout in ('sequential','spatiotemporal'):
  for width in (16,32,64):
   for span in ((1,) if layout=='sequential' else (1,2,4)):
    init=[];updates=[];peak=[]
    for repeat in range(4):
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
     s=m.SourceLocalUnanchoredValue(128,28,37,26009,'cuda',width,2,layout,span)
     torch.cuda.synchronize();t1=time.perf_counter()
     for layer in range(3):s.update(x+layer*.01,layer)
     torch.cuda.synchronize();t2=time.perf_counter()
     assert bool((torch.bincount(s.labels,minlength=26009)>0).all())
     assert torch.equal(s.cluster_source[s.labels],s.source)
     assert torch.equal(s.labels[s.reps],torch.arange(26009,device='cuda'))
     if repeat:init.append(t1-start);updates.append((t2-t1)/3);peak.append(torch.cuda.max_memory_allocated())
     groups=len(s.table); del s
    row=dict(layout=layout,source_width=width,temporal_span=span,iterations=2,source_groups=groups,
             initialization_seconds=init,update_seconds=updates,median_update_seconds=sorted(updates)[1],
             peak_allocated_bytes=max(peak),status='invariants_passed',auc3=None,auc30=None,acc=None,comp=None,cd=None,f1=None,
             quality_unavailable_reason='Synthetic clustering only; no model/camera/reconstruction evaluation',dense_speedup=None,
             dense_speedup_unavailable_reason='No full-forward timing or paired Dense')
    rows.append(row);print(json.dumps(row),flush=True)
    (out/'gpu_diagnostics.json').write_text(json.dumps(dict(scope='synthetic 128-frame 28x37 patches; 64D random features; initialization plus 3 cluster updates, 1 warmup, 3 repeats',rows=rows,quality_complete=False,reconstruction_threshold=None,downsampling=None,saf3r_topk=None,hardware=torch.cuda.get_device_name()),indent=2)+'\n')
 lines=['Synthetic clustering diagnostic; GPU only, no full-forward or dataset quality evaluation.','', '| Layout | Width (clusters) | Frames/source tile | Source groups | Median update (ms) | AUC@3 (fraction) | AUC@30 (fraction) | ACC (m) | Comp (m) | CD (m) | F1 (fraction) | Dense speedup |','|---|---|---|---|---|---|---|---|---|---|---|---|']
 for r in rows:lines.append(f"| {r['layout']} | {r['source_width']} | {r['temporal_span']} | {r['source_groups']} | {r['median_update_seconds']*1000:.3f} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
 lines+=['','All six quality fields unavailable: synthetic clustering only, no camera poses or recon_unposed. Dense speedup unavailable: no full-forward measurement or paired Dense. Timings cannot predict real-sequence quality or full model speed. Raw timings include separate initialization and peak allocated memory.']
 (out/'gpu_diagnostics.md').write_text('\n'.join(lines)+'\n')
