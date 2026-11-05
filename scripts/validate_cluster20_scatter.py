"""Paired old/new cluster trajectories with deterministic reductions, no GT."""
import argparse,json,sys,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
from saf3r.patch.source_local_unanchored import SourceLocalUnanchoredValue as Fast
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('reference',ROOT/'reference/cluster20_speed_20261105/source_local_unanchored.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
torch.manual_seed(631);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
rows=[]
for frames,height,width,k,iterations in [(1,3,5,4,2),(3,8,8,38,2),(2,11,13,57,1),(128,37,28,26521,1)]:
    old=mod.SourceLocalUnanchoredValue(frames,height,width,k,'cuda',iterations=iterations)
    new=Fast(frames,height,width,k,'cuda',iterations=iterations,fused_scatter=True)
    for layer in range(4):
        x=torch.randn(frames*height*width,64,device='cuda')
        if layer==1:x.zero_() # tied distances and migration back to old labels
        if layer==2:x[:len(x)//2]=x[0] # many coincident points/empty repair
        old.update(x,layer);new.update(x,layer)
        row=dict(frames=frames,height=height,width=width,clusters=k,iterations=iterations,layer=layer,labels_equal=torch.equal(old.labels,new.labels),reps_equal=torch.equal(old.reps,new.reps))
        rows.append(row);assert row['labels_equal'] and row['reps_equal'],row
result=dict(passed=True,cases=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Clustering numerical diagnostic only; no pose/reconstruction',speedup_vs_dense=None,speedup_reason='Operator diagnostic, no paired Dense model timing')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),flush=True)
