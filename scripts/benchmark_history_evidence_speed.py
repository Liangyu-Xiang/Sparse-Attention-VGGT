"""Identical synthetic128-frame shapes: AV plus evidence;no full controls."""
import json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.teacher_history_probability_tiles import attention
from saf3r.patch.current_probe_reuse import main_attention
from scripts.search_quality_speed import save
from scripts.quality_metrics import quality_record

torch.manual_seed(0);torch.set_num_threads(2);h,n,d,g=16,133760,64,64;s=640;a=int(n*.18)-s
q=torch.randn(h,g*256,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);special=torch.arange(s,device='cuda');ids=torch.stack([torch.randperm(n-s,device='cuda')[:a]+s for _ in range(g)]).int();w=torch.exp(torch.rand(g,a,device='cuda')*2)
result=[]
for name,fn in [('bare',lambda:main_attention(q,k,v,ids,special,d**-.5,w.log())[0]),*[(f'late_den_s{st}',lambda st=st:attention(q,k,v,ids,special,w,d**-.5,4,True,4,st)[0]) for st in [1,2]]]:
 out=fn();torch.cuda.synchronize();times=[]
 for rep in range(6):
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();value=fn();end.record();end.synchronize();times.append(begin.elapsed_time(end)/1000)
 result.append(dict(name=name,seconds=statistics.median(times),samples=times,kernel=attention.__globals__.get('LAST_KERNEL_METADATA')));print(result[-1],flush=True)
 if name=='bare':ref=out.clone()
 else:result[-1]['relative_l2']=float((out.float()-ref.float()).norm()/ref.float().norm())
save(Path(sys.argv[1]),dict(quality=quality_record({},'Same synthetic shapes local AV/evidence microbenchmark;no geometry'),rows=result,shape=dict(heads=h,n=n,d=d,groups=g,extra_keys=a,specials=s),speedup_vs_dense=None,speedup_reason='Kernel benchmark,not full forward'))
