"""Same-input no-GT checks and realistic-shape M5c microbenchmark."""
import argparse,json,sys,time,statistics,copy
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
FORK=Path('/tmp/zdrop_m5c_pipeline_dynamic_20261102');sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import torch
from saf3r.patch import reuse_m5c_equivalent as ref,reuse_m5c_pipeline as fast
OUT=ROOT/'outputs/m5c_pipeline_dynamic_20261102';OUT.mkdir(exist_ok=True)

def case(g,h,n,p,r,a,d,dtype):
 grid=torch.arange(p,device='cuda')%r;sizes=torch.bincount(grid,minlength=r).float()
 state=NS(n=n,grid=grid,regions=r,grid_sizes=sizes,patches=torch.arange(p,device='cuda'),specials=torch.arange(p,n,device='cuda'),fine=NS(base_count=min(p,17),labels=torch.arange(p,device='cuda')%min(p,17)),m5c_previous={},m5c_next={},m5c_seed_errors=[],m5c_prior_errors=[],m5c_probe_qk=0,m5c_reconstruction_qk=0,m5c_count_errors=[])
 q=torch.randn(h,n,d,device='cuda',dtype=dtype);k=torch.randn_like(q);state.history_pooled_fp32=torch.randn(h,r,d,device='cuda')*.2
 ids=torch.stack([torch.randperm(p,device='cuda')[:a] for _ in range(g)]).int();ids[0,-3:]=-1;ids[1].fill_(-1)
 obs=ref.observe(state,q,k,0,g,d**-.5)
 return state,q,k,ids,obs

def diff(a,b):
 x=(a-b).float();return dict(max_abs=float(x.abs().max()),relative_l2=float(x.norm()/a.float().norm().clamp_min(1e-30)))

def check(seed,dtype,a=67):
 torch.manual_seed(seed);s,q,k,ids,obs=case(5,3,129,113,7,a,32,dtype);a=copy.copy(s);b=copy.copy(s)
 for st in [a,b]:st.m5c_next={};st.m5c_seed_errors=[];st.m5c_prior_errors=[];st.m5c_count_errors=[]
 ref.save_evidence(a,q,k,0,ids,ids,obs,32**-.5);fast.save_evidence(b,q,k,0,ids,ids,obs,32**-.5)
 results={}
 for key,value in b.m5c_next[0].items():
  target=a.m5c_next[0][key].cuda();results[key]=diff(target,value)
  assert torch.allclose(target,value,atol=1e-6,rtol=1e-5),(key,results[key])
 # Feed the same reference history into both prior implementations to isolate fusion.
 old={key:value.cuda() for key,value in a.m5c_next[0].items()}
 a.m5c_previous={0:old};b.m5c_previous={0:old}
 for st in [a,b]:st.m5c_previous_q=q;st.m5c_previous_k=k;st.m5c_previous_scale=32**-.5
 q2=q*.9+.03;obs2=ref.observe(s,q2,k,0,5,32**-.5)
 pa,ca=ref.build_prior(a,0,ids,obs2);pb,cb=fast.build_prior(b,0,ids,obs2)
 for key,x,y in [('prior',pa,pb),('pi',ca,cb)]:
  results[key]=diff(x,y);assert torch.allclose(x,y,atol=1e-6,rtol=1e-5),(key,results[key])
 assert float(b.m5c_seed_errors[0])<1e-5
 from saf3r.patch import zdrop_history_cleanup as rule
 from saf3r.triton import reuse_compact as compact
 labels=s.fine.labels;f=s.fine.base_count;population=torch.bincount(labels,minlength=f).float();reps=torch.arange(f,device='cuda')
 contexts=[rule.gate(pr,pi,population,ids,labels,reps) for pr,pi in [(pa,ca),(pb,cb)]]
 assert all(torch.equal(contexts[0][key],contexts[1][key]) for key in contexts[0]),'History gate differs'
 z=torch.randn(len(labels),32,device='cuda');geo=compact.prepare_geometry(z,labels,reps,population,l2=True);allowed=torch.ones((len(ids),f),device='cuda',dtype=torch.bool)
 allocations=[]
 for pr,pi,ctx in [(pa,ca,contexts[0]),(pb,cb,contexts[1])]:
  moment=torch.zeros(len(ids),f,32,device='cuda').scatter_add_(1,labels[None,:,None].expand(len(ids),-1,32),pr[:,:,None]*z[None])
  if pr is pb:
   from saf3r.triton.reuse_drop_moment import prepare,drop_norm
   geo['drop_values']=drop_norm(pr,z,geo,f,prepare(population))
  else:geo['drop_values']=moment.norm(dim=-1)
  errors,costs=rule.states(pi,z,reps,labels,population,None,allowed,pr,1.,geo,ctx)
  allocations.append([compact.budget(geo['objective'],costs,errors[:,:,0].amax(1)+1e-12,cap,16)[0] for cap in [15,30,60,90]])
 assert all(torch.equal(x,y) for x,y in zip(*allocations)),'State choices differ'
 return dict(seed=seed,dtype=str(dtype),support_width=ids.shape[1],tail_query=True,no_history_group=True,history_gate_identical=True,four_budget_state_choices_identical=True,differences=results,normalization_error=float(b.m5c_seed_errors[0]))

def bench():
 torch.manual_seed(100);s,q,k,ids,obs=case(64,16,133248,132608,512,26009,64,torch.bfloat16)
 ref.save_evidence(s,q,k,0,ids,ids,obs,64**-.5)
 old={key:value.cuda() for key,value in s.m5c_next[0].items()}
 s.m5c_previous_q=q;s.m5c_previous_k=k;s.m5c_previous_scale=64**-.5
 data={}
 for name,api in [('reference',ref),('optimized',fast)]:
  data[name]={}
  for op in ['build_prior','save_evidence']:
   times=[];memory=[]
   for i in range(4):
    s.m5c_previous={0:old};s.m5c_next={};s.m5c_seed_errors=[];s.m5c_prior_errors=[]
    torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    if op=='build_prior':result=api.build_prior(s,0,ids,obs)
    else:result=api.save_evidence(s,q,k,0,ids,ids,obs,64**-.5)
    torch.cuda.synchronize();dt=time.perf_counter()-t
    if i:times.append(dt);memory.append((torch.cuda.max_memory_allocated()-base)/2**20)
    del result
   data[name][op]=dict(median_seconds=statistics.median(times),samples=times,peak_increment_mib=max(memory))
  print('BENCH',name,data[name],flush=True)
 return data

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--checks-only',action='store_true');args=parser.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 checks=[check(seed,dtype,a) for dtype in [torch.float32,torch.bfloat16] for seed in [0,1,2] for a in [67,113]]
 print('PASS TWELVE PIPELINE PROBABILITY/DROP/STATE CASES',flush=True)
 result=dict(checks=checks,microbenchmark=None if args.checks_only else bench(),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='No-GT numerical and kernel diagnostic; no pose/recon_unposed evaluation',dense_speedup=None,dense_speedup_reason='Unchanged Dense not rerun')
 (OUT/('extended_checks.json' if args.checks_only else 'numerical_microbenchmark.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
