"""No-GT Drop arithmetic checks and same-input microbenchmark."""
import json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(Path('/tmp/zdrop_m5c_pipeline_dynamic_20261102')))
import torch,triton
from saf3r.triton.reuse_drop_moment import prepare,drop_norm
OUT=ROOT/'outputs/m5c_pipeline_dynamic_20261102';OUT.mkdir(exist_ok=True)

def geometry(labels,f):
 pop=torch.bincount(labels,minlength=f);width=triton.next_power_of_2(int(pop.max()));order=labels.argsort(stable=True);start=pop.cumsum(0)-pop;rank=torch.empty_like(labels)
 rank[order]=torch.arange(len(labels),device='cuda')-start[labels[order]]
 members=torch.full((f,width),-1,device='cuda',dtype=torch.int32);members[labels,rank]=torch.arange(len(labels),device='cuda',dtype=torch.int32)
 return pop,dict(members=members,width=width)

def reference(prior,z,labels,f):
 total=torch.zeros((len(prior),f,z.shape[1]),device='cuda')
 for j in range(len(prior)):total[j].index_add_(0,labels,prior[j,:,None]*z)
 return total.norm(dim=-1)

def main():
 torch.manual_seed(0);torch.set_num_threads(2);checks=[]
 for d in [17,64]:
  for g in [1,3]:
   n,f=257,29;labels=torch.cat((torch.arange(f,device='cuda'),torch.randint(f,(n-f,),device='cuda')));pop,geo=geometry(labels,f)
   z=torch.randn(n,d,device='cuda');prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True);bins=prepare(pop)
   old=reference(prior,z,labels,f);new=drop_norm(prior,z,geo,f,bins);delta=(old-new)
   assert torch.allclose(old,new,rtol=1e-5,atol=1e-6)
   checks.append(dict(d=d,g=g,max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/old.norm())))
 g,n,f,d=64,132608,26009,64
 labels=torch.cat((torch.arange(f,device='cuda'),torch.randint(f,(n-f,),device='cuda')))
 extra=64-int((labels==0).sum());where=torch.where(labels!=0)[0][f:f+extra];labels[where]=0
 pop,geo=geometry(labels,f);z=torch.randn(n,d,device='cuda');prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True)
 t=time.perf_counter();bins=prepare(pop);torch.cuda.synchronize();prepare_seconds=time.perf_counter()-t
 data={}
 for name,fn in [('reference',lambda:reference(prior,z,labels,f)),('binned',lambda:drop_norm(prior,z,geo,f,bins))]:
  samples=[];memory=[]
  for i in range(6):
   torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();result=fn();torch.cuda.synchronize();dt=time.perf_counter()-t
   if i:samples.append(dt);memory.append((torch.cuda.max_memory_allocated()-base)/2**20)
   del result
  data[name]=dict(median_seconds=statistics.median(samples),samples=samples,peak_increment_mib=max(memory));print(name,data[name],flush=True)
 old=reference(prior,z,labels,f);new=drop_norm(prior,z,geo,f,bins);assert torch.allclose(old,new,rtol=1e-5,atol=1e-6)
 result=dict(checks=checks,realistic_shape=[g,n,f,d],width=geo['width'],bins=[(size,len(ids)) for size,ids in bins],prepare_seconds=prepare_seconds,microbenchmark=data,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='No-GT Drop arithmetic/microbenchmark only',dense_speedup=None,dense_reason='No unchanged Dense run')
 (OUT/'drop_microbenchmark.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',flush=True)
if __name__=='__main__':main()
