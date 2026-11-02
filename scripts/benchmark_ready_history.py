"""Same-shape no-GT timing of producer/consumer work, no unchanged model run."""
import json,statistics,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),'/tmp/zdrop_ready_history_v5_20261102']
from validate_ready_history import case,weighted_attention,old_attention,ready,reference
import torch
OUT=ROOT/'outputs/ready_history_20261102'

def main():
 torch.set_num_threads(2);torch.manual_seed(2)
 s,q,k,v,ids,obs=case(64,16,133248,132608,512,26009,64)
 from saf3r.patch.reuse_residual_exploration import canonical_layout
 s.grid,s.regions=canonical_layout(128,28,37,2,2,q.device);s.grid_sizes=torch.bincount(s.grid,minlength=s.regions).float()
 ids=torch.stack([torch.randperm(len(s.grid),device=q.device)[:s.budget].sort().values for _ in range(64)]).int()
 obs=ready.observe(s,q,k,0,64,64**-.5)
 ready.ensure_cache(s,q);bias=torch.zeros_like(ids,dtype=torch.float32)
 def old():
  output,mass=old_attention(q[:,:2048],k,v,ids,bias,s.specials,64**-.5)
  reference.save_evidence(s,q,k,0,ids,ids,obs,64**-.5)
  return output,mass
 def new():
  output,mass,raw=weighted_attention(q[:,:2048],k,v,ids,bias,s.specials,64**-.5)
  obs['ready_logits']=raw;ready.save_evidence(s,q,k,0,ids,ids,obs,64**-.5)
  del obs['ready_logits'];return output,mass
 result={}
 for name,fn in [('old_producer',old),('ready_producer',new)]:
  times=[];memory=[]
  for i in range(4):
   s.m5c_next={};s.m5c_seed_errors=[];s.m5c_count_errors=[]
   torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();output=fn();torch.cuda.synchronize()
   if i:times.append(time.perf_counter()-start);memory.append((torch.cuda.max_memory_allocated()-base)/2**20)
   del output
  result[name]=dict(median_seconds=statistics.median(times),samples=times,peak_increment_mib=max(memory));print(name,result[name],flush=True)
  if name=='old_producer':refwindow=s.m5c_next[0]
 s.m5c_previous_q=q;s.m5c_previous_k=k;s.m5c_previous_scale=64**-.5
 for name,api in [('old_consumer',reference),('ready_consumer',ready)]:
  times=[];memory=[]
  for i in range(4):
   s.m5c_previous={0:refwindow};s.previous_layer=0;s.ready_written[0]=0;s.m5c_prior_errors=[]
   torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();output=api.build_prior(s,0,ids,obs);torch.cuda.synchronize()
   if i:times.append(time.perf_counter()-start);memory.append((torch.cuda.max_memory_allocated()-base)/2**20)
   del output
  result[name]=dict(median_seconds=statistics.median(times),samples=times,peak_increment_mib=max(memory));print(name,result[name],flush=True)
 result.update(quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='No-GT producer/consumer microbenchmark only',dense_speedup=None,dense_reason='Unchanged Dense not rerun')
 (OUT/'producer_consumer_microbenchmark.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
