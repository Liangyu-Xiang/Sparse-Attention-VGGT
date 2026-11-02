"""Ready-history arithmetic, emitted-logit, quantization and microbenchmark checks."""
import copy,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=Path('/tmp/zdrop_ready_history_v5_20261102');sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import torch
from types import SimpleNamespace as NS
from saf3r.patch import reuse_ready_history as ready,reuse_m5c_pipeline as reference
from saf3r.triton.reuse_explore_av_ready import weighted_attention
from saf3r.triton.reuse_explore_av import weighted_attention as old_attention
from saf3r.triton.history_probe_logits import selected_probe_logits
OUT=ROOT/'outputs/ready_history_20261102';OUT.mkdir(exist_ok=True)

def difference(a,b):
 d=a.float()-b.float();return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.float().norm().clamp_min(1e-30)))

def case(g=5,h=3,n=129,p=113,r=7,a=67,d=32,dtype=torch.bfloat16,cache_dtype=torch.float16):
 grid=torch.arange(p,device='cuda')%r;f=min(17,p)
 s=NS(n=n,groups=g,budget=a,regions=r,grid=grid,grid_sizes=torch.bincount(grid,minlength=r).float(),patches=torch.arange(p,device='cuda'),specials=torch.arange(p,n,device='cuda'),fine=NS(base_count=f,labels=torch.arange(p,device='cuda')%f),m5c_next={},m5c_previous={},m5c_count_errors=[],m5c_seed_errors=[],m5c_prior_errors=[],m5c_probe_qk=0,m5c_reconstruction_qk=0,previous_layer=-1,ready_written={},ready_cache_dtype=cache_dtype)
 q=torch.randn(h,n,d,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
 s.history_pooled_fp32=torch.randn(h,r,d,device='cuda')*.2
 ids=torch.stack([torch.randperm(p,device='cuda')[:a] for _ in range(g)]).int();ids[0,-3:]=-1;ids[1].fill_(-1)
 obs=ready.observe(s,q,k,0,g,d**-.5)
 return s,q,k,v,ids,obs

def check(seed,dtype,storage,a):
 torch.manual_seed(seed);s,q,k,v,ids,obs=case(dtype=dtype,cache_dtype=storage,a=a);bias=torch.randn_like(ids,dtype=torch.float32)*.1
 out,mass,raw=weighted_attention(q,k,v,ids,bias,s.specials,32**-.5)
 target=selected_probe_logits(q,k,ids,obs['positions'],32**-.5)
 finite=ids[:,None,None]>=0;rawdiff=difference(target.masked_fill(~finite,0),raw.masked_fill(~finite,0))
 assert torch.allclose(target,raw,rtol=1e-5,atol=1e-5),rawdiff
 if dtype==torch.bfloat16:
  old,oldmass=old_attention(q,k,v,ids,bias,s.specials,32**-.5)
  assert difference(old,out)['relative_l2']<.005
  assert torch.allclose(oldmass,mass,atol=1e-6,rtol=1e-4)
 oldstate=copy.copy(s);oldstate.m5c_next={};oldstate.m5c_seed_errors=[];oldstate.m5c_count_errors=[]
 reference.save_evidence(oldstate,q,k,0,ids,ids,obs,32**-.5)
 old=oldstate.m5c_next[0];expected=(target-old['history_dense_log_normalizer']).exp()/old['previous_corrected_grid_probability'].gather(-1,s.grid[ids.long().clamp_min(0)][:,None,None,:].expand_as(target)).clamp_min(1e-30)
 expected=expected.masked_fill(~finite,0)
 ready.ensure_cache(s,q);obs['ready_logits']=raw;ready.save_evidence(s,q,k,0,ids,ids,obs,32**-.5)
 from saf3r.triton.reuse_ready_history import decode
 cached=decode(s.ready_edges,ids.shape[1]);cache_diff=difference(expected,cached)
 tolerance=1e-5 if storage==torch.float32 else .001
 assert cache_diff['relative_l2']<tolerance,cache_diff

 oldstate.m5c_previous={0:old};oldstate.m5c_previous_q=q;oldstate.m5c_previous_k=k;oldstate.m5c_previous_scale=32**-.5
 q2=q*.9+.03;obs2=ready.observe(s,q2,k,0,len(ids),32**-.5)
 pa,ca=reference.build_prior(oldstate,0,ids,obs2);s.previous_layer=0;pb,cb=ready.build_prior(s,0,ids,obs2)
 pd=difference(pa,pb);cd=difference(ca,cb)
 assert pd['relative_l2']<tolerance and cd['relative_l2']<tolerance,(pd,cd)
 # A second write reuses the same slots after consumption.
 pointer=s.ready_edges.data_ptr();s.previous_layer=0;ready.save_evidence(s,q,k,0,ids,ids,obs,32**-.5)
 assert pointer==s.ready_edges.data_ptr() and s.ready_written[0]==1
 assert float(torch.stack(s.m5c_seed_errors).max())<1e-5
 ready.release(s);assert s.ready_edges is None
 return dict(seed=seed,q_dtype=str(dtype),cache_dtype=str(storage),support=a,emitted_logits=rawdiff,conditional=cache_diff,prior=pd,pi=cd)

def check_spill():
 torch.manual_seed(7);s,q,k,v,ids,_=case(8,3,256,240,7,67,32)
 s.config=NS(window_groups=2);s.ready_gpu_limit_bytes=2*3*4*((67*14+7)//8)
 ready.ensure_cache(s,q);assert s.ready_resident_groups==2 and s.ready_host.is_pinned()
 old_windows={};previous_q=None;errors=[]
 for layer in range(3):
  current=q*(1-.05*layer)+.01*layer;new_windows={}
  for start in range(0,8,2):
   part=ids[start:start+2];obs=ready.observe(s,current,k,start,2,32**-.5)
   shadow=copy.copy(s);shadow.m5c_count_errors=[];shadow.m5c_seed_errors=[];shadow.m5c_prior_errors=[];shadow.m5c_probe_qk=0;shadow.m5c_reconstruction_qk=0
   if layer:
    shadow.m5c_previous={start:old_windows[start]};shadow.m5c_previous_q=previous_q;shadow.m5c_previous_k=k;shadow.m5c_previous_scale=32**-.5
    expected,_=reference.build_prior(shadow,start,part,obs);actual,_=ready.build_prior(s,start,part,obs)
    errors.append((expected-actual).norm()/expected.norm().clamp_min(1e-30))
   shadow.m5c_next={};reference.save_evidence(shadow,current,k,start,part,part,obs,32**-.5);new_windows[start]=shadow.m5c_next[start]
   obs['ready_logits']=selected_probe_logits(current,k,part,obs['positions'],32**-.5)
   ready.save_evidence(s,current,k,start,part,part,obs,32**-.5)
  old_windows=new_windows;previous_q=current;s.previous_layer=layer
 torch.cuda.synchronize();maximum=float(torch.stack(errors).max());assert maximum<.001,maximum
 ready.release(s)
 return dict(layers=3,groups=8,resident_groups=2,spilled_groups=6,prior_relative_l2=maximum)


def benchmark():
 torch.manual_seed(50);s,q,k,v,ids,obs=case(64,16,2048,2048-16,128,512,64)
 # Micro scope uses 64 groups x 32 queries. Real full-sequence cache is measured separately.
 ready.ensure_cache(s,q);bias=torch.zeros_like(ids,dtype=torch.float32)
 data={}
 for name,fn in [('attention_reference',lambda:old_attention(q,k,v,ids,bias,s.specials,64**-.5)),('attention_emit',lambda:weighted_attention(q,k,v,ids,bias,s.specials,64**-.5))]:
  times=[]
  for i in range(4):
   torch.cuda.synchronize();t=time.perf_counter();x=fn();torch.cuda.synchronize()
   if i:times.append(time.perf_counter()-t)
   del x
  data[name]=dict(median_seconds=statistics.median(times),samples=times)
 return data

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
 from saf3r.triton.reuse_ready_history import pack_half,decode,logsumexp
 values=torch.arange(0x3c01,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.float16).reshape(1,1,1,-1)
 assert torch.equal(values.float(),decode(pack_half(values),values.shape[-1]))
 for a in [1,17,257,26009]:
  x=torch.randn(7,a,device='cuda')*8;x[0].fill_(-torch.inf)
  assert torch.allclose(x.logsumexp(-1),logsumexp(x),atol=1e-5,rtol=1e-6)
 print('PASS ALL POSITIVE FP16 CODES BITWISE PACK/UNPACK AND FOUR LSE LENGTHS',flush=True)
 for dtype in [torch.float32,torch.bfloat16]:
  for storage in [torch.float32,torch.float16]:
   for a in [67,113]:checks.append(check(0,dtype,storage,a))
 result=dict(spill_validation=check_spill(),complete=True,codec_all_positive_fp16_codes_bitwise=True,lse_lengths=[1,17,257,26009],checks=checks,microbenchmark=benchmark(),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},reason='Numerical/cache/kernel diagnostic only; no GT pose/recon_unposed evaluation',dense_speedup=None,dense_reason='Unchanged Dense not rerun')
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS READY HISTORY EMISSION/CALIBRATION/CACHE/PRIOR',flush=True)
if __name__=='__main__':main()
