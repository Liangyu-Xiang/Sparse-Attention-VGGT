"""Same-input shadow v8 estimator and exact current-QK probability diagnostics."""
import argparse,json,sys,time,types
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();root=a.source;sys.path[:0]=[str(root),str(root/'scripts'),str(root/'evaluation')];a.output_dir.mkdir(parents=True,exist_ok=False)
from benchmark_beta_gridmass import build
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_sparse_beta_history as beta,reuse_grid_local_history as old,reuse_headmean_history as mean,reuse_ready_history as cache
from saf3r.triton import reuse_serial_sparseprob as backend
from saf3r.triton.reuse_ready_history import decode
model,state=build(root/'configs/evaluation/reuse_beta_gridmass.json');torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();rows=[];shadow=None;current={};old_ensure=beta.ensure_cache;old_observe=beta.observe;old_build=beta.build_prior;old_save=beta.save_evidence;old_exec=backend.weighted_attention
from saf3r.patch import reuse_probe_cache as probe_cache
old_observe=probe_cache.observe

def ensure(s,q):
 global shadow
 old_ensure(s,q);shadow=types.SimpleNamespace(**s.__dict__);shadow.groups=4;shadow.ready_written={};shadow.m5c_seed_errors=[];shadow.m5c_count_errors=[];shadow.m5c_prior_errors=[];shadow.ready_edges=None;shadow.ready_host=None;mean.ensure_cache(shadow,q)

def observe(s,q,k,start,g,scale):
 obs=old_observe(s,q,k,start,g,scale)
 if start==0:current.update(q=q,k=k,scale=scale,obs=obs)
 return obs

def metric(est,truth):
 x=est/est.sum(-1,keepdim=True).clamp_min(1e-30);y=truth/truth.sum(-1,keepdim=True).clamp_min(1e-30)
 return dict(l1_mean=float((x-y).abs().sum(-1).mean()),kl_truth_est_mean=float((y*(y.clamp_min(1e-30).log()-x.clamp_min(1e-30).log())).sum(-1).mean()),entropy_mean=float(-(x*x.clamp_min(1e-30).log()).sum(-1).mean()))

def build_prior(s,start,ids,obs):
 result=old_build(s,start,ids,obs)
 if start==0:
  count=min(4,len(ids));shadow.previous_layer=s.previous_layer;shadow.fine=s.fine
  small={key:(v[:count] if torch.is_tensor(v) and len(v)==len(ids) else v) for key,v in obs.items()}
  reference,_=mean.build_prior(shadow,0,ids[:count],small)
  q=current['q'];k=current['k'];pos=obs['positions'][:count];logit=torch.einsum('hgpd,hnd->ghpn',q[:,pos].float(),k.float())*current['scale'];prob=logit.softmax(-1)
  truth=(prob[:,:,:,s.patches]*obs['weights'][:count,None,:,None]).sum(2).mean(1)
  prior=result[0][:count];safe=ids[:count].long().clamp_min(0);valid=ids[:count]>=0
  row=dict(layer=s.previous_layer+1,sampling='first4 groups of first64-group window;4 original probes/head',beta_prior=metric(prior,truth),shadow_v8_prior=metric(reference,truth),prior_l1_beta_v8=float((prior-reference).abs().sum(-1).mean()),dense_patch_mass=float(truth.sum(-1).mean()),beta_previous_support_mass=float(prior.gather(1,safe).masked_fill(~valid,0).sum(-1).mean()),shadow_v8_previous_support_mass=float(reference.gather(1,safe).masked_fill(~valid,0).sum(-1).mean()),dense_previous_support_mass=float(truth.gather(1,safe).masked_fill(~valid,0).sum(-1).mean()))
  rows.append(row);current['truth']=truth
 return result

def execute(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs):
 result=old_exec(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs)
 # Only the first window is sampled. All alternate computations are diagnostics;
 # they do not feed into the model trajectory or its history.
 if kwargs.get('evidence_layout') is current.get('obs') or (indices.shape[0]>0 and q.data_ptr()==current.get('q',q).data_ptr()):
  layer=state.previous_layer+1
  if current.get('checked_layer')!=layer:
   current['checked_layer']=layer;count=min(4,len(indices));positions=(torch.arange(count,device=q.device)[:,None]*32+torch.tensor([4,12,20,28],device=q.device)[None]).clamp_max(q.shape[1]-1)
   errors=[];density=[];support_mass=[]
   for group in range(count):
    valid=indices[group]>=0;keys=torch.cat([specials,indices[group,valid].long()]);b=torch.cat([torch.zeros(len(specials),device=q.device),bias[group,valid]])
    pq=q[:,positions[group]].float();logits=pq@k[:,keys].float().transpose(-1,-2)*scale+b;ref=logits.softmax(-1)@v[:,keys].float();act=result[0][:,positions[group]].float();errors.append(float((act-ref).norm()/ref.norm().clamp_min(1e-30)))
    dense=(pq@k.float().transpose(-1,-2)*scale).softmax(-1);support_mass.append(float(dense[:,:,keys].sum(-1).mean()))
   entry=next((x for x in rows if x['layer']==layer),None)
   if entry is None:entry=dict(layer=layer);rows.append(entry)
   entry.update(sparse_core_vs_same_support_fp32_relative_l2_mean=sum(errors)/len(errors),current_support_true_dense_mass_mean=sum(support_mass)/len(support_mass))
 return result

def save(s,q,k,start,ids,actual,obs,scale):
 old_save(s,q,k,start,ids,actual,obs,scale)
 if start==0:
  count=min(4,len(ids));shadow.previous_layer=s.previous_layer;shadow.fine=s.fine
  small={key:(v[:count] if torch.is_tensor(v) and len(v)==len(ids) else v) for key,v in obs.items()}
  old.save_evidence(shadow,q,k,0,ids[:count],actual[:count],small,scale)
  sparse=decode(cache.peek_cache(s,0,count),s.budget)[:,0];conditional=decode(cache.peek_cache(shadow,0,count),s.budget)[:,0];valid=ids[:count]>=0
  entry=next((x for x in rows if x['layer']==s.previous_layer+1),None)
  if entry is None:entry=dict(layer=s.previous_layer+1);rows.append(entry)
  entry.update(sparse_cache_zero_fraction=float(((sparse==0)&valid[:,None]).sum()/valid.sum()/4),v8_conditional_cache_zero_fraction=float(((conditional==0)&valid[:,None]).sum()/valid.sum()/4),sparse_cache_selected_mass=float(sparse.sum(-1).mean()),compensation_logbias_mean=float(obs['sparse_log_weights'][:count][valid].mean()),compensation_logbias_max=float(obs['sparse_log_weights'][:count][valid].max()))
  (a.output_dir/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
beta.ensure_cache=ensure;beta.observe=observe;probe_cache.observe=observe;beta.build_prior=build_prior;beta.save_evidence=save;backend.weighted_attention=execute
begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize()
result=dict(complete=True,sampling='heads128 beta trajectory;first4 groups/window0,all24 layers;original4probes/head',rows=rows,diagnostic_wall_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Same-input estimator and kernel diagnostics; geometry quality is in frozen18-scene experiment; diagnostic wall is not benchmark',dense_speedup=None,dense_reason='This diagnostic includes extra oracle computations, no speedup claimed')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
