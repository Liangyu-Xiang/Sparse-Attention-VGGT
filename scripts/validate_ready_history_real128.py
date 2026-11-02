"""Sample real producer and consumer arithmetic, without GT or reference model."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=Path('/tmp/zdrop_ready_history_v5_20261102');sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import numpy as np
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_residual_exploration as impl,reuse_ready_history as ready,reuse_m5c_pipeline as reference
from saf3r.triton.history_probe_logits import selected_probe_logits
from saf3r.triton.reuse_ready_history import predict,decode
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
OUT=ROOT/'outputs/ready_history_v5_real128_20261102'

def diff(a,b):
 delta=(a-b.float()).float();return dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/a.norm().clamp_min(1e-30)))

def main():
 OUT.mkdir();write_quality(OUT,quality_record({},'Real-input cache-quantization and producer/consumer diagnostic only; no pose/recon_unposed GT metrics'))
 save(OUT/'status.json',dict(stage='loading'));torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
 folder=ROOT/'outputs/zdrop_shared_rms_20261031/7scenes/heads/rep_combined_l2_direction_adaptive'
 inp=json.loads((folder/'input.json').read_text());cfg=Dict(json.loads((folder/'config.json').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inp['image_paths']).cuda();state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text())))
 saved={};records=[];pending={};build=ready.build_prior;evidence=ready.save_evidence;gate=impl.history_rule.gate
 def chosen(s,start,g):return start==0 or start+g==s.groups
 def build_checked(s,start,ids,obs):
  result=build(s,start,ids,obs)
  if not chosen(s,start,len(ids)):return result
  old=saved.pop(start);current=obs['coarse_grid_probability'][:1];ratio=(current+1e-30)/(old['coarse']+1e-30);confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
  base=current*(confidence*old['uniform']+(1-confidence)/s.grid_sizes)
  prior=ready.mean_probes(base,obs['weights'][:1])[:,s.grid].contiguous()
  predict(old['conditional'],ids[:1],s.grid,s.grid_sizes,current,old['coarse'],obs['weights'][:1],prior,1e-30)
  pi=torch.zeros((1,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None],prior)
  item=dict(layer=s.previous_layer+1,start=start,prior=diff(prior,result[0][:1]),pi=diff(pi,result[1][:1]))
  assert item['prior']['relative_l2']<.001 and item['pi']['relative_l2']<.001,item
  records.append(item);pending.update(prior=prior,pi=pi,record=item)
  return result
 def gate_checked(prior,pi,pop,ids,labels,reps,*a,**kw):
  result=gate(prior,pi,pop,ids,labels,reps,*a,**kw)
  if pending:
   expected=gate(pending['prior'],pending['pi'],pop,ids[:1],labels,reps,*a,**kw)
   pending['record']['gate_mismatches']={k:int((v!=result[k][:1]).sum()) for k,v in expected.items()};pending.clear()
  return result
 def evidence_checked(s,q,k,start,ids,actual,obs,scale):
  if not chosen(s,start,len(ids)):return evidence(s,q,k,start,ids,actual,obs,scale)
  subset={key:value[:1] for key,value in obs.items() if key!='ready_logits'}
  shadow=copy.copy(s);shadow.m5c_next={};shadow.m5c_seed_errors=[];shadow.m5c_count_errors=[];shadow.m5c_probe_qk=0
  reference.save_evidence(shadow,q,k,start,ids[:1],actual[:1],subset,scale)
  old=shadow.m5c_next[start];raw=selected_probe_logits(q,k,actual[:1],subset['positions'],scale);regions=s.grid[ids[:1].long().clamp_min(0)]
  conditional=(raw-old['history_dense_log_normalizer']).exp()/old['previous_corrected_grid_probability'].gather(-1,regions[:,None,None,:].expand_as(raw)).clamp_min(1e-30)
  conditional=conditional.masked_fill(ids[:1,None,None,:]<0,0)
  emitted=obs['ready_logits'][:1];valid=ids[:1,None,None,:]>=0
  logit_error=diff(raw.masked_fill(~valid,0),emitted.masked_fill(~valid,0));assert logit_error['relative_l2']<1e-5,logit_error
  evidence(s,q,k,start,ids,actual,obs,scale)
  actual_cache=decode(ready.peek_cache(s,start,1),ids.shape[1]);error=diff(conditional,actual_cache)
  assert error['relative_l2']<.001,error
  records.append(dict(layer=s.previous_layer+1,start=start,emitted_logits=logit_error,conditional=error,half_underflow_count=int(((conditional>0)&(actual_cache==0)).sum())))
  saved[start]=dict(conditional=conditional,coarse=old['previous_coarse_grid_probability'].contiguous(),uniform=old['unobserved_uniform_probability']/old['previous_corrected_grid_probability'].clamp_min(1e-30))
 ready.build_prior=build_checked;ready.save_evidence=evidence_checked;impl.history_rule.gate=gate_checked
 save(OUT/'status.json',dict(stage='numerical_diagnostic'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();assert not saved and state.ready_edges is None
 result=dict(complete=True,records=records,scope='heads128, one query group in first/last window of each layer',gate_mismatches=sum(sum(r.get('gate_mismatches',{}).values()) for r in records),qualified_mismatches=sum(r.get('gate_mismatches',{}).get('qualified',0) for r in records),quality_evaluated=False,dense_speedup=None,dense_reason='Unchanged Dense not rerun')
 save(OUT/'result.json',result);save(OUT/'layers.json',state.records);save(OUT/'status.json',dict(stage='diagnostic_complete'));print('COMPLETE',len(records),'qualified mismatches',result['qualified_mismatches'],flush=True)
if __name__=='__main__':main()
