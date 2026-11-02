"""Real-forward same-input shadow comparison, outside timed runs and without GT."""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=Path('/tmp/zdrop_m5c_equivalent_20261102');sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import numpy as np
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_residual_exploration as impl,reuse_m5c as ref,reuse_m5c_equivalent as fast
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
OUT=ROOT/'outputs/m5c_equivalent_real128_20261102'

def diff(x,y):
 delta=(x-y).float();return dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/x.float().norm().clamp_min(1e-30)))

def main():
 OUT.mkdir();write_quality(OUT,quality_record({},'Real-input numerical shadow diagnostic only, no pose/recon_unposed GT metrics'))
 save(OUT/'status.json',dict(stage='loading'));torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 folder=ROOT/'outputs/zdrop_shared_rms_20261031/7scenes/heads/rep_combined_l2_direction_adaptive'
 inputs=json.loads((folder/'input.json').read_text());cfg=Dict(json.loads((folder/'config.json').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inputs['image_paths']).cuda()
 state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text())))
 records=[];pending={};build=fast.build_prior;evidence=fast.save_evidence;gate=impl.history_rule.gate
 def chosen(s,start,g):return start==0 or start+g==s.groups
 def build_checked(s,start,ids,obs):
  if not chosen(s,start,len(ids)):return build(s,start,ids,obs)
  shadow=copy.copy(s);shadow.m5c_previous={start:s.m5c_previous[start]};shadow.m5c_prior_errors=[]
  pa,ca=ref.build_prior(shadow,start,ids,obs);pb,cb=build(s,start,ids,obs)
  item=dict(layer=s.previous_layer+1,start=start,prior=diff(pa,pb),pi=diff(ca,cb))
  assert item['prior']['relative_l2']<1e-5 and item['pi']['relative_l2']<1e-5,item
  records.append(item);pending.update(reference=(pa,ca),record=item)
  return pb,cb
 def gate_checked(prior,pi,pop,ids,labels,reps,*a,**kw):
  actual=gate(prior,pi,pop,ids,labels,reps,*a,**kw)
  if pending:
   pa,ca=pending['reference'];expected=gate(pa,ca,pop,ids,labels,reps,*a,**kw)
   mismatches={key:int((value!=actual[key]).sum()) for key,value in expected.items()}
   pending['record']['gate_mismatches']=mismatches;pending.clear()
  return actual
 def evidence_checked(s,q,k,start,ids,actual,obs,scale):
  if not chosen(s,start,len(ids)):return evidence(s,q,k,start,ids,actual,obs,scale)
  shadow=copy.copy(s);shadow.m5c_next={};shadow.m5c_seed_errors=[]
  ref.save_evidence(shadow,q,k,start,ids,actual,obs,scale);evidence(s,q,k,start,ids,actual,obs,scale)
  errors={key:diff(shadow.m5c_next[start][key].cuda(),value) for key,value in s.m5c_next[start].items()}
  assert all(item['relative_l2']<1e-5 for item in errors.values()),errors
  records.append(dict(layer=s.previous_layer+1,start=start,cache=errors))
 fast.build_prior=build_checked;fast.save_evidence=evidence_checked;impl.history_rule.gate=gate_checked
 save(OUT/'status.json',dict(stage='numerical_diagnostic'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize()
 result=dict(complete=True,scope='heads128, first and last processing windows of each layer',records=records,gate_mismatches=sum(sum(r.get('gate_mismatches',{}).values()) for r in records),quality_evaluated=False,dense_speedup=None,dense_reason='No unchanged Dense rerun',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
 save(OUT/'result.json',result);save(OUT/'layers.json',state.records);save(OUT/'status.json',dict(stage='diagnostic_complete'))
 print('COMPLETE real128 windows',len(records),'gate mismatches',result['gate_mismatches'],flush=True)
if __name__=='__main__':main()
