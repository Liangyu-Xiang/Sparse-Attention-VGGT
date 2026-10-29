"""Real 128-frame intermediate equivalence checks, not a quality/speed run."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import saf3r.patch.reuse_residual_exploration as base
from saf3r.patch.reuse_cosine_distortion import transfer_weights
from saf3r.triton import reuse_compact as fast
from saf3r.triton.reuse_cosine_distortion import distortion_losses
from scripts import evaluate_reuse_cosine as evaluator
OUT=ROOT/'outputs/reuse_compact_fullscale_20261029'
holder={};checks=[];seen=set();target=(2,12,14,23)
original_install=base.install_reuse_residual_exploration
original_prepare=fast.prepare_geometry
original_top=fast.top_history;original_loss=fast.losses
original_budget=fast.budget;original_pack=fast.pack;original_transfer=fast.transfer
original_finalize=fast.finalize;original_diagnostics=fast.diagnostics

def install(*args,**kwargs):
 state=original_install(*args,**kwargs);holder['state']=state;return state

def prepare(z,labels,reps,population):
 result=original_prepare(z,labels,reps,population)
 holder.update(z=z,labels=labels,reps=reps,population=population)
 return result

def should_check(name):
 layer=holder['state'].previous_layer+1;key=(layer,name)
 if layer in target and key not in seen:seen.add(key);return layer
 return None

def top(ids,prior,reps,geometry,limit):
 result=original_top(ids,prior,reps,geometry,limit);layer=should_check('top')
 if layer is not None:
  weights=prior.gather(1,ids.long().clamp_min(0)).masked_fill(ids<0,0)
  expected,_,discard=base.top_history(ids,weights,torch.ones((len(ids),1),device=prior.device),torch.zeros(prior.shape[1],device=prior.device,dtype=torch.long),holder['labels'],reps,limit)
  assert torch.equal(result[0].long(),expected)
  torch.testing.assert_close(result[1].float(),discard.float(),atol=0,rtol=0)
  checks.append(dict(layer=layer,check='all-window grouped stable Top-4 exact IDs and discard counts',passed=True))
 return result

def loss(labels,hids,prior,geometry,clusters):
 result=original_loss(labels,hids,prior,geometry,clusters);layer=should_check('loss')
 if layer is not None:
  expected=distortion_losses(holder['z'],labels,holder['reps'],hids,prior,clusters)
  torch.testing.assert_close(result,expected,atol=2e-7,rtol=2e-4)
  checks.append(dict(layer=layer,check='all eleven raw state distortions against original per-token Value kernel',max_absolute_error=float((result-expected).abs().max()),passed=True))
 return result

def budget(errors,costs,upper,cap,iterations):
 result=original_budget(errors,costs,upper,cap,iterations);layer=should_check('budget')
 if layer is not None:
  expected=base.budget_states(errors,costs,cap,iterations,lambda_upper=upper)
  assert torch.equal(result[0],expected[0]);torch.testing.assert_close(result[1],expected[1],atol=0,rtol=0)
  checks.append(dict(layer=layer,check='same-input 16-step discrete dual solver exact selections and counts',passed=True))
 return result

def pack(selected,hids,reps,labels,population,cap):
 result=original_pack(selected,hids,reps,labels,population,cap);layer=should_check('pack')
 if layer is not None:
  expected=base.pack_support(selected,hids,reps,labels,population,cap)
  assert torch.equal(result[0].long(),expected[0])
  for key in result[2]:torch.testing.assert_close(result[2][key].float(),expected[3][key].float(),atol=0,rtol=0)
  checks.append(dict(layer=layer,check='ordered support packing exact IDs and state counts',passed=True))
 return result

def transfer(selected,labels,reps,hids,prior,pi,packed,geometry):
 result=original_transfer(selected,labels,reps,hids,prior,pi,packed,geometry);layer=should_check('transfer')
 if layer is not None:
  expected,_=transfer_weights(selected,holder['z'],reps,labels,hids,prior,packed)
  torch.testing.assert_close(result,expected,atol=5e-5,rtol=5e-5)
  assigned=(prior.gather(1,packed.long().clamp_min(0))*result).sum(1)
  total=prior.masked_fill(selected[:,labels]==0,0).sum(1)
  torch.testing.assert_close(assigned,total,atol=3e-6,rtol=1e-5)
  checks.append(dict(layer=layer,check='analytic/multi-nearest weights and mass conservation against original',max_relative_weight_error=float(((result-expected).abs()/expected.clamp_min(1)).max()),passed=True))
 return result

def finalize(errors,pi,population,hids,allowed,drop):
 layer=should_check('finalize');source=errors.clone() if layer is not None else None
 result=original_finalize(errors,pi,population,hids,allowed,drop)
 if layer is not None:
  from saf3r.patch.reuse_cosine_distortion import _costs_and_mask
  expected,costs=_costs_and_mask(pi,population,hids,allowed,source)
  expected[:,:,0]=drop*pi
  torch.testing.assert_close(result[0],expected,atol=0,rtol=0)
  torch.testing.assert_close(result[1],costs,atol=0,rtol=0)
  torch.testing.assert_close(result[2],expected-expected[:,:,1:2],atol=0,rtol=0)
  checks.append(dict(layer=layer,check='fused valid-state masks, all costs and Rep-relative objective exactly match reference',passed=True))
 return result

def diagnostics(errors,selected,hids,old_ids,labels,reps):
 result=original_diagnostics(errors,selected,hids,old_ids,labels,reps);layer=should_check('diagnostics')
 if layer is not None:
  gains=errors[:,:,1:2]-errors;finite=torch.isfinite(gains)
  expected=dict(positive_gain_sum=gains.masked_fill(~finite,0).clamp_min(0).sum((1,2)),negative_gain_sum=gains.masked_fill(~finite,0).clamp_max(0).sum((1,2)),observed_history=(old_ids>=0).sum(1).float())
  for key,value in expected.items():torch.testing.assert_close(result[key],value,atol=2e-6,rtol=1e-5)
  checks.append(dict(layer=layer,check='fused diagnostic reductions match original without changing state selection',passed=True))
 return result

base.install_reuse_residual_exploration=install
fast.finalize=finalize;fast.diagnostics=diagnostics
fast.prepare_geometry=prepare;fast.top_history=top;fast.losses=loss;fast.budget=budget;fast.pack=pack;fast.transfer=transfer
args=SimpleNamespace(output_dir=OUT,method='rep_compact',scene='heads',frames=128,warmup=0,repeats=1,capture=False,diagnostic=True,baseline=None)
try:
 evaluator.run(args)
 assert len(checks)==28
 (OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2))
 for name in ('quality.json','quality.md','diagnostic.json'):
  p=OUT/name;p.write_text(p.read_text().replace('Explicit two-frame implementation diagnostic, pose/recon_unposed not evaluated','128-frame compact numerical diagnostic with reference-kernel comparisons; no pose/reconstruction quality or paired warmed speed comparison'))
 print('COMPACT_FULLSCALE_CHECKS_PASSED',flush=True)
except BaseException as exc:
 if OUT.exists():
  from scripts.quality_metrics import quality_record
  from scripts.evaluate_attention_first_residual_value import write_quality,save
  write_quality(OUT,quality_record({},'Compact validation failed: '+str(exc)))
  save(OUT/'status.json',dict(stage='failed',error=str(exc)))
 raise
