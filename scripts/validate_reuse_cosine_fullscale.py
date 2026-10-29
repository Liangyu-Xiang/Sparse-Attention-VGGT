"""Real 128-frame layout and mass-conservation check; not a quality/speed run."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import saf3r.patch.reuse_residual_exploration as implementation
import saf3r.patch.reuse_cosine_distortion as cosine
from scripts import evaluate_reuse_cosine as evaluator
OUT=ROOT/'outputs/reuse_cosine_corrected_fullscale_20261029'
holder={};checks=[];loss_seen=set();transfer_seen=set()
install=implementation.install_reuse_residual_exploration
losses=cosine.cosine_states;transfer=cosine.transfer_weights

def install_checked(*args,**kwargs):
 state=install(*args,**kwargs);holder['state']=state;return state

def losses_checked(pi,z,reps,labels,population,hids,allowed,prior):
 result=losses(pi,z,reps,labels,population,hids,allowed,prior)
 layer=holder['state'].previous_layer+1
 if layer in (2,12,14,23) and layer not in loss_seen:
  loss_seen.add(layer)
  with_history=torch.where((hids[:2]>=0).any(0).any(-1))[0]
  clusters=with_history[:3] if len(with_history)>=3 else torch.tensor([0,len(reps)//2,len(reps)-1],device=z.device)
  members=torch.where(torch.isin(labels,clusters))[0];member_list=members.cpu().tolist();mapping={j:i for i,j in enumerate(member_list)};clist=clusters.cpu().tolist();cmap={j:i for i,j in enumerate(clist)}
  local_labels=torch.tensor([cmap[c] for c in labels[members].cpu().tolist()])
  local_reps=torch.tensor([mapping[j] for j in reps[clusters].cpu().tolist()])
  local_hids=hids[:2,clusters].cpu().clone()
  for g in range(2):
   for c in range(len(clusters)):
    for k in range(local_hids.shape[-1]):
     j=int(local_hids[g,c,k]);local_hids[g,c,k]=mapping[j] if j>=0 else -1
  expected=losses(pi[:2,clusters].cpu(),z[members].cpu(),local_reps,local_labels,population[clusters].cpu(),local_hids,allowed[:2,clusters].cpu(),prior[:2,members].cpu())
  for a,b in zip(result,expected):torch.testing.assert_close(a[:2,clusters].cpu(),b,atol=1e-8,rtol=1e-4)
  checks.append(dict(layer=layer,check='real noncontiguous history layout vs explicit member-level CPU reference',clusters=clist,passed=True))
 return result

def transfer_checked(selected,z,reps,labels,hids,prior,packed):
 weights,masses=transfer(selected,z,reps,labels,hids,prior,packed)
 layer=holder['state'].previous_layer+1
 if layer in (2,12,14,23) and layer not in transfer_seen:
  transfer_seen.add(layer)
  own=prior.gather(1,packed.long().clamp_min(0))
  actual=(own*weights).sum(1)
  expected=prior.masked_fill(selected[:,labels]==0,0).sum(1)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-5)
  checks.append(dict(layer=layer,check='full-window transfer mass conservation for executed supports',maximum_absolute_error=float((actual-expected).abs().max()),passed=True))
 return weights,masses

implementation.install_reuse_residual_exploration=install_checked
cosine.cosine_states=losses_checked;cosine.transfer_weights=transfer_checked
args=SimpleNamespace(output_dir=OUT,method='cosine_distortion',scene='heads',frames=128,warmup=0,repeats=1,capture=False,diagnostic=True,baseline=None,config=ROOT/'configs/evaluation/reuse_cosine_distortion.json')
try:
 evaluator.run(args)
 assert len(checks)==8
 (OUT/'numerical_checks.json').write_text(json.dumps(checks,indent=2))
 old='Explicit two-frame implementation diagnostic, pose/recon_unposed not evaluated'
 new='128-frame numerical diagnostic with explicit CPU comparisons and synchronization; no quality or warmed speed comparison'
 for name in ('quality.json','quality.md','diagnostic.json'):
  p=OUT/name;p.write_text(p.read_text().replace(old,new))
 print('FULL_SCALE_NUMERICAL_CHECKS_PASSED',flush=True)
except BaseException as exc:
 if OUT.exists():(OUT/'status.json').write_text(json.dumps(dict(stage='failed',error=str(exc)),indent=2))
 raise
