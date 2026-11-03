"""Numerical contracts for engineering execution and NEW modeling primitives."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_light_attention import weighted_attention
from saf3r.triton import reuse_light_states as rule
from saf3r.triton.reuse_global_history import store

def rel(a,b):return float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1.e-30))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.manual_seed(3);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
 for dtype in [torch.float32,torch.bfloat16]:
  h,r,n,d,g,cap=3,79,101,64,3,37
  q=torch.randn(h,r,d,device='cuda',dtype=dtype);k=torch.randn(h,n,d,device='cuda',dtype=dtype);v=torch.randn_like(k)
  ids=torch.stack([torch.randperm(83,device='cuda')[:cap] for _ in range(g)]).int();ids[1]=-1;ids[2,-5:]=-1;bias=torch.rand(g,cap,device='cuda')*2;special=torch.arange(83,n,device='cuda')
  reference=torch.empty_like(q)
  for group in range(g):
   valid=ids[group]>=0;keys=torch.cat([special,ids[group,valid].long()]);b=torch.cat([torch.zeros(len(special),device='cuda'),bias[group,valid]])
   z=q[:,group*32:min((group+1)*32,r)].float()@k[:,keys].float().transpose(-1,-2)/d**.5+b
   reference[:,group*32:min((group+1)*32,r)]=(z.softmax(-1)@v[:,keys].float()).to(dtype)
  for parts,probes in [(1,4),(4,4),(8,4),(4,1)]:
   out,ev=weighted_attention(q,k,v,ids,bias,special,d**-.5,parts=parts,probe_count=probes)
   error=rel(reference,out);assert error<(1e-5 if dtype==torch.float32 else .005),error
   for group in range(g):
    positions=((torch.arange(probes,device='cuda')*8+4 if probes==4 else torch.tensor([16],device='cuda'))+group*32).clamp_max(r-1)
    z=q[:,positions].float()@k[:,ids[group].clamp_min(0).long()].float().transpose(-1,-2)/d**.5
    z=z.masked_fill(ids[group][None,None]<0,-torch.inf)
    finite=torch.isfinite(z);assert torch.equal(finite,torch.isfinite(ev['raw'][group]))
    assert torch.allclose(z[finite],ev['raw'][group][finite],atol=1e-5,rtol=1e-5)
    expected=z.logsumexp(-1);assert torch.allclose(expected,ev['selected_lse'][group],atol=1e-5,rtol=1e-5)
   z=torch.randn(g,h,probes,device='cuda');weights=torch.full((g,probes),1/probes,device='cuda');pp=store(ev['raw'],z,weights,ids)
   expected=(ev['raw']-z[...,None]).exp().mean((1,2)).masked_fill(ids<0,0.)
   assert rel(expected,pp)<1e-5
   checks.append(dict(kind='attention',dtype=str(dtype),parts=parts,relative_l2=error))
 # Non-contiguous Exact mask, irregular clusters and nonuniform historical probabilities.
 g,n,f,d,cap=3,83,13,64,31;z=torch.randn(n,d,device='cuda');labels=torch.arange(n,device='cuda')%f;reps=torch.arange(f,device='cuda');pop=torch.bincount(labels,minlength=f).float();geo=rule.prepare(z,labels,pop)
 ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();ids[1,-5:]=-1
 bg=torch.rand(g,device='cuda')*.001+.001;hp=torch.rand(g,cap,device='cuda')*.01;hp=hp.masked_fill(ids<0,0.)
 prior=bg[:,None].expand(g,n).clone()
 for group in range(g):prior[group,ids[group,ids[group]>=0].long()]=hp[group,ids[group]>=0]
 o0=prior@z;observed=rule.moment(ids,hp,bg,z);assert rel(o0,observed)<1e-5
 allowed=(torch.rand(f,g,device='cuda')>.3).T
 errors,costs,qualified=rule.states(ids,hp,bg,z,reps,labels,pop,allowed,o0,geo)
 expected=torch.empty_like(errors);ec=torch.empty_like(costs);qual=torch.zeros_like(qualified)
 for group in range(g):
  observed=torch.zeros(n,device='cuda',dtype=torch.bool);observed[ids[group,ids[group]>=0].long()]=True
  for c in range(f):
   members=torch.where(labels==c)[0];p=prior[group,members];pi=p.sum();target=(p[:,None]*z[members]).sum(0);rep=reps[c]
   take=observed[members]&((p>pi/pop[c])|(members==rep));qual[group,members]=take
   hm=p[take].sum();hv=(p[take,None]*z[members[take]]).sum(0)
   residuals=[-target+pi*o0[group],pop[c]*prior[group,rep]*z[rep]-target-o0[group]*(pop[c]*prior[group,rep]-pi),hv-target-o0[group]*(hm-pi),torch.zeros(d,device='cuda')]
   e=torch.stack([x.square().sum() for x in residuals]);e[1]=torch.inf if observed[rep] else e[1];e[2]=torch.inf if not take.any() else e[2];e[3]=torch.inf if not allowed[group,c] else 0.
   expected[group,c]=e;ec[group,c]=torch.tensor([0,1,int(take.sum()),int(pop[c])],device='cuda')
 finite=torch.isfinite(expected);assert torch.equal(torch.isfinite(errors),finite);assert rel(expected[finite],errors[finite])<1e-5
 assert torch.equal(costs,ec) and torch.equal(qualified,qual)
 selected=errors.argmin(-1);packed,weights,counts=rule.pack(selected,costs,reps,pop,qualified,geo,n)
 reference_pack=rule.pack_reference(selected,costs,reps,pop,qualified,geo,n)
 assert all(torch.equal(x,y) for x,y in zip((packed,weights,counts),reference_pack))
 for group in range(g):
  keys=packed[group,packed[group]>=0];assert len(keys)==int(counts[group])==len(keys.unique())
  expected_keys=[]
  for c in range(f):
   state=int(selected[group,c]);members=torch.where(labels==c)[0]
   if state==1:expected_keys.extend([int(reps[c])])
   elif state==2:expected_keys.extend(members[qualified[group,members]].tolist())
   elif state==3:expected_keys.extend(members.tolist())
  assert set(keys.tolist())==set(expected_keys)
 checks.append(dict(kind='contribution_states',relative_l2=rel(expected[finite],errors[finite]),qualification_equal=True,costs_equal=True,pack_unique=True))
 result=dict(complete=True,checks=checks,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Small-kernel numerical validation only',dense_speedup=None,dense_reason='No Dense model rerun authorized')
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
