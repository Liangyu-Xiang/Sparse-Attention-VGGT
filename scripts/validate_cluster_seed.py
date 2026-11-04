"""CPU-only numerical invariants; no pose or reconstruction inference."""
import json
from pathlib import Path
import torch
from saf3r.patch.reuse_cluster_seed import prepare_seed
from scripts.quality_metrics import quality_record,quality_markdown

def main():
 out=Path('outputs/cluster_count_validation_20261104');out.mkdir(parents=True,exist_ok=True)
 checks=[]
 for n in [29,101,1036]:
  for f in sorted(set([1,3,n//20,n//10,n//6,n//2])):
   labels=torch.arange(n)%f;reps=torch.arange(f);pop=torch.bincount(labels,minlength=f).float()
   for budget in sorted(set([1,f,max(1,n//5),n])):
    ids,mass,real=prepare_seed(reps,labels,pop,budget)
    assert len(ids)==budget and len(ids.unique())==budget
    assert (mass>0).all() and (ids>=0).all() and (ids<n).all()
    if f<budget:
     actual=torch.zeros(f).scatter_add_(0,labels[ids],mass)
     assert torch.equal(actual,pop) and mass.sum()==n
     assert real.sum()==budget-f and torch.equal(ids[:f],reps)
    else:
     seed=torch.arange(budget)*f//budget
     assert torch.equal(ids,reps[seed]) and torch.equal(mass,pop[seed]) and not real.any()
    checks.append(dict(patches=n,clusters=f,budget=budget,passed=True))
 q=quality_record({},'CPU-only seed invariant test; no model inference or geometry evaluation')
 (out/'result.json').write_text(json.dumps(dict(complete=True,checks=checks,quality=q),indent=2)+'\n')
 (out/'quality.md').write_text(quality_markdown(q));print(len(checks),'seed cases passed')
if __name__=='__main__':main()
