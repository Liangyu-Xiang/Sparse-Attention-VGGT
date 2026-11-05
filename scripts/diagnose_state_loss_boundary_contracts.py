"""Execution-equivalent states and exact-probability counterexamples; CPU only."""
import gzip
import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_markdown


def main():
    base=ROOT/'outputs/global_state_loss_oracle_20261105';out=base/'loss_definition_contracts';out.mkdir(exist_ok=True)
    quality=quality_record({},'Exact-probability state-loss definition counterexamples and frozen-cluster contract audit; no geometry quality or model timing')
    (out/'quality.json').write_text(json.dumps(quality,indent=2)+'\n');(out/'quality.md').write_text(quality_markdown(quality))
    cases=[]
    for name,p in [('uniform',torch.tensor([1/3,1/3,1/3],dtype=torch.float64)),
                   ('nonuniform',torch.tensor([1/11,9/11,1/11],dtype=torch.float64))]:
        values=torch.tensor([1.,1.,0.],dtype=torch.float64)
        dense=float(p@values)
        weight=torch.tensor([2.,0.,1.],dtype=torch.float64)
        sparse=float((p*weight)@values/(p*weight).sum())
        rep_loss=float((p[:2]*(values[:2]-values[0]).square()).sum())
        history_loss=float(p[1]*values[1].square())
        cases.append(dict(name=name,probabilities=p.tolist(),values=values.tolist(),
            rep_keys=[0],history_keys=[0],rep_execution_weights=[2.],history_execution_weights=[2.],
            protected_keys=[2],state_key_cost=1,dense_output=dense,rep_output=sparse,history_output=sparse,
            actual_squared_error=(sparse-dense)**2,legacy_rep_loss=rep_loss,legacy_history_loss=history_loss,
            conclusion='Identical support, weights and cost produce different legacy losses; nonuniform case also has zero Rep proxy loss with nonzero true output error'))
    real={}
    for scene in ('pipes','playground','courtyard'):
        total=singleton=different=0;max_gap=0.
        with gzip.open(base/(scene+'_oracle')/'clusters.jsonl.gz','rt') as stream:
            for line in stream:
                r=json.loads(line);total+=1
                if r['eligible_history_keys']==[r['rep']]:
                    singleton+=1;rep,hist=r['legacy_losses'][1:3]
                    gap=abs(rep-hist)
                    if gap>1e-8:different+=1
                    max_gap=max(max_gap,gap)
        real[scene]=dict(cluster_cases=total,history_singleton_rep=singleton,
                        same_execution_but_different_proxy_losses=different,max_loss_difference=max_gap)
    (out/'result.json').write_text(json.dumps(dict(quality=quality,counterexamples=cases,real_frozen_cluster_contracts=real),indent=2)+'\n')
    print(json.dumps(dict(counterexamples=cases,real_frozen_cluster_contracts=real),indent=2))


if __name__=='__main__':main()
