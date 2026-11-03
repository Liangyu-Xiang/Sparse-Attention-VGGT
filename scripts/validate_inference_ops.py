import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch import zdrop_history_states as old,reuse_inference_ops as new

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);results=[]
    for g,n,f,edges in [(3,173,17,70),(8,8192,1024,1638)]:
        labels=torch.arange(n,device='cuda')%f;reps=torch.arange(f,device='cuda');pop=torch.bincount(labels,minlength=f).float()
        prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True)
        pi=torch.zeros(g,f,device='cuda').scatter_add_(1,labels[None].expand(g,-1),prior)
        ids=torch.stack([torch.randperm(n,device='cuda')[:edges] for _ in range(g)]).int();ids[0,-3:]=-1
        c=old.gate(prior,pi,pop,ids,labels,reps);d=new.gate(prior,pi,pop,ids,labels,reps)
        assert all(torch.equal(c[k],d[k]) for k in c)
        states=torch.tensor([0,1,5,10],device='cuda');selected=states[torch.randint(4,(g,f),device='cuda')]
        selected.masked_fill_((selected==1)&c['remove_rep'],0);selected.masked_fill_((selected==5)&(c['eligible_count']==0),0)
        per_cluster=torch.full((g,f),n,device='cuda',dtype=torch.long)
        js=torch.arange(n,device='cuda')[None].expand(g,-1)
        per_cluster.scatter_reduce_(1,labels[None].expand(g,-1),js.masked_fill(~c['qualified'],n),reduce='amin')
        owner=per_cluster[:,labels].masked_fill(per_cluster[:,labels]==n,-1)
        owner=torch.where(c['qualified'],js,owner).int();c['owners']=owner;d['owners']=owner
        x,real,r=old.pack(selected,None,reps,labels,pop,n,c);y,real2,r2=new.pack(selected,None,reps,labels,pop,n,d)
        assert torch.equal(x,y) and torch.equal(real,real2) and all(torch.equal(r[k],r2[k]) for k in r)
        before=old.transfer(selected,labels,reps,None,prior,pi,x,None,c);after=new.transfer(selected,labels,reps,None,prior,pi,y,None,d)
        rel=float((before-after).norm()/before.norm());assert rel<1e-5 and not d['pack_error'].any() and (d['mass_error']<=1e-5).all()
        results.append(dict(groups=g,tokens=n,clusters=f,gate_bitwise_equal=True,packed_ids_bitwise_equal=True,counts_bitwise_equal=True,weight_relative_l2=rel,mass_error=float(d['mass_error'].max())))
    result=dict(complete=True,checks=results,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Execution-operator correctness diagnostic',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
