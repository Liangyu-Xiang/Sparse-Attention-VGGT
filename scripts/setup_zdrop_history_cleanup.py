"""Create an isolated implementation-only cleanup fork of the active method."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/tmp/zdrop_history_states_20261102');FAST=Path('/tmp/zdrop_history_cleanup_20261102')

def main():
    m=json.loads((ROOT/'outputs/zdrop_history_states_eth3d_20261102/manifest.json').read_text())
    assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in m['source_sha256'].items())
    if FAST.exists():raise ValueError('Fresh cleanup fork required')
    shutil.copytree(BASE,FAST,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['saf3r/patch/zdrop_history_cleanup.py','saf3r/triton/zdrop_history_cleanup.py']:
        shutil.copy2(ROOT/name,FAST/name)
    p=FAST/'saf3r/patch/reuse_residual_exploration.py';s=p.read_text()
    s=s.replace('from . import zdrop_history_states as history_rule','from . import zdrop_history_cleanup as history_rule')
    # The cluster means feed only the inactive output_proxy branch.
    a="        means=torch.zeros((f,features.shape[-1]),device=k.device).index_add_(0,self.fine.labels,features)\n        means/=population[:,None]"
    assert a in s;s=s.replace(a,'        means=None  # Unread in this compact squared-L2 method.')
    a="        pair=self.fine.labels*r+self.grid\n        unique,count=torch.unique(pair,sorted=True,return_counts=True)\n        overlap=torch.sparse_coo_tensor(torch.stack((unique//r,unique%r)),\n            count.float()/self.grid_sizes[unique%r],size=(f,r),device=k.device).coalesce()"
    assert a in s;s=s.replace(a,"        overlap=None\n        if self.previous_layer>=0:\n"+'\n'.join('    '+line for line in a.splitlines()))
    s=s.replace("        q,k,v=prepare_qkv(o,x,pos);", "        if c.history_mode!='dense_m5c' or not c.state_l2 or not c.direction_drop or not c.compact or c.query_shift or c.hierarchical:raise RuntimeError('Cleanup supports only the frozen active method')\n        q,k,v=prepare_qkv(o,x,pos);")
    a="                zvalues=(features.float()/features.float().square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous() if c.state_l2 else normalized_values(features)"
    assert a in s;s=s.replace(a,"                zvalues=(features.float()/features.float().square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous() if self.layer else None")
    s=s.replace('if c.compact and q.is_cuda:', 'if self.layer and c.compact and q.is_cuda:')
    s=s.replace('if (c.state_l2 or c.direction_drop) and geometry is None:', 'if self.layer and (c.state_l2 or c.direction_drop) and geometry is None:')
    s=s.replace("                    pi=torch.sparse.mm(overlap,probability.T).T.contiguous()\n",'')
    s=s.replace('allowed=torch.zeros_like(pi,dtype=torch.bool).scatter_(1,candidate,True)','allowed=torch.zeros_like(residual,dtype=torch.bool).scatter_(1,candidate,True)')
    s=s.replace("                    old_weights=None if s.history_weights is None else s.history_weights[start:start+g]\n",'')
    s=s.replace("                        old_weights=prior.gather(1,old_ids.long().clamp_min(0)).masked_fill(old_ids<0,0)\n                        history_beta=torch.ones_like(beta)\n",'')
    a="                    if geometry is not None:\n                        hids,discarded=compact.top_history(old_ids,prior,s.fine.reps,geometry,c.history_per_cluster)\n                    else:hids,hmass,discarded=top_history(old_ids,old_weights,history_beta,s.grid,s.fine.labels,s.fine.reps,c.history_per_cluster)"
    assert a in s;s=s.replace(a,"                    hids=None;discarded=torch.zeros(g,device=x.device)")
    a="                        selected,count=compact.budget(objective,costs,errors[:,:,0].amax(1)+1e-12,s.budget,c.dual_iterations)"
    assert a in s;s=s.replace(a,a+"\n                        selected=history_rule.decode(selected)")
    s=s.replace("weights.append((evidence*real).half())",'pass  # This legacy half-precision history bank is unread by D2+M5c.')
    s=s.replace('s.history_weights=torch.cat(weights)','s.history_weights=None')
    s=s.replace('s.history_ids.numel()*4+s.history_weights.numel()*2','s.history_ids.numel()*4')
    a=s.index('        if c.rep_gain:\n            # Layer 0 has no candidate allocation;');b=s.index('        s.previous_grid=',a)
    s=s[:a]+'''        if self.layer==0:
            for record in stats:
                for state_id in range(11):
                    record[f'state_{state_id}_clusters']=record['counts'] if state_id==1 else torch.zeros_like(record['counts'])
'''+s[b:]
    s=s.replace('history=[];weights=[];','history=[];')
    p.write_text(s)
    evaluator=FAST/'scripts/evaluate_zdrop_history_states.py'
    t=evaluator.read_text().replace("names=set(sources)|{","names=set(sources)|{'saf3r/patch/zdrop_history_cleanup.py','saf3r/triton/zdrop_history_cleanup.py',")
    evaluator.write_text(t)
    (FAST/'cleanup_audit.json').write_text(json.dumps(dict(base=str(BASE),config_unchanged=True,algorithm_unchanged=True,ordered_states=[0,1,5,10],removed=['obsolete Top4 scratch','old prefix/mixed losses','unused squared-norm Drop scratch','7 unused solver slots','unused Value means','overwritten coarse pi','unread history weights/gather','unused L0 geometry/RMS/overlap','unavailable gain placeholders']),indent=2)+'\n')
    print(FAST)
if __name__=='__main__':main()
