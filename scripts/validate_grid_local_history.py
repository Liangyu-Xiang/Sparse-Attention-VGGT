"""Same-input legacy/local probability, cache, and next-prior validation."""
import argparse,copy,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch
from saf3r.patch import reuse_headmean_history as old,reuse_grid_local_history as new
from saf3r.patch.reuse_residual_exploration import canonical_layout
from saf3r.triton.reuse_ready_history import decode
from saf3r.triton.reuse_grid_local_history import normalizers,compatibility,store


def diff(a,b):
    d=a.float()-b.float()
    return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.float().norm().clamp_min(1e-30)))


def case(h,mode,extreme,singleton=False):
    torch.manual_seed(100+h);g=5;height=1 if singleton else 7;width=1 if singleton else 9;frames=2;p=frames*height*width;a=p;r=2 if singleton else 8
    grid,_=canonical_layout(frames,height,width,2,2,'cuda');sizes=torch.bincount(grid,minlength=r).float()
    config=NS(window_groups=64,history_cache_format=mode,grid_y=2,grid_x=2)
    s=NS(n=140,groups=g,budget=a,regions=r,grid=grid,grid_sizes=sizes,height=height,width=width,config=config,
         fine=NS(base_count=13,labels=torch.arange(p,device='cuda')%13),m5c_count_errors=[],m5c_seed_errors=[],m5c_prior_errors=[],previous_layer=-1,ready_written={},ready_cache_dtype=torch.float32 if mode=='fp32' else torch.float16)
    ids=torch.full((g,a),-1,device='cuda',dtype=torch.int32)
    for j,n in enumerate([p,0,1,max(0,p-3),p//2]):ids[j,:n]=torch.randperm(p,device='cuda')[:n].int()
    raw=torch.randn(g,h,4,a,device='cuda')*(40 if extreme else 2)
    raw.masked_fill_(ids[:,None,None]<0,-torch.inf)
    base=torch.randn(g,h,4,r,device='cuda')*(30 if extreme else 1)
    special=torch.randn(g,h,4,14,device='cuda')+(130 if extreme else 0)
    # Real einsum outputs are head-major views, not contiguous GHPR arrays.
    base=base.transpose(0,1).contiguous().transpose(0,1)
    special=special.transpose(0,1).contiguous().transpose(0,1)
    # Keep m=0 uniform finite, while still exercising large +/- differences.
    base[0]=0
    coarse=torch.cat((base+sizes.log(),special),-1).softmax(-1)[...,:r]
    weights=torch.full((g,4),.25,device='cuda');weights[-1]=torch.tensor([.5,.5,0,0],device='cuda')
    obs=dict(ready_logits=raw,grid_logits=base,special_logits=special,coarse_grid_probability=coarse,weights=weights)
    bank=grid[ids.long().clamp_min(0)].masked_fill(ids<0,r)
    order=bank.argsort(dim=1,stable=True).int();counts=torch.zeros(g,r+1,device='cuda',dtype=torch.int32).scatter_add_(1,bank,torch.ones_like(ids))
    offsets=torch.nn.functional.pad(counts[:,:r].cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    obs.update(selected_grid_order=order,selected_grid_offsets=offsets)
    q=torch.empty(h,s.n,8,device='cuda');s2=copy.copy(s)
    for field in ['m5c_count_errors','m5c_seed_errors','m5c_prior_errors']:setattr(s2,field,[])
    s2.ready_written={}
    old.ensure_cache(s,q);new.ensure_cache(s2,q)
    # Plain half is explicitly initialized for the legacy reference too.
    if mode=='fp16':
        s.ready_cache_plain=True;old.ensure_cache(s,q)
    old.save_evidence(s,q,q,0,ids,ids,obs,1.)
    new.save_evidence(s2,q,q,0,ids,ids,obs,1.)
    c1=decode(s.ready_edges,a);c2=decode(s2.ready_edges,a)
    conditional=diff(c1,c2);uniform=diff(s.ready_uniform,s2.ready_uniform)
    assert conditional['max_abs'] < (2e-5 if mode=='fp32' else 5e-4),conditional
    assert uniform['relative_l2']<2e-4,uniform
    assert torch.equal(s.ready_coarse,s2.ready_coarse)
    den,err=normalizers(raw,base,order,offsets,sizes,20)
    # Independently sum per-head, per-probe mass from local probabilities.
    region=grid[ids.long().clamp_min(0)]
    cp=(raw-den.gather(-1,region[:,None,None].expand_as(raw))).exp()
    u=(base-den).exp();remaining=sizes-counts[:,:r]
    mass=(u*remaining[:,None,None]).scatter_add_(-1,region[:,None,None].expand_as(raw),cp)
    mass_error=float((mass-1).abs().max());assert mass_error<3e-5,mass_error
    # Independent per-grid torch logsumexp, including no-selected/full grids.
    expected=torch.empty_like(den)
    for group in range(g):
        for region_id in range(r):
            selected=(ids[group]>=0)&(grid[ids[group].clamp_min(0).long()]==region_id)
            logits=raw[group,...,selected]
            m=remaining[group,region_id]
            if m>0:logits=torch.cat((logits,(base[group,...,region_id]+m.log())[...,None]),-1)
            expected[group,...,region_id]=logits.logsumexp(-1)
    assert diff(expected,den)['max_abs']<2e-5
    s.previous_layer=s2.previous_layer=0
    obs_next={**obs,'coarse_grid_probability':(coarse*.95+.05*coarse.flip(-1)).contiguous()}
    pa,pi=old.build_prior(s,0,ids,obs_next);pb,pj=new.build_prior(s2,0,ids,obs_next)
    prior=diff(pa,pb);cluster=diff(pi,pj)
    assert prior['relative_l2']<.001 and cluster['relative_l2']<.001,(prior,cluster)
    # Quantization reported separately from FP32 normalization.
    cache32=torch.empty(g,1,4,a,device='cuda');uni32=torch.empty(g,1,4,r,device='cuda')
    store(raw,den,compatibility(den,special),ids,grid,base,cache32,uni32)
    quant=diff(cache32,c2)
    return dict(heads=h,cache=mode,extreme=extreme,singleton_grid=singleton,conditionals=conditional,uniform=uniform,local_mass_max_abs=mass_error,next_prior=prior,next_pi=cluster,quantization=quant,normalizer=diff(expected,den),eps_clipping_exercised=extreme)


def attention_metadata_check():
    from saf3r.triton.reuse_single_qk_grid import weighted_attention
    torch.manual_seed(17);h,n,d,g,p,a=3,79,32,3,67,41
    q=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
    ids=torch.stack([torch.randperm(p,device='cuda')[:a] for _ in range(g)]).int();ids[1,-7:]=-1
    grid=torch.arange(p,device='cuda')%7;key_grid=grid[ids.clamp_min(0).long()];bias=torch.rand(g,a,device='cuda')*4;special=torch.arange(p,n,device='cuda');meta={}
    x,m,raw=weighted_attention(q,k,v,ids,bias,special,d**-.5,key_grid,7)
    y,m2,raw2=weighted_attention(q,k,v,ids,bias,special,d**-.5,key_grid,7,evidence_layout=meta)
    assert torch.equal(x,y) and torch.equal(m,m2) and torch.equal(raw,raw2)
    for group in range(g):
        pos=(torch.tensor([4,12,20,28],device='cuda')+group*32).clamp_max(n-1)
        expected=q[:,pos].float()@k[:,ids[group].clamp_min(0).long()].float().transpose(-1,-2)*d**-.5
        expected.masked_fill_(ids[group][None,None]<0,-torch.inf)
        valid=ids[group]>=0
        assert diff(expected[...,valid],raw[group,...,valid])['relative_l2']<1e-5
    return dict(bf16_attention_bitwise_equal=True,special_keys_preserved=True,probe_logits_unbiased=True,last_query_group_rows=n-64)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();args.output_dir.mkdir(exist_ok=False,parents=True)
    checks=[]
    with torch.inference_mode():
        for h in [1,3,16]:
            for mode in ['fp32','packed','fp16']:
                for extreme in [False,True]:checks.append(case(h,mode,extreme))
        for mode in ['fp32','packed']:checks.append(case(3,mode,False,True))
        execution=attention_metadata_check()
    result=dict(complete=True,checks=checks,execution=execution,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Numerical diagnostic only; no geometry quality run',dense_speedup=None,dense_reason='Kernel correctness validation only')
    (args.output_dir/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
