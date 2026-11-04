"""Numerical contracts for debiased log export, missing completion and cache.

No geometry ground truth is read. Includes all/none/partial regions, zero
assignment weights, very large compensation, tail probes and mutated graphs.
"""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
from saf3r.triton import reuse_probability_history as kernels


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20261104);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
    for dtype in (torch.float16,torch.bfloat16):
      for probes in (1,4):
        h,r,n,d,g,s,cap=3,77,80,64,3,5,51
        q=torch.randn(h,r+4,d,device='cuda',dtype=dtype)[:,2:r+2];k=torch.randn(h,n+s,d,device='cuda',dtype=dtype);v=torch.randn_like(k)
        ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();ids[:,7]=-1;ids[-1]=-1
        special=torch.arange(n,n+s,device='cuda');bias=torch.rand(g,cap,device='cuda')*60;bias[:,3]=-torch.inf
        pos=(torch.arange(g,device='cuda')[:,None]*32+torch.tensor([4] if probes==1 else [4,12,20,28],device='cuda')).clamp_max(r-1).int()
        out,e=direct_sparse_attention(q,k,v,ids,special,d**-.5,bias,pos,True,history_export='debiased_log_lse_probe')
        worst=0.;avworst=0.
        for group in range(g):
            keys=torch.cat((special,ids[group].long().clamp_min(0)));valid=torch.cat((torch.ones(s,device='cuda',dtype=torch.bool),ids[group]>=0))
            b=torch.cat((torch.zeros(s,device='cuda'),bias[group]))
            raw=q[:,pos[group]].double()@k[:,keys].double().transpose(-1,-2)*d**-.5
            logz=(raw+b).masked_fill(~valid,-torch.inf).logsumexp(-1)
            expected=torch.logsumexp(raw-logz[:,:,None],0)-math.log(h)
            actual=torch.cat((e['debiased_log_special_probability'][group],e['debiased_log_patch_probability'][group]),-1)
            err=float((actual[:,valid]-expected[:,valid]).abs().max());worst=max(worst,err);assert err<3e-5,err
            first=group*32;last=min(r,first+32)
            z=q[:,first:last].double()@k[:,keys].double().transpose(-1,-2)*d**-.5+b
            ref=z.masked_fill(~valid,-torch.inf).softmax(-1)@v[:,keys].double()
            rel=float((out[:,first:last].double()-ref).norm()/ref.norm());avworst=max(avworst,rel);assert rel<.006,rel
        assert torch.isneginf(e['debiased_log_patch_probability'][-1]).all()
        checks.append(dict(dtype=str(dtype),probes=probes,log_export_max_abs=worst,av_relative_l2=avworst,zero_assignment_preserves_finite_own_score=bool(torch.isfinite(e['debiased_log_patch_probability'][0,:,3]).all())))
    # Unsorted repeated Grid runs, holes, all-selected and empty-selected Grids.
    g,p,n,r,a=3,4,73,7,64;grid=(torch.arange(n,device='cuda')%r).long();sizes=torch.bincount(grid,minlength=r).float()
    ids=torch.full((g,a),-1,device='cuda',dtype=torch.int32)
    ids[0,:n//2]=torch.randperm(n,device='cuda')[:n//2].int();ids[1,:int((grid==0).sum())]=torch.where(grid==0)[0].int()
    ids[0,2]=-1
    score=torch.randn(g,p,a,device='cuda')*7-110;missing=torch.randn(g,p,r,device='cuda')*3-100
    cache=torch.empty((g,1,p,a),device='cuda',dtype=torch.float16)
    uniform,counts,den=kernels.store(score,missing,ids,grid,sizes,cache)
    reference=torch.zeros_like(score);reference_uniform=torch.empty_like(uniform);refden=torch.empty_like(den)
    for group in range(g):
      for region in range(r):
        mask=(ids[group]>=0)&(grid[ids[group].long().clamp_min(0)]==region);remaining=int(sizes[region]-mask.sum())
        terms=score[group,:,mask].double()
        if remaining:terms=torch.cat((terms,missing[group,:,region,None].double()+math.log(remaining)),-1)
        z=terms.logsumexp(-1);refden[group,:,region]=z
        reference[group,:,mask]=(score[group,:,mask].double()-z[:,None]).float()
        reference_uniform[group,:,region]=((missing[group,:,region].double()-z).exp() if remaining else torch.zeros(p,device='cuda'))
    assert torch.allclose(den,refden,atol=2e-5,rtol=0),(den-refden).abs().max()
    valid=(ids>=0)[:,None].expand(-1,p,-1)
    # FP32 reduction can cross an FP16 rounding boundary versus the FP64
    # oracle. Check the half-ULP storage contract, not bitwise oracle rounding.
    quantization_error=(cache[:,0].float()[valid]-reference[valid]).abs()
    assert (quantization_error <= reference[valid].abs()*.00055+3e-5).all(),quantization_error.max()
    assert torch.allclose(uniform,reference_uniform,atol=2e-6,rtol=2e-5)
    current=torch.rand(g,1,p,r,device='cuda');current=current/(current.sum(-1,keepdim=True)+.3)
    previous=torch.rand_like(current);previous/=previous.sum(-1,keepdim=True)+.3
    weights=torch.ones(g,p,device='cuda')/p;ratio=(current+1e-30)/(previous+1e-30);beta=torch.where(ratio<1,ratio,ratio.pow(-2))
    base=current*(beta*uniform[:,None]+(1-beta)/sizes)
    prior=(base*weights[:,None,:,None]).sum(2).mean(1)[:,grid].contiguous()
    kernels.predict(cache,ids,grid,sizes,current,previous,weights,prior,1e-30)
    target=current.mean((1,2)).sum(-1);mass_error=float((prior.sum(-1)-target).abs().max());assert mass_error<.003,mass_error
    assert torch.isfinite(prior).all() and (prior>=0).all()
    # Actual CUDA graph capture/replay with changed score values.
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):gu,gc,gd=kernels.store(score,missing,ids,grid,sizes,cache)
    score.add_(.17);graph.replay();eu,ec,ed=kernels.store(score,missing,ids,grid,sizes,cache)
    assert torch.allclose(gu,eu) and torch.equal(gc,ec) and torch.allclose(gd,ed)
    result=dict(complete=True,checks=checks,conditional_log_reference_max_abs=float((den-refden).abs().max()),quantized_prior_patch_mass_max_error=mass_error,
        fp16_probability_underflow_test=bool((score.exp().half()==0).all()),cuda_graph_fresh_inputs=True,
        quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},quality_reason='Numerical history diagnostic; no pose/reconstruction evaluation',dense_speedup=None,dense_reason='No paired full-forward timing in numerical diagnostic')
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    from scripts.quality_metrics import quality_record,quality_markdown
    (args.output_dir/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
if __name__=='__main__':main()
