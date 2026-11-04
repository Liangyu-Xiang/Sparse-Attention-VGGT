"""GPU numerical checks for actual sparse export and Grid-mass calibration."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_serial_sparseprob import weighted_attention
from saf3r.triton import reuse_sparse_beta_history as kernel
from saf3r.triton.reuse_ready_history import decode
from saf3r.patch.reuse_sparse_beta_history import calibrate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(4);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
    h,n,rows,d,g,cap,regions=3,83,79,64,3,40,9
    grid=torch.arange(n,device='cuda')%regions;sizes=torch.bincount(grid,minlength=regions).float()
    first=torch.cat([torch.where(grid==0)[0],torch.where(grid!=0)[0][:cap-int(sizes[0])]])
    second=torch.where(grid!=1)[0][:cap]
    ids=torch.stack([first,second,torch.full_like(first,-1)]).int();ids[1,-4:]=-1
    weights=torch.tensor([[.25,.25,.25,.25],[.25,.25,.25,.25],[1/3,1/3,1/3,0]],device='cuda')
    specials=torch.arange(n,n+5,device='cuda')
    bias=torch.rand(g,cap,device='cuda').clamp_min(.1).log().masked_fill(ids<0,-torch.inf)
    for dtype in [torch.float32,torch.bfloat16]:
        q=torch.randn(h,rows,d,device='cuda',dtype=dtype);k=torch.randn(h,n+5,d,device='cuda',dtype=dtype);v=torch.randn_like(k)
        layout={};out,_,raw=weighted_attention(q,k,v,ids,bias,specials,d**-.5,grid[ids.long().clamp_min(0)],regions,evidence_layout=layout)
        expected=torch.zeros((g,4,cap),device='cuda');reference=torch.empty_like(out)
        for group in range(g):
            positions=(torch.arange(4,device='cuda')*8+4+group*32).clamp_max(rows-1)
            valid=ids[group]>=0;all_ids=torch.cat([specials,ids[group,valid].long()]);b=torch.cat([torch.zeros(len(specials),device='cuda'),bias[group,valid]])
            z=q[:,group*32:min((group+1)*32,rows)].float()@k[:,all_ids].float().transpose(-1,-2)*d**-.5+b
            reference[:,group*32:min((group+1)*32,rows)]=(z.softmax(-1)@v[:,all_ids].float()).to(dtype)
            prob=(q[:,positions].float()@k[:,all_ids].float().transpose(-1,-2)*d**-.5+b).softmax(-1)
            expected[group,:,valid]=prob[:,:,len(specials):].mean(0)
        rel=float((reference.float()-out.float()).norm()/reference.float().norm())
        assert rel<(1e-5 if dtype==torch.float32 else .005),rel
        cache=torch.empty((g,1,4,(cap*14+7)//8),device='cuda',dtype=torch.uint8)
        mass=torch.zeros((g,1,4,regions),device='cuda')
        kernel.export(raw,bias,layout['sparse_lse'],ids,grid,cache,mass)
        actual=decode(cache,cap)[:,0]
        error=float((actual-expected).abs().max())
        quantized_reference=expected.half().float()
        assert torch.allclose(actual,quantized_reference,atol=1e-7,rtol=.001),(dtype,error,float((actual-quantized_reference).abs().max()))
        counts=torch.zeros((g,regions),device='cuda',dtype=torch.int32).scatter_add_(1,grid[ids.long().clamp_min(0)],(ids>=0).int())
        refmass=torch.zeros((g,4,regions),device='cuda').scatter_add_(2,grid[ids.long().clamp_min(0)][:,None].expand(-1,4,-1),actual)
        assert torch.allclose(mass[:,0],refmass,atol=1e-7,rtol=1e-6)
        current=torch.rand(g,4,regions,device='cuda');current=current/current.sum(-1,keepdim=True)*.9
        previous=torch.rand_like(current);previous=previous/previous.sum(-1,keepdim=True)*.8
        factor,fallback,uniform,beta=calibrate(current,previous,mass[:,0],counts,sizes)
        prior=(uniform*weights[:,:,None]).sum(1)[:,grid].contiguous()
        kernel.apply(cache,ids,grid,factor,fallback,weights,prior)
        expected_prior=(uniform*weights[:,:,None]).sum(1)[:,grid].clone()
        for group in range(g):
            for slot,key in enumerate(ids[group].tolist()):
                if key>=0:
                    region=int(grid[key]);expected_prior[group,key]=((actual[group,:,slot]*factor[group,:,region]+fallback[group,:,region])*weights[group]).sum()
        assert torch.allclose(prior,expected_prior,atol=1e-7,rtol=1e-6)
        reconstructed=torch.zeros((g,regions),device='cuda').scatter_add_(1,grid[None].expand(g,-1),prior)
        target=(current*weights[:,:,None]).sum(1)
        conserved=float((reconstructed-target).abs().max());assert conserved<2e-6,conserved
        assert torch.isfinite(prior).all() and (prior>=0).all()
        # Explicit all-selected/zero-FP16-mass underflow fallback.
        zero=mass[:,0].clone();zero[0,:,0]=0
        f,fb,u,_=calibrate(current,previous,zero,counts,sizes)
        assert torch.equal(f[0,:,0],torch.zeros(4,device='cuda'))
        assert torch.allclose(fb[0,:,0]*sizes[0],current[0,:,0]) and torch.equal(u[0,:,0],torch.zeros(4,device='cuda'))
        checks.append(dict(dtype=str(dtype),attention_relative_l2=rel,sparse_export_max_abs=error,grid_conservation_max_abs=conserved,
                           includes_padding=True,includes_duplicate_tail_probes=True,includes_empty_full_and_underflow_grids=True))
    result=dict(complete=True,checks=checks,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Small GPU numerical diagnostic; model geometry not evaluated',dense_speedup=None,dense_reason='No unchanged Dense rerun authorized')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
