"""Original kernels are numerical oracles, not full unchanged-control reruns."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,triton
from saf3r.triton import reuse_fused_states as old,reuse_p0_states as new
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch.reuse_inference_ops import pack as old_pack
from saf3r.triton.reuse_p0_pack import pack as new_pack

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(7);torch.set_num_threads(2);checks=[];start=time.perf_counter()
 for groups in [1,3,8]:
  for dtype in [torch.float32,torch.bfloat16]:
   device='cuda';sizes=torch.tensor([0,1,2,3,4,7,8,9,16,17,32,33,64],device=device);f=len(sizes);n=int(sizes.sum());w=64;d=64
   perm=torch.randperm(n,device=device);labels=torch.empty(n,device=device,dtype=torch.int64);members=torch.full((f,w),-1,device=device,dtype=torch.int32);reps=torch.zeros(f,device=device,dtype=torch.int64);offset=0
   for c,size in enumerate(sizes.tolist()):
    ids=perm[offset:offset+size].sort().values;labels[ids]=c;members[c,:size]=ids.int()
    if size:reps[c]=ids[0]
    offset+=size
   z=torch.randn(n,d,device=device,dtype=dtype);dist=torch.rand(n,w,device=device);dr=torch.rand(n,device=device)
   prior=torch.rand(groups,n,device=device);prior/=prior.sum(1,keepdim=True)
   qual=torch.rand(groups,n,device=device)>.7;amount=torch.zeros(groups,f,device=device,dtype=torch.int32).scatter_add_(1,labels[None].expand(groups,-1),qual.int())
   remove=qual[:,reps];allowed=torch.rand(f,groups,device=device).t()>.4
   geo=dict(members=members,distances=dist,rep_distance=dr,width=w,drop_bins=prepare(sizes),l2=True)
   ctx=dict(qualified=qual,eligible_count=amount,remove_rep=remove)
   a_ctx=dict(ctx);b_ctx=dict(ctx);a_geo=dict(geo);b_geo=dict(geo)
   er,co=old.states(prior,z,reps,labels,sizes.float(),allowed,a_geo,a_ctx)
   en,cn=new.states(prior,z,reps,labels,sizes.float(),allowed,b_geo,b_ctx)
   finite=torch.isfinite(er);assert torch.equal(finite,torch.isfinite(en))
   error=float((er[finite]-en[finite]).abs().max());assert torch.allclose(er,en,atol=1e-7,rtol=1e-6),error
   assert torch.equal(co,cn) and torch.equal(a_ctx['owners'],b_ctx['owners'])
   assert torch.allclose(a_geo['objective'],b_geo['objective'],atol=1e-7,rtol=1e-6)
   sel=torch.tensor([0,1,5,10],device=device)[torch.randint(4,(groups,f),device=device)];sel[:,0]=0;sel[(sel==1)&remove]=5
   ia,ra,rec_a=old_pack(sel,None,reps,labels,sizes,n,a_ctx)
   ib,rb,rec_b=new_pack(sel,reps,labels,sizes,n,geo,b_ctx)
   assert torch.equal(ia,ib) and torch.equal(ra,rb)
   assert not a_ctx['pack_error'].any() and not b_ctx['pack_error'].any()
   for key in rec_a:assert torch.equal(rec_a[key],rec_b[key]),key
   checks.append(dict(groups=groups,dtype=str(dtype),max_loss_abs=error,key_order_exact=True,owners_exact=True,includes_empty_cluster=True))
 result=dict(complete=True,checks=checks,runtime_seconds=time.perf_counter()-start,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel numerical diagnostic; no model geometry evaluation',dense_speedup=None,dense_reason='Unchanged Dense not rerun; no paired timing')
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
