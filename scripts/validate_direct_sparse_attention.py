"""Grid-free AV and actual normalized probability A/B,FP16/BF16 and capture."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(12);torch.backends.cuda.matmul.allow_tf32=False;checks=[];start=time.perf_counter()
for dtype in [torch.float16,torch.bfloat16]:
 for rows in [79,17]:
  h,n,d,cap,s=3,97,64,61,5;g=(rows+31)//32;q=torch.randn(h,rows+7,d,device='cuda',dtype=dtype)[:,3:3+rows];k=torch.randn(h,n+s,d,device='cuda',dtype=dtype);v=torch.randn_like(k);ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();ids[:,-13:]=-1;ids[-1,:]=-1;specials=torch.arange(n,n+s,device='cuda')
  for mode in ['pure_sparse','compensated_sparse']:
   bias=None if mode=='pure_sparse' else (torch.rand(g,cap,device='cuda')*4).masked_fill(ids<0,-torch.inf)
   for probes in [1,4]:
    positions=(torch.arange(g,device='cuda')[:,None]*32+torch.tensor([4] if probes==1 else [4,12,20,28],device='cuda')[None]).clamp_max(rows-1).int()
    reference=torch.empty((h,rows,d),device='cuda');prob=torch.zeros((g,probes,s+cap),device='cuda');lse=torch.empty((h,rows),device='cuda')
    for group in range(g):
     valid=ids[group]>=0;keys=torch.cat([specials,ids[group,valid].long()]);b=torch.cat([torch.zeros(s,device='cuda'),bias[group,valid] if bias is not None else torch.zeros(int(valid.sum()),device='cuda')]);z=q[:,group*32:min((group+1)*32,rows)].float()@k[:,keys].float().transpose(-1,-2)*d**-.5+b
     reference[:,group*32:min((group+1)*32,rows)]=z.softmax(-1)@v[:,keys].float();lse[:,group*32:min((group+1)*32,rows)]=z.logsumexp(-1)
     zp=q[:,positions[group]].float()@k[:,keys].float().transpose(-1,-2)*d**-.5+b;pp=zp.softmax(-1).mean(0);prob[group,:,:s]=pp[:,:s];prob[group,:,s:][:,valid]=pp[:,s:]
    for export in ['inline','lse_probe']:
     out,e=direct_sparse_attention(q,k,v,ids,specials,d**-.5,logit_bias=bias,probe_positions=positions,return_history_probability=True,history_export=export)
     rel=float((out.float()-reference).norm()/reference.norm());error=float((out.float()-reference).abs().max());assert rel<(.001 if dtype==torch.float16 else .005),(dtype,rel)
     actual=torch.cat([e['special_probability'],e['patch_probability']],-1);perror=float((actual-prob).abs().max());norm=float((actual.sum(-1)-1).abs().max());assert perror<(2e-4 if export=='inline' else 2e-6) and norm<(5e-4 if export=='inline' else 3e-6),(export,perror,norm)
     assert torch.allclose(e['lse'],lse,atol=2e-6,rtol=2e-6)
     checks.append(dict(dtype=str(dtype),query_rows=rows,mode=mode,probes=probes,export=export,av_relative_l2=rel,av_max_abs=error,probability_max_abs=perror,normalization_max_abs=norm,scratch_bytes=e['scratch_bytes']))
# Real CUDA Graph replay,mutated Q (not cached outputs).
q=torch.randn(2,65,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(2,90,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);ids=torch.arange(32,device='cuda',dtype=torch.int32)[None].repeat(3,1);specials=torch.arange(80,90,device='cuda');bias=torch.rand(3,32,device='cuda')
direct_sparse_attention(q,k,v,ids,specials,.125,bias,return_history_probability=True)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):captured,ce=direct_sparse_attention(q,k,v,ids,specials,.125,bias,return_history_probability=True)
q.add_(.125);graph.replay();expected,ee=direct_sparse_attention(q,k,v,ids,specials,.125,bias,return_history_probability=True);assert torch.equal(captured,expected) and torch.equal(ce['patch_probability'],ee['patch_probability'])
result=dict(complete=True,checks=checks,cuda_graph_replay_fresh_inputs=True,diagnostic_wall_seconds=time.perf_counter()-start,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel/probability numerical and capture tests;no geometry evaluation',dense_speedup=None,dense_reason='Kernel diagnostic,not paired full-model speedup')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
