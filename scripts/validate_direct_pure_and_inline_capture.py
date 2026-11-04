"""Pure-mode old-Grid oracle and INLINE export capture/replay."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
from saf3r.triton.reuse_serial_tuned import weighted_attention
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(16);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
for dtype in [torch.float16,torch.bfloat16]:
 q=torch.randn(3,83,64,device='cuda',dtype=dtype);k=torch.randn(3,112,64,device='cuda',dtype=dtype);v=torch.randn_like(k);ids=torch.stack([torch.randperm(100,device='cuda')[:69] for _ in range(3)]).int();ids[:,-5:]=-1;specials=torch.arange(100,112,device='cuda');bias=torch.zeros_like(ids,dtype=torch.float32).masked_fill(ids<0,-torch.inf)
 out,e=direct_sparse_attention(q,k,v,ids,specials,.125)
 old,_,_=weighted_attention(q,k,v,ids,bias,specials,.125,ids.clamp_min(0)%7,7,emit_evidence=False)
 rel=float((out.float()-old.float()).norm()/old.float().norm());assert rel<.005
 direct_sparse_attention(q,k,v,ids,specials,.125,return_history_probability=True,history_export='inline')
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):co,ce=direct_sparse_attention(q,k,v,ids,specials,.125,return_history_probability=True,history_export='inline')
 q.add_(.0625);graph.replay();ref,re=direct_sparse_attention(q,k,v,ids,specials,.125,return_history_probability=True,history_export='inline');assert torch.equal(co,ref) and torch.equal(ce['patch_probability'],re['patch_probability'])
 checks.append(dict(dtype=str(dtype),pure_vs_grid_relative_l2=rel,inline_capture_fresh_inputs_exact=True))
result=dict(complete=True,checks=checks,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Pure operator and inline-capture diagnostic,no geometry evaluation',dense_speedup=None,dense_reason='Not performance timing',runtime_seconds=None,runtime_reason='Numerical/capture diagnostic,not benchmark',peak_memory_gib=None,peak_reason='Not separately measured;full model and all-window peak recorded elsewhere')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
