"""Numerical and CUDA Graph contracts for one-launch sampled AV+probability."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
outdir=Path(sys.argv[1]);outdir.mkdir(parents=True,exist_ok=False);torch.manual_seed(77);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
for dtype in [torch.float16,torch.bfloat16]:
 for rows in [79,17]:
  h,n,d,a,s=3,97,64,61,5;g=(rows+31)//32;q=torch.randn(h,rows+7,d,device='cuda',dtype=dtype)[:,3:rows+3];k=torch.randn(h,n+s,d,device='cuda',dtype=dtype);v=torch.randn_like(k)
  ids=torch.stack([torch.randperm(n,device='cuda')[:a] for _ in range(g)]).int();ids[:,7]=-1;ids[:,-3:]=-1;special=torch.arange(n,n+s,device='cuda')
  for empty in [False,True]:
   current_ids=ids.clone()
   if empty:current_ids[-1]=-1
   for probes in [1,4]:
    pos=(torch.arange(g,device='cuda')[:,None]*32+torch.tensor([4] if probes==1 else [4,12,20,28],device='cuda')[None]).clamp_max(rows-1).int()
    for compensated in [False,True]:
     bias=None if not compensated else torch.rand(g,a,device='cuda')*60
     if bias is not None:bias[:,3]=-torch.inf
     ref,re=direct_sparse_attention(q,k,v,current_ids,special,.125,bias,pos,True)
     _,rlog=direct_sparse_attention(q,k,v,current_ids,special,.125,bias,pos,True,history_export='debiased_log_lse_probe')
     for stride in [1,2,4,8,16]:
      actual,e=direct_sparse_attention_sampled(q,k,v,current_ids,special,.125,bias,pos,probe_count=probes,probe_group_stride=stride)
      p=torch.cat([e['special_probability'],e['patch_probability']],-1);rp=torch.cat([re['special_probability'],re['patch_probability']],-1)[::stride]
      error=float((p-rp).abs().max());norm=float((p.sum(-1)-1).abs().max());assert error<2e-4 and norm<5e-4,(dtype,rows,probes,stride,error,norm)
      av=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30));assert av<.001,(dtype,av)
      _,log=direct_sparse_attention_sampled(q,k,v,current_ids,special,.125,bias,pos,probe_count=probes,probe_group_stride=stride,debiased_log=True)
      lp=torch.cat([log['debiased_log_special_probability'],log['debiased_log_patch_probability']],-1);oracle=torch.cat([rlog['debiased_log_special_probability'],rlog['debiased_log_patch_probability']],-1)[::stride];finite=torch.isfinite(oracle)
      logerr=float((lp[finite]-oracle[finite]).abs().max());assert torch.isfinite(lp[finite]).all() and logerr<.08,logerr
      checks.append(dict(dtype=str(dtype),rows=rows,empty_patch_tail=empty,probes=probes,compensated=compensated,stride=stride,av_relative_l2=av,probability_max_abs=error,normalization_max_abs=norm,debiased_log_max_abs=logerr))
# Mutate Q,K,V and bias AFTER graph capture; both branches must use fresh values.
graphs=[]
q=torch.randn(2,79,64,device='cuda',dtype=torch.bfloat16);k=torch.randn(2,101,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);ids=torch.arange(48,device='cuda',dtype=torch.int32)[None].repeat(3,1);special=torch.arange(97,101,device='cuda');bias=torch.rand(3,48,device='cuda')
for log in [False,True]:
 for probes,stride in [(1,2),(1,8),(4,2)]:
  def fn():return direct_sparse_attention_sampled(q,k,v,ids,special,.125,bias,probe_count=probes,probe_group_stride=stride,debiased_log=log)
  fn();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):co,ce=fn()
  q.add_(.07);k.mul_(.99);v.add_(.03);bias.add_(.1);graph.replay();eo,ee=fn();key='debiased_log_patch_probability' if log else 'patch_probability'
  assert torch.equal(co,eo) and torch.equal(ce[key],ee[key])
  graphs.append(dict(probes=probes,stride=stride,debiased_log=log,fresh_inputs=True))
result=dict(complete=True,checks=checks,cuda_graphs=graphs,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel numerical/capture tests only,no pose or reconstruction evaluation',dense_speedup=None,dense_reason='No paired full-forward Dense timing in kernel diagnostic')
(outdir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,cases=len(checks),graphs=graphs,max_probability_error=max(x['probability_max_abs'] for x in checks)),indent=2))
from scripts.quality_metrics import quality_record,quality_markdown
(outdir/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
