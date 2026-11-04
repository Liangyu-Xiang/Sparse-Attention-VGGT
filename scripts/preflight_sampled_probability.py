"""Synthetic compile/performance preflight;fixed heads128 dimensions,no GT."""
import json,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
h,r,n,a,s,d=16,2048,133248,26009,640,64;q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);ids=torch.randint(0,n-s,(r//32,a),device='cuda',dtype=torch.int32);special=torch.arange(n-s,n,device='cuda');bias=torch.rand(r//32,a,device='cuda')*3
rows=[]
for stride in [0,1,2,4,8,16,64]:
 fn=(lambda:direct_sparse_attention(q,k,v,ids,special,.125,bias)) if stride==0 else (lambda:direct_sparse_attention_sampled(q,k,v,ids,special,.125,bias,probe_group_stride=stride))
 for _ in range(2):out,e=fn()
 values=[]
 for _ in range(9):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();out,e=fn();en.record();en.synchronize();values.append(st.elapsed_time(en)/1000)
 rows.append(dict(stride=stride,seconds=statistics.median(values),resources=e.get('compiler_resources')))
 if stride==0:reference=out
 else:assert torch.equal(out,reference)
 print(rows[-1],flush=True)
result=dict(complete=True,scope='Synthetic BF16 head128-like dimensions;not actual model trajectory',rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Synthetic kernel preflight;no geometry evaluation',dense_speedup=None,dense_reason='Kernel-only,not full-forward Dense comparison');(p/'result.json').write_text(json.dumps(result,indent=2)+'\n')
from scripts.quality_metrics import quality_record,quality_markdown
(p/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
