"""Paired same-QKV kernel cost,not full model/geometry or control-method reruns."""
import argparse,json,sys,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from saf3r.patch.current_probe_reuse import main_attention
 from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
 from saf3r.patch import streamed_probability_history as hist
 from types import SimpleNamespace
 from scripts.quality_metrics import quality_record
 torch.manual_seed(7);torch.set_num_threads(2);h,n,d,gs,cap,s=16,32768,64,256,4096,128;g=n//gs
 q=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);specials=torch.arange(n-s,n,device='cuda')
 ids=(torch.arange(cap,device='cuda')[None]+torch.arange(g,device='cuda')[:,None]*128)%(n-s);ids=ids.int().contiguous();bias=torch.rand(g,cap,device='cuda')*2;logz=torch.full((h,g),__import__('math').log(n)+.5,device='cuda')
 kwargs=dict(query_group_size=gs,block_queries=64,block_keys=128,num_warps=4,num_stages=2)
 grid=torch.arange(n-s,device='cuda')//64;regions=int(grid.max())+1
 state=SimpleNamespace(config=SimpleNamespace(query_group_size=gs,history_key_block_size=64),groups=g,regions=regions,grid=grid,patches=torch.arange(n-s,device='cuda'),specials=specials,grid_sizes=torch.bincount(grid,minlength=regions).float())
 observation=hist.observe(state,q,k,d**-.5);missing=hist.missing_normalizer(state,observation,0,ids)
 def plain():return main_attention(q,k,v,ids,specials,d**-.5,bias,**kwargs)
 def stream():return main_attention(q,k,v,ids,specials,d**-.5,bias,history_logz=logz,**kwargs)
 def split():return main_attention(q,k,v,ids,specials,d**-.5,bias,history_logz=logz,history_split=True,**kwargs)
 def d2_8():return main_attention(q,k,v,ids,specials,d**-.5,bias,history_bits=8,**kwargs)
 def d2_16():return main_attention(q,k,v,ids,specials,d**-.5,bias,history_bits=16,**kwargs)
 def full8():
  result,e= d2_8();return result,hist.finalize_d2(state,observation,0,ids,e,missing)
 def full16():
  result,e=d2_16();return result,hist.finalize_d2(state,observation,0,ids,e,missing)
 def legacy():return direct_sparse_attention_sampled(q,k,v,ids,specials,d**-.5,bias,probe_count=1,probe_group_stride=1,debiased_log=True,split_export=True,**kwargs)
 ops=dict(plain=plain,d2_8=d2_8,d2_16=d2_16,full8_cache=full8,full16_cache=full16,streamed=stream,streamed_split=split,legacy_inline_export=legacy);samples={key:[] for key in ops}
 with torch.inference_mode():
  ref,_=plain();new,e=stream();old,_=legacy();torch.testing.assert_close(new,ref,atol=.002,rtol=.002);torch.testing.assert_close(old,ref,atol=.002,rtol=.002)
  del ref,new,old,e
  for _ in range(3):
   for op in ops.values():out=op();del out
  for rep in range(9):
   order=list(ops) if rep%2==0 else list(reversed(ops))
   for key in order:
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();out=ops[key]();end.record();end.synchronize();samples[key].append(begin.elapsed_time(end)/1000);del out
 med={key:statistics.median(v) for key,v in samples.items()}
 result=dict(quality=quality_record({},'Same-QKV kernel diagnostic only;no model geometry'),shape=dict(heads=h,tokens=n,head_dimension=d,groups=g,selected_patch_keys=cap,special_keys=s),median_seconds=med,samples=samples,stream_over_plain=med['streamed']/med['plain'],legacy_export_over_plain=med['legacy_inline_export']/med['plain'],persistent_history_dimensions='G x A;no head axis',headwise_legacy_scratch_bytes=g*h*(cap+s)*2,headmean_stream_scratch_bytes=g*(cap+s)*4,speedup_vs_dense=None,speedup_reason='Kernel diagnostic,no full Dense timing',scope='Includes AV and export;excludes coarse observation,old/new cache finalization,transfer and other model modules')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
