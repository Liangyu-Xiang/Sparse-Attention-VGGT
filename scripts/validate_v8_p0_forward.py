"""Check the changed kernels against v8 on actual forward intermediates."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_v8_p0 import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_p0_states,reuse_p0_pack,reuse_fused_states
from saf3r.patch import reuse_inference_ops

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model,state=build(ROOT/'configs/evaluation/reuse_v8_p0.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();checks=[];seen_s=set();seen_p=set();orig_s=reuse_p0_states.states;orig_p=reuse_p0_pack.pack
 def states(*args):
  key=state.previous_layer+1
  result=orig_s(*args)
  if key not in seen_s:
   seen_s.add(key);prior,z,reps,labels,pop,allow,geo,ctx=args;rg=dict(geo);rc=dict(ctx)
   er,co=reuse_fused_states.states(prior,z,reps,labels,pop,allow,rg,rc)
   en,cn=result;finite=torch.isfinite(er);error=float((er[finite]-en[finite]).abs().max())
   assert torch.equal(torch.isfinite(er),torch.isfinite(en)) and torch.allclose(er,en,atol=1e-7,rtol=1e-6),(key,error)
   assert torch.equal(co,cn) and torch.equal(ctx['owners'],rc['owners'])
   assert torch.allclose(geo['objective'],rg['objective'],atol=1e-7,rtol=1e-6)
   checks.append(dict(stage='loss',layer=key,max_abs=error,exact=torch.equal(er,en),owners_exact=True,groups=len(prior)))
  return result
 def pack(*args):
  key=state.previous_layer+1;result=orig_p(*args)
  if key not in seen_p:
   seen_p.add(key);sel,reps,labels,pop,cap,geo,ctx=args;rc=dict(ctx)
   expected=reuse_inference_ops.pack(sel,None,reps,labels,pop,cap,rc)
   assert torch.equal(result[0],expected[0]) and torch.equal(result[1],expected[1]),key
   assert not ctx['pack_error'].any()
   for name in result[2]:assert torch.equal(result[2][name],expected[2][name]),(key,name)
   checks.append(dict(stage='pack',layer=key,ascending_key_order_exact=True,padding_exact=True,records_exact=True))
  return result
 reuse_p0_states.states=states;reuse_p0_pack.pack=pack
 begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();assert len(state.records)==24 and torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all()
 result=dict(complete=True,checks=checks,layers=len(state.records),diagnostic_wall_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Forward kernel comparison with extra oracle execution; geometry not evaluated and wall is not a benchmark',dense_speedup=None,dense_reason='Unchanged Dense not rerun; no paired timing')
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
