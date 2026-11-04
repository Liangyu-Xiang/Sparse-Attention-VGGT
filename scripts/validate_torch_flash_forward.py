"""Same-input native Flash AV versus original AV/evidence in every global layer."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from torch.profiler import profile,ProfilerActivity
from benchmark_torch_flash import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_torch_flash as backend,reuse_serial_tuned as old
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_v8_torch_flash.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();original=backend.weighted_attention;seen=set();checks=[]
def execute(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs):
 layer=state.previous_layer+1
 if layer in seen:return original(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs)
 seen.add(layer)
 with profile(activities=[ProfilerActivity.CPU]) as prof:result=original(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs)
 layout={};other=old.weighted_attention(q,k,v,indices,bias,specials,scale,key_grid,regions,emit_evidence=kwargs.get('emit_evidence',True),evidence_layout=layout)
 rel=float((result[0].float()-other[0].float()).norm()/other[0].float().norm().clamp_min(1e-30));summary=float((result[1]-other[1]).abs().max());assert rel<.01 and summary<2e-6,(layer,rel,summary)
 raw=None
 if result[2] is not None:
  finite=torch.isfinite(result[2]);assert torch.equal(finite,torch.isfinite(other[2]));raw=float((result[2][finite]-other[2][finite]).abs().max());assert raw<1e-5,(layer,raw)
 ops=[x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key];assert 'aten::_scaled_dot_product_flash_attention' in ops and not any('efficient_attention' in x or 'attention_math' in x for x in ops),ops
 checks.append(dict(layer=layer,groups=len(indices),av_relative_l2=rel,raw_probe_max_abs=raw,grid_summary_max_abs=summary,dispatch=ops))
 (a.output_dir/'progress.json').write_text(json.dumps(checks,indent=2)+'\n');return result
backend.weighted_attention=execute;begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();assert len(checks)==24 and len(state.records)==24 and torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all()
result=dict(complete=True,checks=checks,scope='First64-group window of every layer,heads128,entire first window output/probes/Grid evidence compared on identical Q/K/V and support/bias',diagnostic_wall_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Native-dispatch and same-input AV/evidence diagnostic; geometry evaluation separate',dense_speedup=None,dense_reason='Extra original-kernel reference computations; not a timing benchmark')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
