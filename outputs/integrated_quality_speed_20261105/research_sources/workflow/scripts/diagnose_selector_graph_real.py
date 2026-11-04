"""Real first nonzero-layer window: same selector eager vs CUDA Graph."""
import argparse,importlib,json,statistics,time,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.quality_metrics import quality_record,quality_markdown
from scripts.evaluate_zdrop_history_states import run
OUT=ROOT/'outputs/source_local_quality_20261105/graph_window_diagnostic'
class FinishedDiagnostic(Exception):pass
wg=importlib.import_module('saf3r.patch.reuse_window_graph')

def bench(fn,repeats=7):
 fn();torch.cuda.synchronize();samples=[]
 for _ in range(repeats):
  torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
 return dict(samples_seconds=samples,median_seconds=statistics.median(samples))

def capture(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
 geometry['p0_loss_pack']=False
 arguments=(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier)
 fn=lambda:wg.eager(*arguments,uniform=uniform)
 for _ in range(3):fn()
 torch.cuda.synchronize();start=time.perf_counter()
 window=wg.Window(*arguments,bucket_bins=True,uniform=uniform)
 torch.cuda.synchronize();creation=time.perf_counter()-start
 eager=bench(fn)
 replay=bench(lambda:window.graph.replay())
 wrapper=bench(lambda:window(prior,pi,old_ids,allowed,uniform))
 constants=bench(lambda:window.update_constants(z,reps,labels,population,geometry))
 comparisons=[]
 for shift in (0,1,2):
  p=prior.roll(shift,1).contiguous();u=uniform.roll(shift,1).contiguous()
  a=allowed if shift==0 else ~allowed
  expected=wg.eager(p,pi,old_ids,a,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier,uniform=u)
  actual=window(p,pi,old_ids,a,u)
  differences={}
  for name,left,right in list(zip(('ids','real','weights'),actual[:3],expected[:3]))+[(k,actual[3][k],expected[3][k]) for k in actual[3]]:
   if not torch.equal(left,right):
    diff=(left.double()-right.double()).abs()
    differences[name]=dict(mismatched=int((left!=right).sum()),max_abs=float(diff.max()))
  comparisons.append(dict(shift=shift,differences=differences,exact_supports=all(k not in differences for k in ('ids','real','weights'))))
 data=dict(scope='Real heads 128-frame first layer-1 window only; not full-forward',gpu=torch.cuda.get_device_name(),
  layer=layer,groups=len(prior),clusters=len(reps),width=geometry['width'],creation_and_capture_seconds=creation,
  eager=eager,pure_graph_replay=replay,graph_with_input_and_record_copies=wrapper,geometry_update=constants,comparisons=comparisons,
  full_wrapper_speedup=eager['median_seconds']/wrapper['median_seconds'],graph_vs_eager_note='One fixed real window; repeated inputs for timing. Creation excludes already-warmed eager kernels, includes Window warm run and capture. Separate geometry update measured, not part of per-window wrapper.',
  quality=quality_record({},'Explicit isolated selector diagnostic; no complete forward/pose/recon_unposed evaluated'))
 data['quality']['speedup_unavailable']='No complete full-forward or paired Dense timing'
 (OUT/'selector.json').write_text(json.dumps(data,indent=2)+'\n')
 (OUT/'quality.json').write_text(json.dumps(data['quality'],indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(data['quality']))
 (OUT/'status.json').write_text(json.dumps(dict(stage='diagnostic_complete'))+'\n')
 print(json.dumps(data,indent=2),flush=True);raise FinishedDiagnostic()
wg.select=capture
args=SimpleNamespace(output_dir=OUT,method='zdrop_history_states',scene='heads',dataset='7scenes',frames=128,
 config=ROOT/'configs/evaluation/reuse_shared_value_loss_q256_adaptive_reference.json',capture=False,warmup=1,repeats=3,diagnostic=True,selector_execution='graph',baseline=None)
try:run(args)
except FinishedDiagnostic:pass
