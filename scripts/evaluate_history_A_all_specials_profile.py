"""Run unchanged full-forward timing CLI, then profile two extra forwards on same model."""
import json,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from saf3r.utils import model_utils,data_utils
 from scripts.quality_metrics import quality_record,quality_markdown
 output=Path(sys.argv[sys.argv.index('--output-dir')+1]);name=output.name
 models=[];inputs=[];build=model_utils.build_model;load=data_utils.load_images
 def wrapped_build(*a,**kw):
  m=build(*a,**kw);models.append(m);return m
 def wrapped_load(*a,**kw):
  x=load(*a,**kw);inputs.append(x);return x
 model_utils.build_model=wrapped_build;data_utils.load_images=wrapped_load
 try:runpy.run_path(str(ROOT/'scripts/evaluate_zdrop_history_states.py'),run_name='__main__')
 finally:model_utils.build_model=build;data_utils.load_images=load
 model=models[0];images=inputs[0].cuda();state=model.reuse_explore_state;state.profiling=True;samples=[]
 for i in range(2):
  torch.cuda.synchronize();begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(images)
  torch.cuda.synchronize();seconds=time.perf_counter()-begin;del prediction;times={};per_layer={}
  for layer,stage,a,b in state.profile_events:
   value=a.elapsed_time(b);times[stage]=times.get(stage,0.)+value;key=f'{layer}:{stage}';per_layer[key]=per_layer.get(key,0.)+value
  names=['value_update','pooling','cluster_seed_prepare','coarse_observation','m5c_coarse_observation','reuse_and_budget','residual_fine_probe']
  samples.append(dict(instrumented_full_forward_seconds=seconds,stage_gpu_ms=times,layer_stage_gpu_ms=per_layer,selection_prior_budget_gpu_ms=times.get('reuse_and_budget',0.),mask_pipeline_gpu_ms=sum(times.get(k,0.) for k in names),graph_creations=state.graph_cache_creations,graph_evictions=state.graph_cache_evictions))
 out=output.parent.parent/'profiles_quiet';out.mkdir(exist_ok=True,parents=True)
 q=quality_record({},'Two additional production-path CUDA-event profile forwards after normal timing; no pose/recon_unposed evaluation; not production end-to-end speed')
 (out/f'{name}_quality.json').write_text(json.dumps(q,indent=2)+'\n');(out/f'{name}_quality.md').write_text(quality_markdown(q))
 (out/f'{name}.json').write_text(json.dumps(dict(method=name,config=json.loads((output/'config.json').read_text()),quality=q,samples=samples,scope='Same model/input/GPU as quiet2warm+5repeat production timings; two subsequent instrumented forwards, rep_history_diagnostics=False; no overlapping quality jobs. Mask pipeline CUDA event total includes clustering, shared statistics, coarse observation and prior/gating/loss/solver/packing/weight stages; selection subset excludes clustering. Events may include CPU submission gaps; additional full-forward seconds are not substituted for uninstrumented production timing.'),indent=2)+'\n')
 print('QUIET_MASK_PROFILE_COMPLETE',name,[(s['mask_pipeline_gpu_ms'],s['selection_prior_budget_gpu_ms']) for s in samples],flush=True)
if __name__=='__main__':main()
