"""Run normal changed-method inference, then capture camera features against saved Dense."""
import json,runpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from saf3r.utils import model_utils,data_utils
 from scripts.camera_token_diagnostic import Trace,similarity,save
 from scripts.quality_metrics import quality_record,quality_markdown
 output=Path(sys.argv[sys.argv.index('--output-dir')+1]);scene=sys.argv[sys.argv.index('--scene')+1]
 models=[];inputs=[];build=model_utils.build_model;load=data_utils.load_images
 def wrapped_build(*a,**kw):
  m=build(*a,**kw);models.append(m);return m
 def wrapped_load(*a,**kw):
  x=load(*a,**kw);inputs.append(x);return x
 model_utils.build_model=wrapped_build;data_utils.load_images=wrapped_load
 try:runpy.run_path(str(ROOT/'scripts/evaluate_zdrop_history_states.py'),run_name='__main__')
 finally:model_utils.build_model=build;data_utils.load_images=load
 # Normal prediction is saved before any tracing hooks are installed.
 model=models[0];images=inputs[0].cuda();base=ROOT/'outputs/camera_token_comparison_20261105/full'/scene/'dense'
 old=json.loads((base/'input.json').read_text());current=json.loads((output/'input.json').read_text())
 for k in ['image_paths','frame_indices','shape','checkpoint_sha256']:assert old[k]==current[k],k
 reference=torch.load(base/'camera_tensors.pt',map_location='cpu',weights_only=True,mmap=True)['camera_features_and_attention_summaries']
 class FeatureTrace(Trace):
  def full_raw(self,*a):pass
  def attn_pre(self,kind,layer,block,wrapper,original,x,pos):self.feature(f'{kind}_{layer:02d}_attention_input_norm',x)
  def pre_proj(self,kind,layer,original,x):self.feature(f'{kind}_{layer:02d}_attention_pre_Wo',x)
 trace=FeatureTrace(model,'ours',images,output,reference);trace.install();torch.manual_seed(0)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
  torch.cuda.synchronize();elapsed=time.perf_counter()-begin
 finally:trace.close()
 normal=torch.load(output/'predictions.pt',map_location='cpu',weights_only=True,mmap=True)
 repeat={k:similarity(pred[k].cpu().reshape(-1,pred[k].shape[-1]),normal[k].reshape(-1,normal[k].shape[-1]),False) for k in ['pose_enc','depth','depth_conf']}
 assert len(trace.features)==306,len(trace.features)
 assert all(torch.isfinite(pred[k]).all() for k in ['pose_enc','depth','depth_conf'])
 q=quality_record({},'Separate camera-feature diagnostic forward; normal uninstrumented prediction is evaluated for pose/recon_unposed in the parent quality run')
 folder=output/'camera';folder.mkdir();save(folder/'quality.json',q);(folder/'quality.md').write_text(quality_markdown(q))
 save(folder/'result.json',dict(complete=True,scene=scene,features=trace.features,quality=q,normal_vs_diagnostic=repeat,instrumented_seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,dense_reference=str(base),input=current,scope='All24 frame and24 global blocks; camera and register feature outputs at6 stages; camera-head input,normalization,and16 trunk blocks. Saved Dense reference reused; no Dense rerun; no attention PMF reconstruction in this experiment.'))
 torch.save(trace.raw,folder/'camera_features.pt');print('CAMERA_COMPLETE',scene,len(trace.features),repeat['pose_enc'],flush=True)
if __name__=='__main__':main()
