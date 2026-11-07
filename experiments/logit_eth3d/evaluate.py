"""ETH3D changed-method quality, timings and separate fixed-support diagnostics."""
import argparse,gc,hashlib,json,sys,time,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
# Select repository namespaces without importing optional unrelated models.
m=types.ModuleType('benchmarks');m.__path__=[str(ROOT/'evaluation/benchmarks')];sys.modules['benchmarks']=m
import saf3r
m=types.ModuleType('saf3r.patch');m.__path__=[str(ROOT/'saf3r/patch')];sys.modules['saf3r.patch']=m;saf3r.patch=m
from scripts.quality_metrics import quality_record,quality_markdown

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def quality(out,q):
 save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q))

def run(a):
 import numpy as np,torch,yaml,open3d as o3d
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from saf3r.patch.previous_logit_delta_forward import install
 from benchmarks.da3.bench.registries import MV_REGISTRY
 from benchmarks.da3.da3bench_utils import sample_frames
 from evaluate_da3bench import load_scene_data,evaluate
 out=a.output;out.mkdir(parents=True,exist_ok=False);quality(out,quality_record({},'Run in progress'))
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 expected=json.loads(a.expected_input.read_text());dataset=MV_REGISTRY.get('eth3d')()
 positions,scene=sample_frames(dataset.get_data(a.scene),a.scene,128,sample_mode='uniform',seed=0)
 assert scene.image_files==expected['image_paths'] and positions==expected['frame_indices']
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=expected['checkpoint']
 install('centered050');model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',scene.image_files).cuda()
 assert list(images.shape)==expected['shape']
 sha=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest();assert sha==expected['checkpoint_sha256']
 config=json.loads(a.config.read_text());state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
 save(out/'input.json',expected);save(out/'config.json',config)
 save(out/'status.json',dict(stage='full_forward',scene=a.scene))
 samples=[];raw=None
 for iteration in range(4):
  del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-begin,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
  print('FORWARD',a.scene,iteration,samples[-1],flush=True)
 assert len(state.records)==24
 def cpu(x):
  if torch.is_tensor(x):return x.item() if x.numel()==1 else x.cpu().tolist()
  if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
  if isinstance(x,list):return [cpu(v) for v in x]
  return x
 layers=cpu(state.records);save(out/'layers.json',layers)
 assert all(x['history_source_layer']==x['layer']-1 and not x['unobserved_fine_history_inherited'] and x['execution_compensation']=='sqrt_population_prior_residual' and x['refinement']=='previous_logit_delta_centered050' and x['current_fine_patch_qk_before_mask']==0 and x['validation_query_probes']==0 for x in layers)
 assert torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all()
 ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
 pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
 timing=dict(warmup=1,repeats=3,median_seconds=float(np.median([s['seconds'] for s in samples[1:]])),cold_samples=samples[:1],samples=samples[1:],scope='Full model forward: reset,coarse pooling,clustering,selection,history export/update,special correction,heads; diagnostic excluded')
 common=dict(changed_component='Previous actual sparse-mask logit residual,centered on observed support per block and shrunk by0.5;no rolling history or current fine QK before selection',method='previous_logit_delta_centered050_actual925',scene=a.scene,dataset='eth3d',frames=len(images),input=expected,config=config,timing=timing,quality_protocol=dict(tasks=['pose','recon_unposed'],sampling='uniform',seed=0,f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,cd_definition='(ACC+Comp)/2 unsquared metres',reconstruction='unposed',alignment_fusion='Unchanged baseline evaluate_da3bench/ETH3D implementation',SAF3R_TopK_profile='N/A:not SAF3R',AUC_F1_units='fractions',distance_units='metres'),hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')),speedup_vs_dense=None,speedup_reason='Unchanged Dense not rerun; no paired contemporaneous warmed full-forward timing')
 save(out/'forward.json',common)
 del raw;raw=None
 # Numerical/fixed-support RGB checks are separate from quality/timing.
 del model,state,images,raw;gc.collect();torch.cuda.empty_cache()
 save(out/'status.json',dict(stage='quality',scene=a.scene))
 metrics={};errors={}
 try:
  _,gt=load_scene_data(dataset,'eth3d',a.scene,128,'uniform',seed=0)
  for task in ['pose','recon_unposed']:
   try:
    np.random.seed(0);o3d.utility.random.seed(0);metrics.update(evaluate(dataset,gt,pred,[task]))
   except Exception as exc:errors[task]=f'{type(exc).__name__}: {exc}'
 except Exception as exc:errors['ground_truth']=f'{type(exc).__name__}: {exc}'
 q=quality_record(metrics,'; '.join(f'{k}: {v}' for k,v in errors.items()) or 'Task unavailable')
 if q['complete']:assert abs(q['values']['cd']-(q['values']['acc']+q['values']['comp'])/2)<1e-6
 quality(out,q);save(out/'result.json',dict(common,quality=q,metrics=metrics,quality_errors=errors));save(out/'status.json',dict(stage='complete' if q['complete'] else 'quality_incomplete'))
 print('QUALITY',a.scene,q,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['centered050'],default='centered050');p.add_argument('--scene',required=True);p.add_argument('--expected-input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:run(a)
 except BaseException as exc:
  if a.output.exists():quality(a.output,quality_record({},f'{type(exc).__name__}: {exc}'));save(a.output/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
  raise
