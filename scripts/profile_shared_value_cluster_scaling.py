"""CUDA Event stages and warmed forward scaling for Shared Value Loss (7Scenes heads).

From the cloned repository root, after installing dependencies, checkpoint and data:
CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_shared_value_cluster_scaling \
    --frames 128 --clusters 10 --repeats 3 --output outputs/local_scaling/10_128

For accurate measurements, reserve the physical GPU (one CUDA program at a time).
Uses 20% Key Retention for ALL cluster counts. Extra profiling pass after timers.
Geometry metrics are NOT evaluated here: use evaluate_zdrop_history_states.py.
"""
import argparse, json, time, traceback, sys
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser()
p.add_argument('--frames',type=int,required=True,help='Uniformly sampled frames of 7Scenes heads')
p.add_argument('--clusters',type=int,choices=(20,10,5),required=True)
p.add_argument('--output',type=Path,required=True,help='A new, non-existing output directory')
p.add_argument('--repeats',type=int,default=3,help='Number of warmed timed forwards')
a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'evaluation'),str(root/'scripts')]
import torch, numpy as np, yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.da3bench_utils import sample_frames

def dump(name,obj):
 (out/name).write_text(json.dumps(obj,indent=2,default=str)+'\n')

try:
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dataset=MV_REGISTRY.get('7scenes')()
 indices,scene=sample_frames(dataset.get_data('heads'),'heads',a.frames,sample_mode='uniform',seed=0)
 dump('input.json',dict(scene='heads',frames=len(scene.image_files),indices=indices,image_paths=scene.image_files))
 config_path=root/('configs/evaluation/reuse_shared_value_loss_256.json' if a.clusters==20 else f'configs/evaluation/reuse_shared_value_loss_clusters{a.clusters}.json')
 config=ReuseExploreConfig(**json.loads(config_path.read_text()))
 dump('config.json',json.loads(config_path.read_text()))
 model_conf=Dict(yaml.safe_load((root/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
 model_conf.pop('patch_module');model_conf.pop('sparse_config_path')
 model_conf.ckpt_path=str(root/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(model_conf,'cuda')
 images=load_images(model,'vggt',scene.image_files).cuda()
 s=install_reuse_residual_exploration(model,config)
 torch.cuda.synchronize()
 t=[]
 for rep in range(a.repeats+1):
  torch.cuda.reset_peak_memory_stats()
  start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   raw=model(images)
  torch.cuda.synchronize()
  sample=dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
  t.append(sample)
  print('TIMING',a.clusters,a.frames,rep,json.dumps(sample),flush=True)
  del raw
 result=dict(clusters=a.clusters,frames=a.frames,device_name=torch.cuda.get_device_name(),
  physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),timing=t,
  warmed_median_seconds=float(np.median([x['seconds'] for x in t[1:]])),
  max_peak_gib=max(x['peak_allocated_gib'] for x in t),
  max_geometry_width=max(x['geometry_width'] for x in s.records[1:]),
  avg_actual_density=float(np.mean([float(x['actual_unique_density']) for x in s.records[1:]])))
 dump('timing.json',result)
 s.profiling=True;s.profile_events.clear()
 torch.cuda.synchronize()
 aev=torch.cuda.Event(enable_timing=True);bev=torch.cuda.Event(enable_timing=True)
 aev.record();wall=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  raw=model(images)
 bev.record();torch.cuda.synchronize()
 result['profile_wall_seconds']=time.perf_counter()-wall
 result['profile_cuda_interval_seconds']=aev.elapsed_time(bev)/1000
 stages=defaultdict(float);calls=defaultdict(int)
 for layer,name,beg,end in s.profile_events:
  stages[name]+=beg.elapsed_time(end)/1000;calls[name]+=1
 result['stages_seconds']=dict(sorted(stages.items(),key=lambda kv:-kv[1]))
 result['stage_calls']=dict(calls)
 result['outside_named_stages']=result['profile_cuda_interval_seconds']-sum(stages.values())
 result['profile_note']='Non-overlapping CUDA Event stages; overlap/queue scheduling and CPU gaps included. Do not interpret stage intervals as kernel self time.'
 dump('result.json',result)
 print('SUCCESS',a.clusters,a.frames,'median',result['warmed_median_seconds'],'profile',result['profile_cuda_interval_seconds'],'stages',result['stages_seconds'],flush=True)
except Exception as e:
 dump('failure.json',dict(error=str(e),type=type(e).__name__,traceback=traceback.format_exc()))
 print('FAIL',a.clusters,a.frames,type(e).__name__,str(e),flush=True)
 raise
