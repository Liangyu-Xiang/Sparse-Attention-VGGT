"""ScanNet50 DINO Top-64 paper-mechanism reproduction; 300/500/700 frames."""
import os,sys,json,time,traceback,subprocess,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
DATA=Path('/mnt/rfs/users/xiangliangyu/datasets')
IM=DATA/'scannet50_processed_full_unpacked/scannet50_processed'
GT=DATA/'scannet50_gt_meshes_unpacked/scannet-dataset'
FRAMES=int(os.environ.get('SCANNET_FRAMES','300'))
OUT=Path(os.environ.get('SCANNET_OUTPUT',str(ROOT/f'outputs/scannet50_dino64_{FRAMES}_20261029')))
FIELDS=['auc03','auc30','acc','comp','cd','f1']
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2)); q.replace(p)
def table(rows):
 return '| Scene/method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+''.join('| '+name+' | '+' | '.join('N/A' if v[k] is None else f'{v[k]*(100 if k in ("auc03","auc30","f1") else 1):.6f}' for k in FIELDS)+' | '+('N/A' if speed is None else f'{speed:.4f}x')+' |\n' for name,v,speed in rows)
def run(scene,method,frames=FRAMES):
 import torch,yaml,open3d as o3d
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
 from benchmarks.da3.bench.utils import compute_pose,evaluate_3d_reconstruction
 from benchmarks.vggt.geometry import unproject_depth_map_to_point_map
 out=OUT/scene/method;out.mkdir(parents=True,exist_ok=True)
 result=dict(scene=scene,method=method,frames=frames,values=dict.fromkeys(FIELDS),speedup=None,speedup_unavailable='Paired timings not yet available',reason='Run in progress')
 save(out/'result.json',result)
 try:
  torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  valid=[];poses=[]
  for p in sorted((IM/scene/'color').glob('*'),key=lambda p:int(p.stem)):
   pose=np.loadtxt(IM/scene/'pose'/f'{p.stem}.txt')
   if np.isfinite(pose).all():valid.append(str(p));poses.append(pose)
  idx=np.linspace(0,len(valid)-1,min(frames,len(valid)),dtype=int)
  paths=[valid[i] for i in idx];gt=np.asarray(poses)[idx]
  cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/scannet50_paper_dino64.yaml').read_text())['model'])
  if method=='dense':cfg.pop('patch_module');cfg.pop('sparse_config_path')
  cfg.ckpt_path=str(ROOT/cfg.ckpt_path)
  save(out/'input.json',dict(paths=paths,checkpoint=cfg.ckpt_path,frames=len(paths)))
  save(out/'config.json',dict(cfg))
  model=build_model(cfg,'cuda');images=load_images(model,'vggt',paths).cuda()
  if method=='saf3r':
   from saf3r.patch import update_args_per_forward
   update_args_per_forward(model,'vggt',*images.shape[-2:])
  save(out/'input.json',dict(paths=paths,shape=list(images.shape),checkpoint=cfg.ckpt_path))
  save(out/'config.json',dict(cfg))
  times=[];peaks=[];raw=None
  for i in range(4):
   del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   if method=='saf3r':update_args_per_forward(model,'vggt',*images.shape[-2:])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if i:times.append(elapsed);peaks.append(torch.cuda.max_memory_allocated()/2**30)
   print(scene,method,'forward',i,elapsed,flush=True)
  result.update(timing=dict(warmup=1,repeats=3,seconds=times,median_seconds=float(np.median(times)),peak_allocated_gib=peaks,scope='complete model forward including routing and initialization per forward'),hardware=dict(gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__))
  save(out/'result.json',result)
  ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:]);ext=ext.reshape(-1,3,4).float().cpu().numpy();intr=intr.reshape(-1,3,3).float().cpu().numpy()
  depth=raw['depth'].reshape(len(paths),*images.shape[-2:]).float().cpu().numpy();conf=raw['depth_conf'].reshape(depth.shape).float().cpu().numpy()
  del model,images,raw;torch.cuda.empty_cache()
  pred=np.repeat(np.eye(4)[None],len(paths),axis=0);pred[:,:3]=ext
  pose_metrics=compute_pose(torch.tensor(pred,dtype=torch.float64),torch.tensor(np.linalg.inv(gt),dtype=torch.float64))
  from benchmarks.vggt.fastvggt_utils import evaluate_scene_and_save,align_point_clouds_scale
  all_points=[]
  for i in range(len(paths)):
   points=unproject_depth_map_to_point_map(depth[i:i+1],ext[i:i+1],intr[i:i+1]).reshape(-1,3)
   mask=np.isfinite(points).all(1)&(conf[i].reshape(-1)>=1.0)&(depth[i].reshape(-1)>0)
   all_points.append(points[mask])
  first_pose=np.loadtxt(IM/scene/'pose'/f'{Path(valid[0]).stem}.txt')
  np.random.seed(0)
  legacy=evaluate_scene_and_save(scene,gt,first_pose,[int(Path(p).stem) for p in paths],list(pred),all_points,str(GT),.5,result['timing']['median_seconds'],False,str(out/'trajectory.png'))
  # Required six metrics on same FastVGGT bbox alignment and sampled/downsampled support.
  merged=np.vstack(all_points);del all_points
  world=merged@first_pose[:3,:3].T+first_pose[:3,3];del merged
  truth=np.asarray(o3d.io.read_point_cloud(str(GT/scene/f'{scene}_vh_clean_2.ply')).points)
  world,scale=align_point_clouds_scale(world,truth)
  np.random.seed(0)
  if len(world)>100000:world=world[np.random.choice(len(world),100000,replace=False)]
  if len(truth)>100000:truth=truth[np.random.choice(len(truth),100000,replace=False)]
  cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world)).voxel_down_sample(.05)
  target=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(truth)).voxel_down_sample(.05)
  da=np.clip(np.asarray(cloud.compute_point_cloud_distance(target)),0,.5)
  dc=np.clip(np.asarray(target.compute_point_cloud_distance(cloud)),0,.5)
  if not len(da) or not len(dc):raise ValueError('Empty reconstruction/GT')
  precision=float(np.mean(da<.05));recall=float(np.mean(dc<.05))
  recon=dict(acc=float(da.mean()),comp=float(dc.mean()),overall=float((da.mean()+dc.mean())/2),fscore=2*precision*recall/(precision+recall) if precision+recall else 0.,precision=precision,recall=recall)
  result['fastvggt_metrics']=legacy
  result['values']=dict(auc03=float(pose_metrics.auc03),auc30=float(pose_metrics.auc30),acc=float(recon['acc']),comp=float(recon['comp']),cd=float(recon['overall']),f1=float(recon['fscore']))
  result.update(raw_metrics=dict(pose=dict(pose_metrics),recon_unposed=recon),alignment=dict(scale=scale,protocol='FastVGGT first GT pose then point-cloud bbox scale/translation'),reason=None)
  if method=='dense':result['speedup']=1.;result['speedup_unavailable']=None
  else:
   baseline=json.loads((OUT/scene/'dense/result.json').read_text())
   if json.loads((OUT/scene/'dense/input.json').read_text())!=json.loads((out/'input.json').read_text()):raise ValueError('Input mismatch')
   if baseline.get('timing'):
    result['speedup']=baseline['timing']['median_seconds']/result['timing']['median_seconds'];result['speedup_unavailable']=None
   else:result['speedup_unavailable']='Paired Dense forward failed: '+str(baseline['reason'])
 except Exception:
  result['reason']=traceback.format_exc();result['values']=dict.fromkeys(FIELDS);result['failure_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;print(result['reason'],flush=True)
 save(out/'result.json',result);(out/'quality.md').write_text(table([(scene+'/'+method,result['values'],result['speedup'])])+('\n'+result['reason'] if result['reason'] else ''))
 return result['reason'] is None

def report():
 rows=[];results=[]
 for p in sorted(OUT.glob('scene*/*/result.json')):
  r=json.loads(p.read_text());results.append(r);rows.append((r['scene']+'/'+r['method'],r['values'],r['speedup']))
 for method in ['dense','saf3r']:
  rs=[r for r in results if r['method']==method and r['reason'] is None]
  if len(rs)==50:rows.append(('MACRO/'+method,{k:float(np.mean([r['values'][k] for r in rs])) for k in FIELDS},float(np.mean([r['speedup'] for r in rs]))))
 save(OUT/'summary.json',results)
 (OUT/'report.md').write_text(f'Local paper-mechanism reproduction attempt: {FRAMES} frames, DINO feature per-frame Top-64, max_topk=44800 (no cap for <=700 frames). ETH3D calibrated profile. FastVGGT first-pose + bbox cloud alignment; seeded 100000 point sampling then 0.05m voxel; distances clipped to 0.5m. Six metrics: CD=(ACC+Comp)/2 unsquared metres, F1 threshold 0.05m. Original FastVGGT summed CD and ATE/ARE/RPE are separate raw fields. 1 warmup + 3 complete forwards including DINO index computation. Macro only with 50 successful scenes; OOM runs are unavailable with reasons. H100 paper timing is not a local baseline. Per-scene K unpublished: fixed K64 is declared, not exact paper Table 4 settings.\n\n'+table(rows))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--method');p.add_argument('--frames',type=int,default=FRAMES);p.add_argument('--worker',type=int);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.worker is not None:
  scenes=(DATA/'scannet_50.yaml').read_text().split()
  for scene in scenes[a.worker::8]:
   for method in ['dense','saf3r']:
    dest=OUT/scene/method;dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.log').open('w') as log:subprocess.run([sys.executable,__file__,'--scene',scene,'--method',method,'--frames',str(FRAMES)],stdout=log,stderr=subprocess.STDOUT)
   import fcntl
   with (OUT/"report.lock").open("w") as lock:
    fcntl.flock(lock,fcntl.LOCK_EX);report()
 elif a.scene:sys.exit(0 if run(a.scene,a.method,a.frames) else 1)
 else:report()
