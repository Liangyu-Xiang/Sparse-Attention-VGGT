"""FastVGGT BF16-aligned matched Dense / frozen fast_subgroup evaluation."""
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.quality_metrics import quality_record,quality_markdown,quality_cells,FIELDS,LABELS
SCENES=['chess','fire','heads','office','pumpkin','redkitchen','stairs']
METHODS=['dense','fast_subgroup']
CONFIG=ROOT/'configs/evaluation/beta_unified_subgroup_20261109.json'
PROFILE=ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json'
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=lambda v:v.detach().cpu().tolist() if hasattr(v,'detach') else v.item() if hasattr(v,'item') else str(v))+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(out,r):
 save(out/'result.json',r);save(out/'quality.json',r['quality']);(out/'quality.md').write_text(quality_markdown(r['quality']))
def worker(a):
 out=a.output/a.dataset/str(a.frames)/a.scene/a.method
 started=time.monotonic()
 r=dict(dataset=a.dataset,scene=a.scene,requested_frames=a.frames,method=a.method,status='loading',quality=quality_record({},'Run in progress'),timing=None)
 record(out,r)
 try:
  import types
  ns=types.ModuleType('benchmarks');ns.__path__=[str(ROOT/'evaluation/benchmarks')];sys.modules['benchmarks']=ns
  import torch,numpy as np,open3d as o3d
  from addict import Dict
  from saf3r.utils.model_utils import build_model
  from saf3r.utils.data_utils import load_images
  from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
  from benchmarks.da3.bench.utils import compute_pose,evaluate_3d_reconstruction
  torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  dataset=None
  if a.dataset=='7scenes':
   from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
   from benchmarks.da3.da3bench_utils import sample_frames
   from evaluate_da3bench import load_scene_data,evaluate
   dataset=SevenScenes();dataset.data_root=str(a.data/'7scenes/extracted/7scenes')
   full=dataset.get_data(a.scene)
   if len(full.image_files)<a.frames:raise ValueError(f'Requested {a.frames} unique frames, standard sequence has {len(full.image_files)}; no duplication or silent reduction')
   indices,scene=sample_frames(full,a.scene,a.frames,sample_mode='uniform',seed=0);paths=scene.image_files
   protocol=dict(tasks=['pose','recon_unposed'],pose='DA3 compute_pose',alignment='existing DA3 unposed alignment/TSDF',threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,voxel_length_m=dataset.voxel_length,sdf_trunc_m=dataset.sdf_trunc,mesh_samples=dataset.sampling_number,max_depth=dataset.max_depth)
  else:
   base=a.data/'scannet50_processed'/a.scene;valid=[];poses=[]
   for p in sorted((base/'color').glob('*'),key=lambda p:int(p.stem)):
    pose=np.loadtxt(base/'pose'/f'{p.stem}.txt')
    if np.isfinite(pose).all():valid.append(str(p));poses.append(pose)
   if not valid:raise ValueError('No valid ScanNet frames')
   indices=list(range(len(valid)))
   if len(valid)>a.frames:
    step=max(1,(len(valid)-1)//(a.frames-1));indices=[0]+list(range(1,len(valid),step))[:a.frames-1]
   paths=[valid[i] for i in indices];gt=np.asarray(poses)[indices]
   protocol=dict(tasks=['pose','recon_unposed'],pose='DA3 compute_pose',alignment='single camera-centre Sim3, predicted depth and poses; no ICP',threshold_m=.05,downsample_m=.05,fusion='incremental 0.05m voxel every 8 frames',confidence_min=1.,truth='GT clean mesh vertices; local matched protocol, not published paper scores',sampling_source='https://github.com/mystorm16/FastVGGT/blob/main/vggt/utils/eval_utils.py#L551',short_sequence='use all available valid frames; input_frame is an upper bound')
  checkpoint=a.checkpoint.resolve();input_data=dict(image_paths=paths,frame_indices=[int(i) for i in indices],source_frame_ids=[int(Path(p).stem) for p in paths] if a.dataset=='scannet50' else None,checkpoint=str(checkpoint),checkpoint_sha256=a.checkpoint_sha256,preprocessing='repository load_images vggt',sampling='7Scenes uniform; ScanNet FastVGGT first+remaining integer stride, keep all when short',seed=0)
  cfg=Dict(name='VGGT',dpt_only=True,vram_efficient_bf16=True,ckpt_path=str(checkpoint))
  protocol['inference_precision']='BF16 weights, special tokens, residual/retained features and prediction-head autocast; point/track disabled'
  if a.method=='saf3r_dino128':cfg.update(sparse_config_path=str(PROFILE),patch_module=Dict(type='headsparse',topk_mode='token',lazy_dino_topk=False,topk=128,per_frame_topk=True,max_topk=len(paths)*128,include_first_frame=False))
  r.update(actual_frames=len(paths),input=input_data,protocol=protocol,model_config=dict(cfg),frozen_config=json.loads(CONFIG.read_text()) if a.method=='fast_subgroup' else None,saf3r=dict(topk=128,source='DINO final features, per frame',max_topk=len(paths)*128,profile=str(PROFILE),profile_sha256=sha(PROFILE),official_default=False) if a.method=='saf3r_dino128' else None,hardware=dict(gpu=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(0).uuid),torch=torch.__version__,cuda=torch.version.cuda,open3d=o3d.__version__))
  record(out,r)
  model=build_model(cfg,'cuda')
  assert model.point_head is None and model.track_head is None
  assert {p.dtype for p in model.parameters()}=={torch.bfloat16}
  assert model.aggregator.save_intermediates is False, 'VGGT* requires dropping unused intermediate outputs'
  assert list(model.aggregator.DPT_BLOCKS)==[4,11,17,23], 'VGGT* retained layers changed'
  r['vggt_vram_efficient']=dict(enabled=True,retained_encoder_layers=[4,11,17,23],retained_layers_count=4,reference='FastVGGT paper section3.5 and official eval/model BF16 implementation',weight_dtype='bfloat16',retained_dtype='bfloat16',point_head=False,track_head=False,checkpoint_unchanged=True,attention='Dense for baseline; no token merging')
  images=load_images(model,'vggt',paths).cuda();r['input']['shape']=list(images.shape)
  if a.method=='fast_subgroup':
   from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
   from saf3r.patch.beta_fast_subgroup_forward import install
   install()
   tile = os.environ.get("FAST_SUBGROUP_QUERY_TILE")
   if tile is not None:
    from saf3r.triton import beta_subgroup_reuse_attention as subgroup
    kernel = subgroup.attention
    def attention_with_tiles(*args, **kwargs):
     args = list(args)
     args[10] = int(os.environ.get('FAST_SUBGROUP_WARPS', 4))
     args[11] = int(os.environ.get('FAST_SUBGROUP_STAGES', 2))
     return kernel(*args, block_queries=int(tile), block_keys=int(os.environ.get('FAST_SUBGROUP_KEY_TILE', 128)), **kwargs)
    subgroup.attention = attention_with_tiles
   state=install_reuse_residual_exploration(model,ReuseExploreConfig(**r['frozen_config']))
  else:state=None
  if a.method=='saf3r_dino128':from saf3r.patch import update_args_per_forward
  samples=[];cold=[];raw=None
  for i in range(4):
   del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
   if a.method=='saf3r_dino128':update_args_per_forward(model,'vggt',*images.shape[-2:])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
   torch.cuda.synchronize();s=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   (cold if i==0 else samples).append(s);print('FORWARD',i,s,flush=True)
  r['timing']=dict(warmup=1,repeats=3,median_seconds=float(np.median([s['seconds'] for s in samples])),samples=samples,cold_samples=cold,scope='full model forward including reset, clustering, probability probes, routing, history, reordering and prediction heads; loading excluded')
  record(out,r)
  if state is not None:save(out/'layers.json',getattr(state,'records',[]))
  if a.diagnostic:
   r.update(status='diagnostic_complete',quality=quality_record({},'8-frame implementation preflight; not a 300/500/1000-frame quality evaluation'));return True
  ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:]);ext=ext.reshape(-1,3,4).float().cpu().numpy();intr=intr.reshape(-1,3,3).float().cpu().numpy()
  depth=raw['depth'].reshape(len(paths),*images.shape[-2:]).float().cpu().numpy();conf=raw['depth_conf'].reshape(depth.shape).float().cpu().numpy()
  processed=images.float().cpu().permute(0,2,3,1).numpy()
  del raw,model,images,state;__import__('gc').collect();torch.cuda.empty_cache()
  metrics={};errors={}
  if dataset is not None:
   _,truth=load_scene_data(dataset,'7scenes',a.scene,a.frames,'uniform',seed=0)
   pred=Dict(extrinsics=ext,intrinsics=intr,depth=depth,conf=conf,processed_images=processed)
   for task in ('pose','recon_unposed'):
    try:
     np.random.seed(0);o3d.utility.random.seed(0);metrics.update(evaluate(dataset,truth,pred,[task]))
    except Exception:errors[task]=traceback.format_exc()
  else:
   from benchmarks.vggt.geometry import unproject_depth_map_to_point_map
   pred=np.repeat(np.eye(4)[None],len(paths),axis=0);pred[:,:3]=ext
   try:metrics['pose']=dict(compute_pose(torch.tensor(pred,dtype=torch.float64),torch.tensor(np.linalg.inv(gt),dtype=torch.float64)))
   except Exception:errors['pose']=traceback.format_exc()
   try:
    pc=np.linalg.inv(pred)[:,:3,3];gc=gt[:,:3,3];x=pc-pc.mean(0);y=gc-gc.mean(0);u,s,vt=np.linalg.svd(y.T@x/len(x));d=np.ones(3);d[-1]=np.linalg.det(u@vt);rot=u@np.diag(d)@vt;scale=float(np.sum(s*d)/np.mean(np.sum(x*x,axis=1)));trans=gc.mean(0)-scale*rot@pc.mean(0)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate alignment')
    cloud=o3d.geometry.PointCloud()
    for i in range(len(paths)):
     points=unproject_depth_map_to_point_map(depth[i:i+1],ext[i:i+1],intr[i:i+1]).reshape(-1,3);mask=np.isfinite(points).all(1)&(conf[i].reshape(-1)>=1)&(depth[i].reshape(-1)>0);points=scale*points[mask]@rot.T+trans
     cloud+=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).voxel_down_sample(.05)
     if i%8==7:cloud=cloud.voxel_down_sample(.05)
    truth=o3d.io.read_point_cloud(str(a.data/'scannet50_gt/scannet-dataset'/a.scene/f'{a.scene}_vh_clean_2.ply'))
    if not len(truth.points) or not len(cloud.points):raise ValueError('Empty prediction/GT cloud')
    metrics['recon_unposed']=evaluate_3d_reconstruction(cloud,truth,threshold=.05,down_sample=.05)
    r['alignment']=dict(scale=scale,rotation=rot.tolist(),translation=trans.tolist())
   except Exception:errors['recon_unposed']=traceback.format_exc()
  if 'recon_unposed' in metrics:metrics['recon_unposed']['overall']=(float(metrics['recon_unposed']['acc'])+float(metrics['recon_unposed']['comp']))/2
  r.update(metrics=metrics,errors=errors,quality=quality_record(metrics,str(errors) or 'Metric unavailable'))
  if a.method=='dense':r['quality'].update(speedup_vs_vggt=1.,speedup_unavailable=None)
  else:
   bp=out.parent/'dense/result.json';b=json.loads(bp.read_text()) if bp.exists() else {}
   if b.get('timing') and all(b.get(k)==r.get(k) for k in ('input','hardware','protocol')):r['quality'].update(speedup_vs_vggt=b['timing']['median_seconds']/r['timing']['median_seconds'],speedup_unavailable=None)
   else:r['quality']['speedup_unavailable']='No contemporaneous paired Dense timing; unchanged Dense reused for quality only'
  r['status']='complete' if r['quality']['complete'] else 'quality_incomplete'
 except Exception:
  r.update(status='failed',error=traceback.format_exc(),quality=quality_record({},traceback.format_exc()))
 finally:
  r['wall_seconds']=time.monotonic()-started
  try:
   r['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
  except Exception:pass
  record(out,r)
 return r['quality']['complete']
def report(out):
 rows=[];macros=[]
 for p in sorted(out.glob('*/*/*/*/result.json')):
  r=json.loads(p.read_text());rows.append(r)
 for ds in ('scannet50',):
  for n in (500,700):
   for method in (['dense'] if n==500 else METHODS):
    rs=[r for r in rows if (r['dataset'],r['requested_frames'],r['method'])==(ds,n,method)];required=7 if ds=='7scenes' else 50;complete=[r for r in rs if r['quality']['complete']]
    q=quality_record({},f'Complete scenes {len(complete)}/{required}; all scenes required for dataset macro')
    if len(complete)==required:
     q.update(values={k:sum(r['quality']['values'][k] for r in complete)/required for k in FIELDS},complete=True,unavailable={})
    macros.append(dict(dataset=ds,requested_frames=n,method=method,complete_scenes=len(complete),required_scenes=required,quality=q,mean_scene_median_seconds=sum(r['timing']['median_seconds'] for r in rs)/required if len(rs)==required and all(r.get('timing') for r in rs) else None))
 for r in macros:
  b=next(x for x in macros if (x['dataset'],x['requested_frames'],x['method'])==(r['dataset'],r['requested_frames'],'dense'))
  if r['mean_scene_median_seconds'] and b['mean_scene_median_seconds'] and all(x['quality'].get('speedup_vs_vggt') is not None for x in rows if (x['dataset'],x['requested_frames'],x['method'])==(r['dataset'],r['requested_frames'],r['method'])):r['quality'].update(speedup_vs_vggt=b['mean_scene_median_seconds']/r['mean_scene_median_seconds'],speedup_unavailable=None)
 for r in rows+macros:
  candidates=rows if 'scene' in r else macros
  b=next((x for x in candidates if (x['dataset'],x['requested_frames'],x.get('scene'),x['method'])==(r['dataset'],r['requested_frames'],r.get('scene'),'dense')),None)
  r['delta_vs_matched_dense']={k:r['quality']['values'][k]-b['quality']['values'][k] for k in FIELDS} if b and r['quality']['complete'] and b['quality']['complete'] else None
 def table(items):
  lines=['| Dataset | Frames | Scene | Method | '+ ' | '.join(LABELS)+' | Actual frames | Warm s | Peak GiB | Speedup | Status |','|---|---:|---|---|'+'---:|'*10+'---|']
  for r in items:
   t=r.get('timing');sec=t['median_seconds'] if t else r.get('mean_scene_median_seconds');peak=max(x['peak_allocated_gib'] for x in t['samples']) if t else None;speed=r['quality'].get('speedup_vs_vggt')
   lines.append('| '+' | '.join([r['dataset'],str(r['requested_frames']),r.get('scene',f"MACRO {r.get('complete_scenes')}/{r.get('required_scenes')}"),r['method']]+quality_cells(r['quality'])+[str(r.get('actual_frames','—')),'N/A' if sec is None else f'{sec:.4f}','N/A' if peak is None else f'{peak:.3f}','N/A' if speed is None else f'{speed:.3f}x',r.get('status','complete' if r['quality']['complete'] else 'incomplete')])+' |')
  return lines
 save(out/'summary.json',dict(macro=macros,per_scene=rows))
 (out/'report.md').write_text('\n'.join(['# Local matched long-sequence evaluation (FastVGGT maximum-frame sampling)','','Source b1f12ba8 plus audited BF16 alignment and 64-bit probability offsets; VGGT* memory-efficient baseline retains only encoder layers4/11/17/23 (FastVGGT section3.5), not all24; frozen fast_subgroup; SAF3R remains stopped. New precision batch: never combine its timings/quality with earlier mixed-precision batches; fast_subgroup at500frames remains stopped. No test-truth tuning. One warmup + three full forwards, same assigned physical GPU per matched cohort. Speedup = Dense / method; macro speedup = ratio of scene-mean medians. AUC/F1 percentages, distances metres, CD=(ACC+Comp)/2 unsquared. All scenes required for dataset quality macros. ScanNet frame counts are maxima: shorter valid sequences use every available frame, following FastVGGT. Actual counts are recorded per scene; no duplication. See raw result.json for unavailable reasons, timing samples, hardware, input and protocol.','','## Dataset macro','']+table(macros)+['','## Per scene','']+table(rows))+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--diagnostic',action='store_true');p.add_argument('--report-only',action='store_true');p.add_argument('--dataset',choices=['scannet50']);p.add_argument('--scene');p.add_argument('--method',choices=METHODS);p.add_argument('--frames',type=int);p.add_argument('--data',type=Path,default=Path('/tmp/vggt-eval-data'));p.add_argument('--checkpoint',type=Path,default=Path('/tmp/vggt-checkpoints/model_tracker_fixed_e20.pt'));p.add_argument('--checkpoint-sha256');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.report_only:report(a.output)
 elif a.worker:sys.exit(0 if worker(a) else 1)
 else:p.error('Use dispatch_bf16_700_sequences.py to launch the frozen campaign')
