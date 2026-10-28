"""ScanNet-50: public non-lazy DINO64 configuration and user six metrics.

Paper-compatible public configuration, NOT recovery of unpublished scene-wise K.
Production model/selector code is not modified by this evaluator.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.quality_metrics import quality_record
from scripts.evaluate_attention_first_residual_value import save,write_quality

DATA=Path('/data_NVME1/mmc_lyxiang/Dataset/scannet50_processed')
GT=Path('/data_SSD1/mmc_lyxiang/dataset/scannet50_data/extracted/scannet-dataset')
CHECKPOINT=Path('/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt')
PROFILE=ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json'
PROTOCOL=dict(sampling='uniform over RGB frames with valid finite poses',seed=0,
    confidence_threshold=1.0,reconstruction='unposed depth unprojection with predicted cameras; no GT-camera fusion',
    alignment='FastVGGT first-camera gauge followed by GT/pred bounding-box diagonal scale and center alignment; no ICP',
    point_cap=100000,downsample_m=0.05,distance_clip_m=0.5,f1_threshold_m=0.05,
    cd_definition='(ACC+Comp)/2 unsquared clipped distances; official FastVGGT sum saved separately',
    auc='DA3-Bench relative rotation/translation angular AUC, all distinct camera pairs',
    warmup=1,repeats=3,scope='full forward including DINO Top-K construction and caching',
    parameter_provenance='Paper 300/500/700 uniform; public non-lazy DINO Top64/max2048 example, not unpublished per-scene K reproduction')

def distance_metrics(a,b,clip=.5,threshold=.05):
    import numpy as np
    if not len(a) or not len(b):raise ValueError('Empty reconstructed or GT point cloud')
    acc=float(np.minimum(a,clip).mean());comp=float(np.minimum(b,clip).mean())
    precision=float((a<threshold).mean());recall=float((b<threshold).mean())
    return dict(acc=acc,comp=comp,overall=(acc+comp)/2,
        precision=precision,recall=recall,fscore=2*precision*recall/max(precision+recall,1e-30),
        fastvggt_sum_cd=acc+comp)

def select(scene,frames):
    from benchmarks.vggt.fastvggt_utils import load_poses,get_sorted_image_paths,build_frame_selection
    base=DATA/scene
    poses,first,valid=load_poses(str(base/'pose'))
    if poses is None:raise ValueError('No valid poses')
    ids,paths,indices=build_frame_selection(get_sorted_image_paths(str(base/'color')),valid,frames,'uniform',5)
    if len(paths)<2:raise ValueError('Too few valid input frames')
    mesh=GT/scene/(scene+'_vh_clean_2.ply')
    if not mesh.is_file():raise FileNotFoundError(mesh)
    return ids,paths,poses[indices],first,mesh

def reconstruction(depth,conf,extr,intr,first,mesh,out):
    import numpy as np
    import open3d as o3d
    from benchmarks.vggt.geometry import depth_to_world_coords_points
    from benchmarks.vggt.fastvggt_utils import align_point_clouds_scale
    points=[]
    for i in range(len(depth)):
        world,_,_=depth_to_world_coords_points(depth[i],extr[i],intr[i])
        valid=(conf[i]>=PROTOCOL['confidence_threshold']) & np.isfinite(world).all(-1) & (depth[i]>0)
        points.append(world[valid])
    pred=np.concatenate(points);del points
    pred=pred@first[:3,:3].T+first[:3,3]
    target=np.asarray(o3d.io.read_point_cloud(str(mesh)).points)
    if not len(pred) or not len(target):raise ValueError('Empty point cloud before alignment')
    pred,scale=align_point_clouds_scale(pred,target)
    rng=np.random.RandomState(0)
    cap=PROTOCOL['point_cap']
    if len(pred)>cap:pred=pred[rng.choice(len(pred),cap,replace=False)]
    if len(target)>cap:target=target[rng.choice(len(target),cap,replace=False)]
    clouds=[]
    for values in (pred,target):
        cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(values))
        clouds.append(cloud.voxel_down_sample(PROTOCOL['downsample_m']))
    a=np.asarray(clouds[0].compute_point_cloud_distance(clouds[1]))
    b=np.asarray(clouds[1].compute_point_cloud_distance(clouds[0]))
    np.savez_compressed(out/'reconstruction_distances.npz',pred_to_gt=a,gt_to_pred=b)
    metrics=distance_metrics(a,b)
    metrics.update(scale=float(scale),pred_points=len(a),gt_points=len(b))
    return metrics

def run(args):
    import numpy as np
    import open3d as o3d
    import torch
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from benchmarks.da3.bench.utils import compute_pose
    out=args.output_dir
    if out.exists():raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True)
    write_quality(out,quality_record({},'Run in progress'))
    save(out/'status.json',dict(stage='loading'))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids,paths,c2ws,first,mesh=select(args.scene,args.frames)
    manifest_path=out.parents[3]/'input_manifest.json'
    if manifest_path.exists():
        expected=json.loads(manifest_path.read_text())[f'{args.frames}/{args.scene}']
        if expected['frame_ids']!=ids or expected['image_paths']!=paths:
            raise ValueError('Input differs from frozen campaign frame manifest')
    cfg=Dict(name='VGGT',dpt_only=True,ckpt_path=str(CHECKPOINT))
    if args.method=='saf3r':
        cfg.sparse_config_path=str(PROFILE)
        cfg.patch_module=Dict(type='headsparse',topk_mode='token',lazy_dino_topk=False,
            topk=64,max_topk=2048,per_frame_topk=True,include_first_frame=False)
    save(out/'config.json',dict(model=dict(cfg),protocol=PROTOCOL))
    sources=['scripts/evaluate_scannet_paper64.py','scripts/run_scannet_paper64.py',
        'saf3r/models/official_vggt/models/vggt.py','saf3r/models/official_vggt/models/aggregator.py',
        'saf3r/patch/patch_vggt.py','saf3r/patch/utils.py','saf3r/utils/data_utils.py',
        'evaluation/benchmarks/vggt/fastvggt_utils.py','evaluation/benchmarks/da3/bench/utils.py',
        'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    hashes={}
    for name in sources:
        data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    save(out/'source_sha256.json',hashes)
    model=build_model(cfg,'cuda')
    images=load_images(model,'vggt',paths).cuda()
    if args.method=='saf3r':
        from saf3r.patch import update_args_per_forward
        update_args_per_forward(model,'vggt',*images.shape[-2:])
    inp=dict(image_paths=paths,frame_ids=ids,shape=list(images.shape),checkpoint=str(CHECKPOINT),gt_mesh=str(mesh))
    save(out/'input.json',inp)
    hardware=dict(gpu=torch.cuda.get_device_name(),uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,cuda=torch.version.cuda)
    raw=None;samples=[];cold=[]
    for iteration in range(4):
        del raw;raw=None
        save(out/'status.json',dict(stage='forward',iteration=iteration))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (cold if iteration==0 else samples).append(sample)
        print('FORWARD',args.scene,args.frames,args.method,iteration,sample,flush=True)
    timing=dict(median_seconds=float(np.median([x['seconds'] for x in samples])),samples=samples,cold=cold,warmup=1,repeats=3)
    common=dict(scene=args.scene,dataset='scannet50',requested_frames=args.frames,frames=len(paths),method=args.method,
        input=inp,hardware=hardware,protocol=PROTOCOL,timing=timing,source_sha256=hashes)
    save(out/'forward.json',common)
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    ext=ext[0].float().cpu().numpy();intr=intr[0].float().cpu().numpy()
    depth=raw['depth'][0,...,0].float().cpu().numpy();conf=raw['depth_conf'][0].float().cpu().numpy()
    del model,images,raw;gc.collect();torch.cuda.empty_cache()
    save(out/'status.json',dict(stage='quality'))
    np.savez_compressed(out/'poses.npz',pred_w2c=ext,pred_intrinsics=intr,gt_c2w=c2ws,frame_ids=ids)
    metrics={};errors={}
    try:
        pred=np.broadcast_to(np.eye(4),(len(ext),4,4)).copy();pred[:,:3]=ext
        metrics['pose']=dict(compute_pose(torch.from_numpy(pred).float(),torch.from_numpy(np.linalg.inv(c2ws)).float()))
    except Exception as e:errors['pose']=f'{type(e).__name__}: {e}'
    try:metrics['recon_unposed']=reconstruction(depth,conf,ext,intr,first,mesh,out)
    except Exception as e:errors['recon_unposed']=f'{type(e).__name__}: {e}'
    q=quality_record(metrics,'; '.join(errors.values()) or 'Metric unavailable')
    if args.method=='dense':q.update(speedup_vs_vggt=1.,speedup_unavailable=None)
    save(out/'result.json',dict(common,quality=q,metrics=metrics,quality_errors=errors));write_quality(out,q)
    save(out/'status.json',dict(stage='complete' if q['complete'] else 'quality_incomplete'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['dense','saf3r'],required=True)
    p.add_argument('--scene',required=True);p.add_argument('--frames',type=int,choices=[300,500,700],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        a.output_dir.mkdir(parents=True,exist_ok=True)
        reason=f'{type(e).__name__}: {e}'
        save(a.output_dir/'failure.json',dict(reason=reason,traceback=traceback.format_exc(),method=a.method,requested_frames=a.frames,scene=a.scene))
        write_quality(a.output_dir,quality_record({},reason))
        save(a.output_dir/'status.json',dict(stage='failed',error=reason))
        raise
