"""Capture matched depths, then CPU-only 7Scenes recon_unposed evaluation.

Value-PAM replay uses the exact completed online supports, not another PAM fit.
Replay/capture timings never replace original online method timings.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))


def method_root(comparison, value_root, method):
    return value_root if method == "value_pam" else comparison / method


def validate_support(medoids, n, k):
    ids = np.asarray(medoids)
    if (ids.shape != (k,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != k
            or np.any(ids < 0) or np.any(ids >= n)):
        raise ValueError("Invalid or incomplete saved Value-PAM support")
    return torch.from_numpy(ids.astype(np.int64))


class ReplayMedoidAttention(nn.Module):
    def __init__(self, original, frames, patches_per_frame, medoids):
        super().__init__()
        self.original = original
        self.frames, self.patches_per_frame = frames, patches_per_frame
        self.register_buffer("medoids", medoids, persistent=False)

    def forward(self, x, pos=None):
        from scripts.run_value_medoid_sparse import token_indices, selected_attention
        b, n, c = x.shape
        if self.training or b != 1 or n != self.frames*(self.patches_per_frame+5):
            raise ValueError("Replay layout or inference mode mismatch")
        m = self.original
        qkv = m.qkv(x).reshape(b,n,3,m.num_heads,m.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv.unbind(0)
        q,k = m.q_norm(q),m.k_norm(k)
        if m.rope is not None:
            q,k = m.rope(q,pos),m.rope(k,pos)
        patches,specials = token_indices(self.frames,self.patches_per_frame,5,x.device)
        output = selected_attention(q,k,v,patches,specials,self.medoids.to(x.device))
        return m.proj_drop(m.proj(output.transpose(1,2).reshape(b,n,c)))


def check_replay_poses(reference_path, extrinsics, intrinsics):
    with np.load(reference_path) as saved:
        ref_ext = saved["extrinsics"].reshape(extrinsics.shape)
        ref_intr = saved["intrinsics"].reshape(intrinsics.shape)
    # Require a faithful replay before accepting geometry from a second pass.
    np.testing.assert_allclose(extrinsics,ref_ext,rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(intrinsics,ref_intr,rtol=1e-5,atol=1e-3)
    return {"extrinsics_max_abs":float(np.max(np.abs(extrinsics-ref_ext))),
            "intrinsics_max_abs":float(np.max(np.abs(intrinsics-ref_intr)))}


def capture(args, manifest):
    from addict import Dict
    from omegaconf import OmegaConf
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from scripts.report_value_sparse_comparison import matched_input

    cfg_file = "eval_saf3r_vggt.yaml" if args.method == "saf3r" else "eval_vggt.yaml"
    cfg = Dict(OmegaConf.to_container(OmegaConf.load(ROOT/"configs/evaluation"/cfg_file),resolve=True)["model"])
    model = build_model(cfg,"cuda").eval()
    for scene in args.scenes:
        entry = manifest["scenes"][scene]
        source = method_root(args.comparison,args.value_root,args.method)/scene
        if not (source/"result.json").is_file():
            raise RuntimeError(f"{args.method}/{scene}: original online result not complete")
        images = load_and_process_images_vggt(entry["paths"]).to("cuda")
        expected = {"paths":entry["paths"],"shape":list(images.shape)}
        if not matched_input(expected,json.loads((source/"input.json").read_text())):
            raise ValueError("Original and reconstruction capture inputs differ")
        if args.method == "saf3r":
            from saf3r.patch import update_args_per_forward
            update_args_per_forward(model,"vggt",*images.shape[-2:])
        elif args.method == "value_pam":
            n = len(images)*(images.shape[-2]//14)*(images.shape[-1]//14)
            for layer in range(9):
                directory = source/f"layer_{layer:02d}"
                stats = json.loads((directory/"clustering.json").read_text())
                if not stats["converged"] or stats["n"] != n or stats["k"] != round(.25*n):
                    raise ValueError("Missing converged online Value-PAM fit")
                with np.load(directory/"clusters.npz") as saved:
                    medoids = validate_support(saved["medoids"],n,round(.25*n))
                # Capture is normally one scene/process. Unwrap when a caller
                # explicitly captures multiple Value scenes in one process.
                original = model.aggregator.global_blocks[layer].attn
                if isinstance(original,ReplayMedoidAttention):
                    original = original.original
                model.aggregator.global_blocks[layer].attn = ReplayMedoidAttention(
                    original,len(images),n//len(images),medoids).eval()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            pred = model(images)
        ext,intr = pose_encoding_to_extri_intri(pred["pose_enc"],images.shape[-2:])
        ext_np,intr_np = ext[0].float().cpu().numpy(),intr[0].float().cpu().numpy()
        checks = check_replay_poses(source/"poses.npz",ext_np,intr_np)
        directory = args.comparison/"reconstruction"/args.method/scene
        directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/"prediction.npz",depth=pred["depth"][0,...,0].float().cpu().numpy(),
                            extrinsics=ext_np,intrinsics=intr_np)
        (directory/"capture.json").write_text(json.dumps({"method":args.method,"scene":scene,
            "paths":entry["paths"],"shape":list(images.shape),"pose_replay_check":checks,
            "value_supports_replayed_without_reclustering":args.method=="value_pam",
            "not_original_online_latency":True},indent=2)+"\n")
        print(f"深度已保存：{args.method}/{scene}，重放位姿检查{checks}",flush=True)
        del pred,images,ext,intr
        torch.cuda.empty_cache()


def strict_sample(mesh, n, seed=42):
    import open3d as o3d
    if len(mesh.vertices)==0 or len(mesh.triangles)==0:
        raise ValueError("Empty reconstruction/GT mesh: refusing random fallback points")
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError("Non-finite mesh vertices")
    o3d.utility.random.seed(seed)
    cloud = mesh.sample_points_uniformly(number_of_points=n)
    if len(cloud.points)!=n or not np.isfinite(np.asarray(cloud.points)).all():
        raise ValueError("Invalid sampled cloud")
    return cloud


def score(args, manifest):
    import cv2
    import open3d as o3d
    from addict import Dict
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import create_tsdf_volume,fuse_depth_to_tsdf,evaluate_3d_reconstruction
    dataset = SevenScenes()
    cache = args.comparison/"reconstruction/gt_cache"
    cache.mkdir(parents=True,exist_ok=True)
    for scene in args.scenes:
        start = time.perf_counter()
        entry = manifest["scenes"][scene]
        directory = args.comparison/"reconstruction"/args.method/scene
        metadata = json.loads((directory/"capture.json").read_text())
        if metadata["paths"]!=entry["paths"]:
            raise ValueError("Reconstruction input manifest mismatch")
        paths = entry["paths"]
        depth_files = [p.replace(".color.png",".depth.png") for p in paths]
        if not all(Path(p).is_file() for p in depth_files):
            raise FileNotFoundError("Missing GT depth validity masks")
        ext_gt = np.stack([np.linalg.inv(np.loadtxt(p.replace(".color.png",".pose.txt"))) for p in paths]).astype(np.float32)
        gt = Dict(dataset_name="7scenes",extrinsics=ext_gt,image_files=paths,
                  aux=Dict(gt_depth_files=depth_files))
        with np.load(directory/"prediction.npz") as saved:
            pred = Dict({name:saved[name] for name in ("depth","extrinsics","intrinsics")})
        images = [cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB) for p in paths]
        shapes = [im.shape[:2] for im in images]
        depths,intrinsics,extrinsics = dataset._prep_unposed(pred,gt,shapes,scene=scene)
        volume = create_tsdf_volume(voxel_length=dataset.voxel_length,sdf_trunc=dataset.sdf_trunc)
        mesh = fuse_depth_to_tsdf(volume,depths,np.stack(images),intrinsics,extrinsics,max_depth=dataset.max_depth)
        # Same sampling protocol as fuse3d_7scenes, but forbid its helper's
        # random-cube fallback on sampling failure/invalid meshes.
        predicted = strict_sample(mesh,dataset.sampling_number)
        mesh_path = Path(paths[0]).parents[2]/"meshes"/f"{scene}.ply"
        with (cache/f"{scene}.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            cached = cache/f"{scene}.ply"
            if not cached.exists():
                truth_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                truth = strict_sample(truth_mesh,dataset.sampling_number)
                if not o3d.io.write_point_cloud(str(cached),truth):
                    raise RuntimeError("Could not save GT reference cloud")
            truth = o3d.io.read_point_cloud(str(cached))
            if len(truth.points)!=dataset.sampling_number:
                raise ValueError("GT cache point count mismatch")
        metrics = evaluate_3d_reconstruction(predicted,truth,threshold=dataset.eval_threshold,
                                             down_sample=dataset.down_sample)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Invalid reconstruction metrics")
        o3d.io.write_triangle_mesh(str(directory/"reconstruction_mesh.ply"),mesh)
        o3d.io.write_point_cloud(str(directory/"reconstruction_points.ply"),predicted)
        result = {"method":args.method,"scene":scene,"frames":manifest["frames"],
                  "recon_unposed":metrics,"threshold_m":dataset.eval_threshold,
                  "voxel_length_m":dataset.voxel_length,"sdf_trunc_m":dataset.sdf_trunc,
                  "evaluation_downsample_m":dataset.down_sample,"sampled_points":dataset.sampling_number,
                  "alignment":"predicted poses + Umeyama/RANSAC scale and rigid alignment to GT; GT depth validity masks; not GT posed fusion",
                  "gt_cloud":str(cached),"postprocessing_seconds":time.perf_counter()-start,
                  "pose_replay_check":metadata["pose_replay_check"]}
        temporary = directory/"result.json.tmp"
        temporary.write_text(json.dumps(result,indent=2)+"\n")
        temporary.replace(directory/"result.json")
        print(json.dumps(result),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method",choices=("dense","saf3r","value_pam"),required=True)
    parser.add_argument("--stage",choices=("capture","score"),required=True)
    parser.add_argument("--scenes",nargs="+",default=["chess","fire","heads"])
    parser.add_argument("--comparison",type=Path,default=ROOT/"outputs/compare_value_sparse_50")
    parser.add_argument("--value-root",type=Path,default=ROOT/"outputs/value_medoid_sparse_50")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.comparison/"inputs.json").read_text())
    (capture if args.stage=="capture" else score)(args,manifest)


if __name__=="__main__":
    main()
