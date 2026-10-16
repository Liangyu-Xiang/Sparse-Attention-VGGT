"""Online mean-V PAM key selection in VGGT's first nine global layers.

Uses the existing full FP32 PAM unchanged. No approximations or frame-size
fallbacks. A live-memory guard stops the experiment before a known OOM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.visualize_vggt_value_medoids import pairwise_distances, pam, best_swap


class MemoryPreflightError(RuntimeError):
    pass


def token_indices(frames, patches_per_frame, special_per_frame, device):
    ids = torch.arange(frames * (patches_per_frame + special_per_frame), device=device)
    special = ids.remainder(patches_per_frame + special_per_frame) < special_per_frame
    return ids[~special], ids[special]


def mean_patch_values(v, patches):
    # Mean RAW V in FP32, before pooling/normalisation or medoid selection.
    return v[0, :, patches, :].float().mean(dim=0)


def selected_attention(q, k, v, patches, specials, medoids):
    """All queries survive; special queries/keys remain dense."""
    keys = torch.cat((specials, patches[medoids])).sort().values
    patch_output = F.scaled_dot_product_attention(
        q[:, :, patches], k[:, :, keys], v[:, :, keys], dropout_p=0.0)
    # VGGT RoPE can promote Q/K to FP32 while autocast SDPA returns BF16.
    # Match the actual attention result, not Q's promoted dtype.
    output = torch.empty(q.shape,device=q.device,dtype=patch_output.dtype)
    output[:, :, patches] = patch_output
    output[:, :, specials] = F.scaled_dot_product_attention(
        q[:, :, specials], k, v, dropout_p=0.0)
    return output


def check_pam_memory(n, device, report_path, metadata):
    """Current eager best_swap needs up to four simultaneous FP32 NxN arrays."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    matrix = 4 * n * n
    # Small O(NK)/index/reduction outputs and allocator/workspace headroom.
    # 4*N*N is the dominant live tensor peak, not a guaranteed total bound.
    required = 4 * matrix
    # Keep a minimal allocator margin; the full-size swap smoke check below
    # measures actual feasibility instead of rejecting a marginal fit based
    # solely on an arbitrary large safety allowance.
    margin = 64 * 2**20
    stats = {**metadata, "patch_tokens": n, "single_distance_bytes": matrix,
             "dominant_pam_peak_bytes": required, "headroom_bytes": margin,
             "free_bytes_before_pam": free, "gpu_total_bytes": total,
             "live_torch_bytes": torch.cuda.memory_allocated(device),
             "passed": required + margin <= free}
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print("PAM显存检查：" + json.dumps(stats, ensure_ascii=False), flush=True)
    if not stats["passed"]:
        raise MemoryPreflightError(
            f"PAM主要矩阵峰值{required/2**30:.2f} GiB + "
            f"{margin/2**30:.2f} GiB余量 > 当前可用{free/2**30:.2f} GiB")


class MeanValueMedoidAttention(nn.Module):
    def __init__(self, original, frames, patches_per_frame, layer, output, max_swaps=3000):
        super().__init__()
        self.original = original
        self.frames, self.patches_per_frame = frames, patches_per_frame
        self.layer, self.output, self.max_swaps = layer, Path(output), max_swaps

    def forward(self, x, pos=None):
        if self.training or x.shape[0] != 1:
            raise ValueError("Only inference with batch size 1 is supported")
        start = time.perf_counter()
        b, n, c = x.shape
        m = self.original
        assert n == self.frames * (self.patches_per_frame + 5)
        qkv = m.qkv(x).reshape(b, n, 3, m.num_heads, m.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = m.q_norm(q), m.k_norm(k)
        if m.rope is not None:
            q, k = m.rope(q, pos), m.rope(k, pos)
        patches, specials = token_indices(self.frames, self.patches_per_frame, 5, x.device)
        values = mean_patch_values(v, patches)
        count = round(0.25 * len(patches))
        directory = self.output / f"layer_{self.layer:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        if x.is_cuda:
            check_pam_memory(len(patches), x.device, directory / "memory_preflight.json",
                             {"global_layer_1based": self.layer + 1, "retained_keys": count})
        # Explicitly disable outer VGGT BF16 autocast: all PAM matrix products
        # and swap costs must remain FP32, exactly as in the prior analysis.
        with torch.autocast(device_type=x.device.type, enabled=False):
            distance = pairwise_distances(values)
            if x.is_cuda:
                print(f"第{self.layer+1}层：完整规模交换搜索显存冒烟检查", flush=True)
                # Diagnostic only: do not use this arbitrary set as the PAM
                # initialisation or as attention support. Both original
                # starts below remain entirely unchanged.
                best_swap(distance, torch.arange(count, device=x.device))
                torch.cuda.synchronize(x.device)
                memory = {"global_layer_1based": self.layer + 1,
                          "full_size_swap_smoke_passed": True,
                          "peak_allocated_gib": torch.cuda.max_memory_allocated(x.device)/2**30}
                (directory / "memory_smoke.json").write_text(json.dumps(memory, indent=2) + "\n")
                print("交换搜索显存检查通过：" + json.dumps(memory), flush=True)
            starts = []
            def progress(event):
                event = {"global_layer_1based":self.layer+1,"elapsed_seconds":time.perf_counter()-start,**event}
                temporary = directory/"progress.json.tmp"
                temporary.write_text(json.dumps(event,ensure_ascii=False,indent=2)+"\n")
                temporary.replace(directory/"progress.json")
                print("PAM进度："+json.dumps(event,ensure_ascii=False),flush=True)
            for method in ("build", "sample"):
                print(f"第{self.layer+1}层：开始完整PAM {method}，N={len(values)}，K={count}", flush=True)
                starts.append(pam(distance, count, method, 20261015 + 100 * self.layer, self.max_swaps, progress=progress))
            result = min(starts, key=lambda item: item["objective"])
            if not result["converged"]:
                raise RuntimeError("PAM未收敛，停止，不将未收敛结果当作最终代表点")
            medoids = torch.as_tensor(result["medoids"], device=x.device, dtype=torch.long)
        np.savez_compressed(directory / "clusters.npz", **{
            name: result[name] for name in ("medoids", "assignments", "distances")})
        stats = {name: item for name, item in result.items()
                 if name not in ("medoids", "assignments", "distances")}
        stats.update(retention=0.25, n=len(values), k=count, heads_averaged=m.num_heads,
                     starts=[{name: item for name, item in run.items()
                              if name not in ("medoids", "assignments", "distances")}
                             for run in starts])
        (directory / "clustering.json").write_text(json.dumps(stats, indent=2) + "\n")
        del distance, values, starts
        if x.is_cuda:
            torch.cuda.empty_cache()
        output = selected_attention(q, k, v, patches, specials, medoids)
        output = m.proj_drop(m.proj(output.transpose(1, 2).reshape(b, n, c)))
        print(f"第{self.layer+1}层完成，累计耗时{time.perf_counter()-start:.1f}秒", flush=True)
        return output


def main():
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--scenes", nargs="+", default=["chess", "fire", "heads"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "workspace/benchmark_dataset/7scenes/7Scenes")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/vggt/model_tracker_fixed_e20.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/value_medoid_sparse_50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--status-file", type=Path, help="Independent status path for parallel scene workers")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    status_file = args.status_file or args.output / "status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status = {"status": "initializing", "frames": args.frames, "retention": .25,
              "global_layers_1based": list(range(1, 10)), "heads": "mean raw V then shared keys",
              "algorithm": "unchanged full FP32 PAM, BUILD + sample starts",
              "scenes": args.scenes, "completed_scenes": [], "actual_oom_triggered": False,
              "pid": os.getpid(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def save_status():
        temporary = status_file.with_suffix(status_file.suffix + ".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status_file)

    save_status()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = build_model(args.checkpoint, args.device)
        originals = [model.aggregator.global_blocks[i].attn for i in range(9)]
        for scene in args.scenes:
            directory = args.output / scene
            directory.mkdir(exist_ok=True)
            sequence = args.data_root / scene / ("seq-02" if scene == "stairs" else "seq-01")
            eligible = [sequence / f"frame-{i:06d}.color.png"
                        for i in range(500 if scene == "stairs" else 1000)
                        if (sequence / f"frame-{i:06d}.color.png").is_file()
                        and (sequence / f"frame-{i:06d}.pose.txt").is_file()]
            if len(eligible) < args.frames:
                raise RuntimeError(f"{scene}帧数不足，不自动缩短输入")
            paths = [eligible[i] for i in np.linspace(0, len(eligible)-1, args.frames, dtype=np.int64)]
            images = load_and_process_images_vggt(list(map(str, paths))).to(args.device)
            height, width = images.shape[-2:]
            (directory / "input.json").write_text(json.dumps({"paths": list(map(str, paths)), "shape": list(images.shape)}, indent=2) + "\n")
            for layer, original in enumerate(originals):
                model.aggregator.global_blocks[layer].attn = MeanValueMedoidAttention(
                    original, args.frames, (height//14)*(width//14), layer, directory).eval()
            status.update(status="running", current_scene=scene)
            save_status()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"], (height, width))
            np.savez_compressed(directory / "poses.npz", extrinsics=extrinsics.float().cpu().numpy(),
                                intrinsics=intrinsics.float().cpu().numpy())
            sys.path.insert(0, str(ROOT / "evaluation"))
            from benchmarks.da3.bench.utils import compute_pose
            from benchmarks.da3.geometry import as_homogeneous
            gt = np.stack([np.linalg.inv(np.loadtxt(str(p).replace(".color.png", ".pose.txt"))) for p in paths])
            metrics = compute_pose(torch.from_numpy(as_homogeneous(extrinsics[0].float().cpu().numpy())),
                                   torch.from_numpy(gt))
            result = {"scene": scene, "frames": args.frames, "latency_seconds": elapsed,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                      "pose": {name: float(value) for name, value in metrics.items()}}
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            status["completed_scenes"].append(result)
            save_status()
            del images, prediction
            torch.cuda.empty_cache()
        status["status"] = "completed"
    except MemoryPreflightError as error:
        status.update(status="stopped_memory_preflight", reason=str(error))
        print("显存不足预警，按照用户要求停止：" + str(error), flush=True)
    except torch.cuda.OutOfMemoryError as error:
        status.update(status="stopped_oom", actual_oom_triggered=True, reason=str(error))
        print("实际CUDA OOM，按照用户要求停止，不降低帧数或修改算法。", flush=True)
    except Exception as error:
        status.update(status="failed", reason=repr(error))
        raise
    finally:
        status["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        save_status()


if __name__ == "__main__":
    main()
