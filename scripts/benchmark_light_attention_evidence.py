"""Fixed-real-QKV sparse attention evidence ablation (no evaluation GT).

Benchmarks one real window each at layers 0/12/23 with identical Q/K/V,
selected support and logit bias. The native dense Flash baseline uses the
same Q/K/V, but all K/V and no sparse compensation bias.

Run under a single-GPU lease; results are NOT end-to-end speedups.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "evaluation")]
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images


def measure(fn, warmup=2, repeats=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gpu_ms, wall_ms = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        gpu_ms.append(start.elapsed_time(end))
        wall_ms.append((time.perf_counter() - before) * 1000)
    return dict(gpu_median_ms=statistics.median(gpu_ms),
                wall_median_ms=statistics.median(wall_ms),
                gpu_samples_ms=gpu_ms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, state = build(args.config)
    images = load_images(model, "vggt", json.loads(args.input.read_text())["image_paths"]).cuda()
    import saf3r.triton.reuse_light_attention as module
    original = module.weighted_attention
    seen, rows = set(), []

    def hooked(q, k, v, ids, bias, specials, scale, **kwargs):
        # Return the original output to the model; benchmarking is diagnostic only.
        original_out = original(q, k, v, ids, bias, specials, scale, **kwargs)
        layer = state.previous_layer + 1
        if layer not in args.layers or layer in seen:
            return original_out
        seen.add(layer)
        row = dict(layer=layer, shape_q=list(q.shape), shape_k=list(k.shape),
                   selected_keys=ids.shape[-1], groups=ids.shape[0],
                   special_keys=specials.numel(), configurations={})
        baseline = None
        for label, probe_count, evidence in [
            ("sparse_pure", 1, False),
            ("sparse_probe1", 1, True),
            ("sparse_probe4", 4, True),
        ]:
            options = dict(kwargs)
            options.update(probe_count=probe_count, emit_evidence=evidence)

            def fn(options=options):
                return original(q, k, v, ids, bias, specials, scale, **options)[0]

            out = fn().float()
            if baseline is None:
                baseline = out
            ratio = float((out - baseline).norm() / baseline.norm().clamp_min(1e-12))
            row["configurations"][label] = dict(**measure(fn), relative_l2_vs_pure=ratio)

        def dense():
            # Native full-key Flash Attention: same q,k,v; no sparse logit bias.
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    dropout_p=0., is_causal=False)
        try:
            row["configurations"]["dense_flash"] = measure(dense)
        except Exception as exc:
            row["configurations"]["dense_flash"] = dict(error=repr(exc))
        rows.append(row)
        print(json.dumps(row), flush=True)
        return original_out

    module.weighted_attention = hooked
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(images)
    finally:
        module.weighted_attention = original

    result = dict(complete=len(rows) == len(args.layers),
                  gpu=torch.cuda.get_device_name(),
                  scope="same-real-QKV fixed-support microbenchmark; no quality GT",
                  rows=rows, quality={k: None for k in
                                   ["auc03", "auc30", "acc", "comp", "cd", "f1"]},
                  dense_end_to_end_speedup=None)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
