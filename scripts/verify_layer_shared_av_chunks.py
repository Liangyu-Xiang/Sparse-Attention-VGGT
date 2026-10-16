"""Controlled real128-frame trunk check: identical masks/states across chunks."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np
from scripts.overlay_vggt_query_attention import build_model, run_to_global_layer
from saf3r.patch.layer_shared_av import install_layer_shared_av
from saf3r.utils.data_utils import load_and_process_images_vggt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader = SevenScenes(); reader.data_root = str(ROOT/'workspace/benchmark_dataset/7scenes')
    data = reader.get_data('chess')
    positions = np.linspace(0, len(data.image_files)-1, 128, dtype=int)
    images = load_and_process_images_vggt([data.image_files[i] for i in positions]).cuda()[None]
    model = build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt', 'cuda')
    state = install_layer_shared_av(model, sampling=.0025, dominance=.9, logit_scale=32.)
    results = {}
    current = []
    def after(module, inputs, output):
        current.append(dict(layer=module.layer, output=digest(output),
            history=None if state.history is None else digest(state.history),
            fine_labels=digest(state.fine[0]), representatives=digest(state.fine[1])))
    for block in model.aggregator.global_blocks[:7]: block.attn.register_forward_hook(after)
    for chunk in [128, 1024]:
        state.query_chunk = chunk
        state.reset(128, images.shape[-2]//14, images.shape[-1]//14)
        current = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            run_to_global_layer(model, images, 6)
        torch.cuda.synchronize()
        results[str(chunk)] = current
        print(chunk, current, flush=True)
    equal = results['128'] == results['1024']
    target = ROOT/'outputs/layer_shared_av/chunk_invariance_v3.json'
    target.write_text(json.dumps(dict(equal=equal, frames=128, layers=list(range(7)),
        algorithm_sha256=hashlib.sha256((ROOT/'saf3r/patch/layer_shared_av.py').read_bytes()).hexdigest(),
        fingerprints=results), indent=2)+'\n')
    if not equal: raise AssertionError('Chunk changed actual per-query execution/state')


if __name__ == '__main__': main()
