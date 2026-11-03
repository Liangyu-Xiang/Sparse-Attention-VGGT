"""Real fixed-QKV fixed-mask Split-K ablation on an exclusive GPU.

Compare parts=1/2/4/8 with and without 1-probe evidence. The model
still returns its original configured Attention; only timing is diagnostic.
"""
import argparse, json, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images

def timing(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(7):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize()
        ts.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(ts),samples_ms=ts)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--layers',type=int,nargs='+',default=[0,12,23])
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model,s=build(a.config)
    images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
    import saf3r.triton.reuse_light_attention as mod
    normal=mod.weighted_attention
    rows=[];seen=set()
    def hooked(q,k,v,ids,bias,special,scale,**kw):
        old=normal(q,k,v,ids,bias,special,scale,**kw)
        layer=s.previous_layer+1
        if layer in a.layers and layer not in seen:
            seen.add(layer)
            ref=old[0].float()
            row=dict(layer=layer,support=ids.shape[1],rows=[])
            for evidence in (False,True):
                for parts in (1,2,4,8):
                    opts={**kw,'parts':parts,'emit_evidence':evidence,'probe_count':1}
                    def fn():return normal(q,k,v,ids,bias,special,scale,**opts)[0]
                    output=fn().float()
                    error=float((output-ref).norm()/ref.norm().clamp_min(1e-12))
                    tm=timing(fn)
                    row['rows'].append(dict(parts=parts,evidence=evidence,
                                            rel_l2_to_reference=error,**tm))
            rows.append(row);print(json.dumps(row),flush=True)
        return old
    mod.weighted_attention=hooked
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            model(images)
    finally:
        mod.weighted_attention=normal
    result=dict(complete=len(rows)==len(a.layers),gpu=torch.cuda.get_device_name(),
                rows=rows,scope='Real QKV fixed mask fixed bias kernel ablation',
                quality=None)
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
