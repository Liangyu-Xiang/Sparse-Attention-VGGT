"""Paired real-QKV test of fused selected-probe LSE versus raw-only extraction.

Both variants compute identical indexed Attention and exported raw logits.
The raw-only variant recomputes selected LSE from saved logits instead of
updating tile-wise selected LogSumExp inside every Attention tile.
"""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images

def median_gpu_ms(fn,reps=7):
    for _ in range(2):fn()
    ts=[]
    for _ in range(reps):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();fn();b.record();b.synchronize();ts.append(a.elapsed_time(b))
    return dict(median_ms=statistics.median(ts),samples_ms=ts)

def rel_l2(a,b):
    return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))

def main():
    p=argparse.ArgumentParser()
    for name in ('config','input','output_dir'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model,s=build(a.config)
    images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
    import saf3r.triton.reuse_light_attention as mod
    original=mod.weighted_attention;rows=[];seen=set()
    def hooked(q,k,v,ids,bias,special,scale,**kw):
        baseline=original(q,k,v,ids,bias,special,scale,**kw)
        layer=s.previous_layer+1
        if layer not in (0,12,23) or layer in seen:return baseline
        seen.add(layer);row=dict(layer=layer,methods={})
        for probes in (1,4):
            control_kw={**kw,'probe_count':probes,'emit_evidence':True,'fast_evidence':False}
            reference,ref_info=original(q,k,v,ids,bias,special,scale,**control_kw)
            for fast in (False,True):
                opts={**control_kw,'fast_evidence':fast}
                output,info=original(q,k,v,ids,bias,special,scale,**opts)
                max_abs_lse=float((info['selected_lse']-ref_info['selected_lse']).abs().nan_to_num().max())
                max_abs_special=float((info['special_lse']-ref_info['special_lse']).abs().nan_to_num().max())
                assert rel_l2(output,reference)<1e-5
                assert max_abs_lse<0.003,(layer,probes,fast,max_abs_lse)
                assert max_abs_special<0.003
                assert rel_l2(info['raw'],ref_info['raw'])<1e-6
                def fn():
                    return original(q,k,v,ids,bias,special,scale,**opts)
                row['methods'][f'probe{probes}_'+('rawonly' if fast else 'fused')]=dict(
                    **median_gpu_ms(fn),rel_l2_output=rel_l2(output,reference),
                    max_abs_selected_lse=max_abs_lse,max_abs_special_lse=max_abs_special)
        rows.append(row);print(json.dumps(row),flush=True)
        return baseline
    mod.weighted_attention=hooked
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
    finally:mod.weighted_attention=original
    result=dict(complete=len(rows)==3,rows=rows,gpu=torch.cuda.get_device_name(),
                quality=None,scope='Fixed real QKV/mask; timing includes LSE reduction')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
