"""Real Q/K/V tile tuning only; no geometric ground truth accessed."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_light_reuse import build
from saf3r.utils.data_utils import load_images

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 model,s=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
 import saf3r.triton.reuse_light_attention as module
 original=module.weighted_attention;rows=[];seen=set()
 def hooked(q,k,v,ids,bias,special,scale,**kw):
  out=original(q,k,v,ids,bias,special,scale,**kw);layer=s.previous_layer+1
  if layer in [0,12,23] and layer not in seen:
   seen.add(layer);samples={}
   for parts in [4,8]:
    for warps in [4,8]:
     bk=64
     values=[]
     for i in range(5):
      start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();x=original(q,k,v,ids,bias,special,scale,**{**kw,'parts':parts,'block_keys':bk,'warps':warps});end.record();end.synchronize()
      if i>=2:values.append(start.elapsed_time(end))
     error=float((x[0].float()-out[0].float()).norm()/out[0].float().norm());assert error<.005,error
     samples[f'{parts}x{bk}w{warps}']=dict(milliseconds=statistics.median(values),relative_l2=error)
   rows.append(dict(layer=layer,samples=samples));print(json.dumps(rows[-1]),flush=True)
  return out
 module.weighted_attention=hooked
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='No-GT real-window execution tuning only',dense_speedup=None,dense_reason='No Dense rerun by user request');(a.output_dir/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
