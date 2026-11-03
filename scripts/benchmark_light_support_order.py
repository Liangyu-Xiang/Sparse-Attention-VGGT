"""Real fixed-mask locality test, including sort/gather cost, no quality GT."""
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
 normal=module.weighted_attention;seen=set();rows=[]
 def hooked(q,k,v,ids,bias,special,scale,**kw):
  baseline=normal(q,k,v,ids,bias,special,scale,**kw);layer=s.previous_layer+1
  if layer in [0,12,23] and layer not in seen:
   seen.add(layer);samples={}
   for ordering in ['cluster','token']:
    values=[]
    for i in range(7):
     x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record()
     if ordering=='token':
      sorted_ids,order=ids.masked_fill(ids<0,len(k[0])).sort(dim=1,stable=True);sorted_ids=sorted_ids.masked_fill(sorted_ids>=len(k[0]),-1);bb=bias.gather(1,order).contiguous()
     else:sorted_ids=ids;bb=bias
     out=normal(q,k,v,sorted_ids,bb,special,scale,**kw);y.record();y.synchronize()
     if i>=2:values.append(x.elapsed_time(y))
    error=float((baseline[0].float()-out[0].float()).norm()/baseline[0].float().norm());assert error<.005
    samples[ordering]=dict(median_ms=statistics.median(values),relative_l2=error)
   row=dict(layer=layer,samples=samples);rows.append(row);print(json.dumps(row),flush=True)
  return baseline
 module.weighted_attention=hooked
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Fixed-mask locality execution microbenchmark only',dense_speedup=None,dense_reason='No Dense model rerun');(a.output_dir/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
