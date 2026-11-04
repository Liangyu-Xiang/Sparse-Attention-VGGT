"""Capture fixed production inputs; standalone operators never feed the model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from scripts.direct_detail_microbench import run
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
class Fixed:pass
rec=Fixed();rec.state=state;rec.state_snapshot=None;rec.snapshot=None
import saf3r.patch.reuse_window_graph as window
original=window.Window.__call__
def capture(instance,*args,**kwargs):
 result=original(instance,*args,**kwargs)
 if rec.state_snapshot is None:rec.state_snapshot=instance
 return result
window.Window.__call__=capture
import saf3r.triton.reuse_direct_sparse_attention as backend
attention=backend.direct_sparse_attention
def selected(*args,**kwargs):
 if rec.snapshot is None and state.previous_layer+1==12:rec.snapshot=(args,kwargs)
 return attention(*args,**kwargs)
backend.direct_sparse_attention=selected
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();run(rec,a.output_dir);print('COMPONENTS_COMPLETE',flush=True)
