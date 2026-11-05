"""Fresh-process patch-prefix precision diagnostic; no GT and no mask selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--autotune',type=int,choices=[0,1],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=bool(a.autotune)
cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();agg=model.aggregator
capture={}
def hook(module,args,result):capture['patch_projection']=result[[0,-1]].clone()
handle=agg.patch_embed.patch_embed.proj.register_forward_hook(hook)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    x=(images[None]-agg._resnet_mean)/agg._resnet_std
    result=agg.patch_embed(x.reshape(-1,*images.shape[1:]))
    capture['prefix_tokens']=result['x_norm_patchtokens'][[0,-1]].clone()
handle.remove();torch.cuda.synchronize();torch.save({k:v.cpu() for k,v in capture.items()},a.output/'features.pt')
meta=dict(autotune=bool(a.autotune),shape=list(images.shape),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Fresh-process prefix numerical diagnostic, no geometry',speedup_vs_dense=None,speedup_reason='Prefix diagnostic only')
(a.output/'result.json').write_text(json.dumps(meta,indent=2)+'\n');print('DONE',flush=True)
