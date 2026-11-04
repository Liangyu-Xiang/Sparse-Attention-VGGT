import argparse,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation'),str(ROOT/'scripts')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from grid_local_scopes import Recorder

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();c=ReuseExploreConfig(**json.loads(a.config.read_text()));s=install_reuse_residual_exploration(model,c)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
    torch.cuda.synchronize();rec=Recorder(s);rec.install()
    import importlib
    backend=importlib.import_module('saf3r.triton.'+('reuse_serial_sparseprob' if c.execution_kernel=='serial_sparseprob' else 'reuse_serial_tuned'))
    rec.wrap(backend,'weighted_attention','attention_wrapper')
    class Launch:
        def __init__(self,kernel,label):self.kernel=kernel;self.label=label
        def __getitem__(self,grid):
            call=self.kernel[grid]
            def run(*args,**kwargs):
                with rec.scope(self.label):return call(*args,**kwargs)
            return run
    for name,label in [('_attention','attention_core_fused_grid'),('_merge','attention_merge'),('_reduce','attention_grid_normalize_reduce')]:
        if hasattr(backend,name):setattr(backend,name,Launch(getattr(backend,name),label))
    for b in model.aggregator.global_blocks:
        rec.wrap(b.attn.original.proj,'forward','output_projection');rec.wrap(b.attn.original.proj_drop,'forward','output_dropout')
    torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();wall=time.perf_counter()-begin;rows,groups,total,unique=rec.finish()
    result=dict(wall_seconds=wall,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,global_attention_seconds=total,groups=groups,events=rows,sum_error_seconds=unique-total,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Timing-only profile; complete geometry quality evaluated separately',dense_speedup=None,dense_reason='User requested no Dense rerun')
    (a.output_dir/'profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(wall_seconds=wall,groups=groups),indent=2))

if __name__=='__main__':main()
