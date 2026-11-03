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
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
        torch.cuda.synchronize()
    groups={};intervals=[]
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:continue
        value=groups.setdefault(event.name,dict(calls=0,milliseconds=0.))
        value['calls']+=1;value['milliseconds']+=event.time_range.elapsed_us()/1000
        intervals.append((event.time_range.start,event.time_range.end))
    intervals.sort();merged=[]
    for first,last in intervals:
        if merged and first<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],last)
        else:merged.append([first,last])
    unique=sum(last-first for first,last in merged)/1000
    total=sum(v['milliseconds'] for v in groups.values())
    result=dict(complete=True,kernels=groups,sum_kernel_milliseconds=total,union_device_interval_milliseconds=unique,overlap_milliseconds=total-unique,timing_scope='Profiler-instrumented CUDA activity diagnostic, not end-to-end speed measurement',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='CUDA activity profiling only',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    for name,value in sorted(groups.items(),key=lambda kv:-kv[1]['milliseconds'])[:25]:print(name,value,flush=True)

if __name__=='__main__':main()
