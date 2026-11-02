"""Kernel launch/memory audit on full heads128-size producer tensors."""
import argparse,collections,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch.reuse_residual_exploration import canonical_layout
from saf3r.patch import reuse_headmean_history as old,reuse_grid_local_history as new


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);g,h,a=64,16,26009;grid,r=canonical_layout(128,37,28,2,2,'cuda');sizes=torch.bincount(grid,minlength=r).float()
    ids=((torch.arange(a,device='cuda')[None]*5+torch.arange(g,device='cuda')[:,None]*211)%len(grid)).int()
    bank=grid[ids.long()];order=bank.argsort(dim=1,stable=True).int();counts=torch.zeros(g,r,device='cuda',dtype=torch.int32).scatter_add_(1,bank,torch.ones_like(ids));off=torch.nn.functional.pad(counts.cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    raw=torch.randn(g,h,4,a,device='cuda');base=torch.randn(h,g,4,r,device='cuda').transpose(0,1);special=torch.randn(h,g,4,640,device='cuda').transpose(0,1)
    coarse=torch.cat((base+sizes.log(),special),-1).softmax(-1)[...,:r]
    obs=dict(ready_logits=raw,grid_logits=base,special_logits=special,coarse_grid_probability=coarse,selected_grid_order=order,selected_grid_offsets=off)
    s=NS(n=133248,groups=g,budget=a,regions=r,height=37,width=28,grid=grid,grid_sizes=sizes,config=NS(window_groups=64,grid_y=2,grid_x=2,history_cache_format='packed'),m5c_count_errors=[],m5c_seed_errors=[],previous_layer=-1,ready_written={})
    q=torch.empty(h,1,64,device='cuda');old.ensure_cache(s,q)
    results={}
    for name,fn in [('reference',old.save_evidence),('local_packed',new.save_evidence),('local_fp16',new.save_evidence)]:
        s.config.history_cache_format='fp16' if name=='local_fp16' else 'packed';new.ensure_cache(s,q)
        for _ in range(2):fn(s,q,q,0,ids,ids,obs,1.)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
            fn(s,q,q,0,ids,ids,obs,1.)
        torch.cuda.synchronize();counter=collections.Counter();copies=collections.Counter();launches=0;gpu_us=0.
        for e in prof.events():
            if e.name in ['cudaLaunchKernel','cuLaunchKernel']:launches+=1
            if str(e.device_type)=='DeviceType.CUDA':
                if e.name.startswith(('Memcpy','Memset')):copies[e.name]+=1
                else:counter[e.name]+=1;gpu_us+=e.device_time_total
        results[name]=dict(gpu_kernel_count=sum(counter.values()),cpu_launch_api_count=launches,kernels=dict(counter),copy_or_memset=dict(copies),gpu_active_kernel_ms=gpu_us/1000,temporary_peak_delta_bytes=torch.cuda.max_memory_allocated()-before,cache_bytes=s.ready_edges.numel()*s.ready_edges.element_size())
    audit=dict(results=results,shapes={k:list(obs[k].shape) for k in ['ready_logits','grid_logits','special_logits']},strides={k:list(obs[k].stride()) for k in ['ready_logits','grid_logits','special_logits']},dtypes={k:str(obs[k].dtype) for k in ['ready_logits','grid_logits','special_logits']},raw_bytes=raw.numel()*4,scope='Full64-group heads128-sized synthetic tensors; kernel-count/memory diagnostic, not model timing',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel launch audit only',dense_speedup=None,dense_reason='No end-to-end control')
    (args.output_dir/'result.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
