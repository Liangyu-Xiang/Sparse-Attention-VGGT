"""Independent production-kernel timeline, no timestamp marker kernels."""
import argparse,contextlib,csv,json,os,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from scripts.direct_detail_profile import csvout,save
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,s=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
@contextlib.contextmanager
def scope(name):
 with torch.profiler.record_function('direct/'+name):
  torch.cuda.nvtx.range_push(name)
  try:yield
  finally:torch.cuda.nvtx.range_pop()
s.direct_scope=scope
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();begin=time.perf_counter()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=False) as prof:
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 torch.cuda.synchronize()
wall=time.perf_counter()-begin;tmp=Path(tempfile.gettempdir())/('direct_trace_'+str(os.getpid())+'.json');prof.export_chrome_trace(str(tmp));shutil.copyfile(tmp,a.output_dir/'profiler_trace.json');tmp.unlink();kernels={};cpu=[];launches=[];mem=[]
for e in prof.events():
 if e.device_type==torch.autograd.DeviceType.CUDA:
  row=kernels.setdefault(e.name,dict(kernel=e.name,calls=0,gpu_kernel_self_seconds=0.,includes_children=False));row['calls']+=1;row['gpu_kernel_self_seconds']+=e.time_range.elapsed_us()/1.e6
 if e.name in ('cudaLaunchKernel','cudaGraphLaunch','cuLaunchKernelEx','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync','cudaMalloc','cudaFree'):
  launches.append(dict(operation=e.name,cpu_start_us=e.time_range.start,cpu_duration_us=e.time_range.elapsed_us(),thread=e.thread))
 if e.name=='[memory]':mem.append(dict(operation=e.name,cpu_time_us=e.time_range.start,net_device_bytes=e.device_memory_usage,net_cpu_bytes=e.cpu_memory_usage))
for e in prof.key_averages():cpu.append(dict(operation=e.key,calls=e.count,cpu_self_seconds=e.self_cpu_time_total/1.e6,cpu_total_seconds=e.cpu_time_total/1.e6,device_memory_net_bytes=e.self_device_memory_usage))
csvout(a.output_dir/'kernel_self_time.csv',sorted(kernels.values(),key=lambda r:r['gpu_kernel_self_seconds'],reverse=True));csvout(a.output_dir/'cpu_operator_self_time.csv',sorted(cpu,key=lambda r:r['cpu_self_seconds'],reverse=True));csvout(a.output_dir/'kernel_launches.csv',launches);csvout(a.output_dir/'memory_allocations.csv',mem)
save(a.output_dir/'operator_trace_info.json',dict(wall_seconds=wall,trace_overhead_includes_profiler=True,timestamp_markers=False,graph_replay_kernels_included=True,kernel_self_total=sum(r['gpu_kernel_self_seconds'] for r in kernels.values()),kernel_count=sum(r['calls'] for r in kernels.values()),quality={m:None for m in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Operator trace only; pose/reconstruction not rerun',dense_speedup=None,dense_reason='Unchanged Dense not rerun'))
print('TRACE_COMPLETE',wall,flush=True)
