"""Matched AV/probability-export kernels; alternate results never feed model.

Default covers first real window of EVERY global layer. --all-windows covers
all1584 real windows for heads128. No Dense/SAF3R control reruns or GT reads.
"""
import argparse,json,statistics,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch
from scripts.benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_direct_sparse_attention as direct
# Bind the actual JIT body before installing diagnostic launch wrappers.
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled


def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--all-windows',action='store_true');p.add_argument('--focus',action='store_true');p.add_argument('--branch',action='store_true');p.add_argument('--probes',type=int,choices=[1,4],default=4);p.add_argument('--repeats',type=int,default=7);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model,state=build(ROOT/'configs/evaluation/reuse_direct_probability.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
 original=direct.direct_sparse_attention;rows=[];resources={};current_method='';layer_windows={};warmed=set()
 for name in ['_attention','_inline_mean','_inline_log_mean','_probe']:
  kernel=getattr(direct,name)
  class Resource:
   def __init__(self,kernel,name):self.kernel=kernel;self.name=name
   def __getitem__(self,grid):
    launch=self.kernel[grid]
    def run(*args,**kwargs):
     compiled=launch(*args,**kwargs)
     if current_method:resources.setdefault(current_method,{}).setdefault(self.name+('_mode'+str(args[-1]) if self.name=='_attention' else ''),dict(registers=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),shared_bytes=getattr(compiled.metadata,'shared',None)))
     return compiled
    return run
  setattr(direct,name,Resource(kernel,name))
 def run(q,k,v,ids,specials,scale,*args,**kwargs):
  nonlocal current_method
  layer=state.previous_layer+1;window=layer_windows.get(layer,0);layer_windows[layer]=window+1
  # Preserve the original current-probability trajectory exactly.
  reference=original(q,k,v,ids,specials,scale,*args,**kwargs)
  if not a.all_windows and window:return reference
  bias=kwargs['logit_bias'];positions=kwargs['probe_positions'][:,:a.probes].contiguous();shape=(q.shape,ids.shape);results={};timings={}
  methods={'av_only':None,'old_inline':'inline','probe_probability':'lse_probe','fast_inline':'inline_fast','masked_inline':'inline_masked','probe_debiased_log':'debiased_log_lse_probe','inline_debiased_log':'debiased_log_inline','masked_debiased_log':'debiased_log_inline_masked'}
  if a.probes==1:methods.update({f'sampled_inline_{stride}':'inline_masked' for stride in [2,4,8,16]})
  if a.branch:
   from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
   methods.update({f'branch_inline_{stride}':'inline_masked' for stride in [1,4,8,16,64]})
  if a.branch and a.focus:methods={k:v for k,v in methods.items() if k in ['av_only','probe_probability','branch_inline_1','branch_inline_8','branch_inline_16']}
  elif a.focus:methods={k:v for k,v in methods.items() if k in ['av_only','probe_probability','masked_inline','sampled_inline_8','sampled_inline_16']}
  for method,export in methods.items():
   current_method=method
   stride=int(method.rsplit('_',1)[-1]) if method.startswith(('sampled_inline_','branch_inline_')) else 1
   if method.startswith('branch_inline_'):
    def fn():return direct_sparse_attention_sampled(q,k,v,ids,specials,scale,bias,positions,probe_count=a.probes,probe_group_stride=stride)
   else:
    def fn():return original(q,k,v,ids,specials,scale,bias,positions,export is not None,history_export=export or 'lse_probe',probe_group_stride=stride)
   tag=(method,shape)
   if tag not in warmed:
    for _ in range(2):fn()
    torch.cuda.synchronize();warmed.add(tag)
   samples=[]
   for _ in range(a.repeats):
    st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();result=fn();en.record();en.synchronize();samples.append(st.elapsed_time(en)/1000)
   timings[method+'_seconds']=statistics.median(samples);results[method]=result
   if method.startswith('branch_inline_'):resources[method]=result[1]['compiler_resources']
  current_method=''
  errors={}
  for method,(out,evidence) in results.items():
   rel=float((out.float()-results['av_only'][0].float()).norm()/results['av_only'][0].float().norm().clamp_min(1e-30));assert rel<.005,(layer,window,method,rel)
   errors[method+'_av_relative_l2']=rel
  def prob(name,key):return torch.cat([results[name][1]['special_'+key],results[name][1]['patch_'+key]],-1)
  fast=prob('fast_inline' if 'fast_inline' in results else 'branch_inline_1' if a.branch else 'masked_inline','probability');oracle=prob('probe_probability','probability')
  errors['fast_probability_max_abs']=float((fast-oracle).abs().max());errors['fast_probability_l1_mean']=float((fast-oracle).abs().sum(-1).mean());errors['fast_normalization_max_abs']=float((fast.sum(-1)-1).abs().max());assert errors['fast_probability_max_abs']<2e-4,(layer,window,errors)
  for stride in ([] if a.branch and a.focus else [8,16] if a.focus else [2,4,8,16]) if a.probes==1 else []:
   sampled=prob(f'sampled_inline_{stride}','probability');err=float((sampled-oracle[::stride]).abs().max());errors[f'stride_{stride}_probability_max_abs']=err;assert err<2e-4,(stride,err)
  for stride in ([1,8,16] if a.focus else [1,4,8,16,64]) if a.branch else []:
   sampled=prob(f'branch_inline_{stride}','probability');err=float((sampled-oracle[::stride]).abs().max());errors[f'branch_stride_{stride}_probability_max_abs']=err;errors[f'branch_stride_{stride}_normalization_max_abs']=float((sampled.sum(-1)-1).abs().max());assert err<2e-4,(stride,err)
  masked=prob('branch_inline_1' if a.branch and a.focus else 'masked_inline','probability');errors['masked_probability_max_abs']=float((masked-oracle).abs().max());assert errors['masked_probability_max_abs']<2e-4
  if 'inline_debiased_log' in results:
   logfast=torch.cat([results['inline_debiased_log'][1]['debiased_log_special_probability'],results['inline_debiased_log'][1]['debiased_log_patch_probability']],-1)
   logoracle=torch.cat([results['probe_debiased_log'][1]['debiased_log_special_probability'],results['probe_debiased_log'][1]['debiased_log_patch_probability']],-1)
   finite=torch.isfinite(logoracle);errors['inline_log_max_abs']=float((logfast[finite]-logoracle[finite]).abs().max());errors['inline_log_mean_abs']=float((logfast[finite]-logoracle[finite]).abs().mean());assert torch.isfinite(logfast[finite]).all()
  row=dict(layer=layer,window=window,groups=len(ids),query_rows=q.shape[1],heads=len(q),key_rows=k.shape[1],retained_mean=float((ids>=0).sum(1).float().mean()),**timings,**errors,inline_scratch_bytes=results['branch_inline_1' if a.branch and a.focus else 'masked_inline'][1]['scratch_bytes'])
  rows.append(row);(a.output_dir/'progress.json').write_text(json.dumps(dict(windows=len(rows),last=row),indent=2)+'\n')
  return reference
 # Compile/warm the original trajectory before installing the diagnostic hook.
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 torch.cuda.synchronize();direct.direct_sparse_attention=run;torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
 torch.cuda.synchronize();totals={name:sum(x[name] for x in rows) for name in rows[0] if name.endswith('_seconds')};layers=[]
 for layer in range(24):
  group=[x for x in rows if x['layer']==layer];layers.append(dict(layer=layer,windows=len(group),**{name:sum(x[name] for x in group) for name in totals}))
 result=dict(complete=True,sampling='ALL real windows/all24 layers' if a.all_windows else 'First real64-group window of EACH of24 global layers; NOT all windows',trajectory='frozen Direct probability-history config; alternate outputs never feed model',input=json.loads(a.input.read_text()),probes=a.probes,repeats=a.repeats,totals=totals,windows=rows,layers=layers,compiler_resources=resources,diagnostic_wall_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
  quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel-only fixed-input diagnostic; no pose or reconstruction evaluation; no model backend change',dense_speedup=None,dense_reason='Kernel diagnostic; no paired end-to-end Dense timing',source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['saf3r/triton/reuse_direct_sparse_attention.py','saf3r/triton/reuse_direct_sampled_probability.py','scripts/benchmark_inline_probability.py']})
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,windows=len(rows),totals=totals,resources=resources),indent=2),flush=True)
 from scripts.quality_metrics import quality_record,quality_markdown
 (a.output_dir/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
if __name__=='__main__':main()
