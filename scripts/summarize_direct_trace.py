"""Stream a large Kineto trace with bounded memory; no CUDA context."""
import argparse,collections,csv,gzip,json,shutil,time
from pathlib import Path

def events(path):
 decoder=json.JSONDecoder()
 with path.open() as f:
  buf=''
  while '"traceEvents"' not in buf:buf+=f.read(65536)
  pos=buf.index('[',buf.index('"traceEvents"'))+1
  while True:
   while pos<len(buf) and buf[pos] in ' \t\r\n,':pos+=1
   if pos<len(buf) and buf[pos]==']':return
   try:value,end=decoder.raw_decode(buf,pos)
   except json.JSONDecodeError:
    more=f.read(2**20)
    if not more:raise
    buf=buf[pos:]+more;pos=0;continue
   yield value;pos=end
   if pos>2**20:buf=buf[pos:];pos=0

def write(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();kernels={};launches=[];alloc=[];copies=[];cpu=collections.defaultdict(list);first=None;last=0.;count=0
 for e in events(a.trace):
  count+=1;name=e.get('name','');cat=e.get('cat','');dur=e.get('dur',0.);ts=e.get('ts',0.);args=e.get('args',{});end=ts+dur
  if cat in ('kernel','gpu_memcpy','gpu_memset','cpu_op','cuda_runtime'):
   first=ts if first is None else min(first,ts);last=max(last,end)
  if cat=='kernel':
   r=kernels.setdefault(name,dict(kernel=name,calls=0,gpu_kernel_self_seconds=0.,includes_children=False));r['calls']+=1;r['gpu_kernel_self_seconds']+=dur/1.e6
  if cat=='cuda_runtime':launches.append(dict(operation=name,cpu_start_us=ts,cpu_duration_us=dur,thread=e.get('tid')))
  if cat in ('gpu_memcpy','gpu_memset'):copies.append(dict(operation=name,category=cat,gpu_seconds=dur/1.e6,bytes=args.get('Bytes'),scope='CUPTI transfer/memset payload, not total kernel DRAM traffic'))
  if name=='[memory]':alloc.append(dict(cpu_time_us=ts,**args))
  if cat=='cpu_op':cpu[e.get('tid')].append((ts,end,name))
 cpugroups={}
 for thread,intervals in cpu.items():
  stack=[]
  def finish():
   start,end,name,children=stack.pop();r=cpugroups.setdefault(name,dict(operation=name,calls=0,cpu_self_seconds=0.,cpu_inclusive_seconds=0.));r['calls']+=1;r['cpu_inclusive_seconds']+=(end-start)/1.e6;r['cpu_self_seconds']+=max(0.,end-start-children)/1.e6
   if stack:stack[-1][3]+=end-start
  for ts,end,name in sorted(intervals,key=lambda r:(r[0],-r[1])):
   while stack and ts>=stack[-1][1]:finish()
   stack.append([ts,end,name,0.])
  while stack:finish()
 out=a.output_dir;write(out/'kernel_self_time.csv',sorted(kernels.values(),key=lambda r:r['gpu_kernel_self_seconds'],reverse=True));write(out/'kernel_launches.csv',launches);write(out/'memory_allocations.csv',alloc);write(out/'memory_transfers.csv',copies);write(out/'cpu_operator_self_time.csv',sorted(cpugroups.values(),key=lambda r:r['cpu_self_seconds'],reverse=True))
 info=dict(complete=True,events=count,trace_span_seconds=(last-first)/1.e6,kernel_self_total=sum(r['gpu_kernel_self_seconds'] for r in kernels.values()),kernel_count=sum(r['calls'] for r in kernels.values()),runtime_calls=len(launches),alloc_events=len(alloc),timestamp_markers=False,graph_replay_kernels_included=True,quality={m:None for m in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Operator trace only; no pose or reconstruction evaluation',dense_speedup=None,dense_reason='Unchanged Dense not rerun');(out/'operator_trace_info.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
 local=Path('/tmp/direct_probability_trace_archived.json.gz')
 with a.trace.open('rb') as inp,gzip.open(local,'wb',compresslevel=1) as compressed:shutil.copyfileobj(inp,compressed,length=2**20)
 shutil.copyfile(local,out/'profiler_trace.json.gz');print('COMPRESSED_TRACE',local.stat().st_size,flush=True)
if __name__=='__main__':main()
