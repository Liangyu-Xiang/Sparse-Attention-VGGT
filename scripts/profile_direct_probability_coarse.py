"""Low-overhead exclusive A/B/C CUDA-event stages; separate from detailed markers."""
import argparse,contextlib,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from scripts.direct_detail_profile import save,csvout
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,s=build(a.config);images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();events=[]
@contextlib.contextmanager
def scope(name,layer=None):
 st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
 try:yield
 finally:en.record();events.append((name,s.previous_layer+1 if layer is None else layer,st,en))
def wrap(owner,name,label,layer=None):
 orig=getattr(owner,name)
 def call(*args,**kwargs):
  with scope(label,layer):return orig(*args,**kwargs)
 setattr(owner,name,call)
import saf3r.patch.reuse_residual_exploration as impl
wrap(impl,'prepare_qkv','A/qkv_norm_rope')
orig=impl.ReuseResidualExploration.forward
def fwd(att,*args,**kwargs):
 with scope('global',att.layer):return orig(att,*args,**kwargs)
impl.ReuseResidualExploration.forward=fwd
for i,b in enumerate(model.aggregator.global_blocks):wrap(b.attn.original.proj,'forward','A/output_projection',i)
import saf3r.patch.reuse_probability_history as history
wrap(history,'build_prior','C5/prior')
wrap(history,'save_evidence','C4/calibration_cache')
wrap(history,'observe_execution','C4/probe_grid_summary')
import saf3r.patch.adaptive_value_clustering as cluster
for name in ['centers_from_labels','center_graph','local_assignment','repair_empty']:
 wrap(cluster,name,'C1/'+name)
import saf3r.triton.reuse_compact as compact
wrap(compact,'prepare_geometry','C1/shared_geometry')
import saf3r.patch.reuse_window_graph as window
wrap(window,'select','C6_C10/selection_graph')
from scripts.direct_detail_profile import instrument
class Simple:
 sources=[]
 scope=staticmethod(scope)
instrument(window,'Window.__call__','C10/graph_call',Simple())
instrument(window,'Window.update_constants','C10/constants',Simple())
s.direct_scope=lambda name:scope('B/core' if name=='direct_av' else 'C3/export');s.profiling=True;s.profile_events=[];torch.cuda.synchronize();begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();wall=time.perf_counter()-begin;rows=[dict(operation=name,layer=layer,gpu_seconds=st.elapsed_time(en)/1000,includes_children=True) for name,layer,st,en in events];rows.extend(dict(operation='stage/'+name,layer=layer,gpu_seconds=st.elapsed_time(en)/1000,includes_children=True) for layer,name,st,en in s.profile_events)
values={name:sum(r['gpu_seconds'] for r in rows if r['operation']==name) for name in {r['operation'] for r in rows}};global_time=values['global'];inside_a=sum(v for k,v in values.items() if k.startswith('A/'));b=values['B/core'];top=[dict(category='A',gpu_phase_seconds=wall-global_time+inside_a,includes_children=False),dict(category='B',gpu_phase_seconds=b,includes_children=False),dict(category='C',gpu_phase_seconds=global_time-inside_a-b,includes_children=False)];save(a.output_dir/'coarse_phase_profile.json',dict(wall_seconds=wall,groups=values,top_level=top,scope='CUDA events; outer global scopes exclude shared QKV/output projection, direct core counted once. A contains shared-model residual and its scheduling gaps. Graph internals obtained from independent trace; no marker kernels here.',quality={m:None for m in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Profiling only',dense_speedup=None,dense_reason='No Dense rerun'));csvout(a.output_dir/'top_level.csv',top);csvout(a.output_dir/'coarse_phase_per_layer.csv',rows);print('COARSE_COMPLETE',wall,top,flush=True)
