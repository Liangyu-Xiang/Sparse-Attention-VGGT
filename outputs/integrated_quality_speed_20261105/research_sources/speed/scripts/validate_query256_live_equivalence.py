"""Full-input lockstep check of every modified node against literal frozen operators.

Unmodified nondeterministic clustering/scatters are shared live inputs, never
cached or replayed from an older layer/input. Not a speed benchmark.
"""
import argparse,hashlib,importlib,json,sys,time,types
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import numpy as np,torch,triton,yaml
from addict import Dict
from scripts.quality_metrics import quality_record
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from saf3r.patch import reuse_window_graph as wg,reuse_inference_ops as rule
from saf3r.triton import reuse_direct_sampled_probability as attn,reuse_shared_value_states as shared
from saf3r.triton.reuse_drop_moment import prepare
from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.da3bench_utils import sample_frames
frozen=ROOT/'outputs/query256_efficiency_20261105/frozen_baseline/saf3r'
for name,path in [('frozen_reference',frozen),('frozen_reference.triton',frozen/'triton')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
ref_attention=importlib.import_module('frozen_reference.triton.reuse_direct_sampled_probability').direct_sparse_attention_sampled
ref_states=importlib.import_module('frozen_reference.triton.reuse_shared_value_states').states
out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n')
result=dict(passed=False,scene=a.scene,frames=128,quality=quality_record({},'Live fixed-input operator equivalence diagnostic; quality tasks not run here, no paired Dense timing'),attention_checks=0,graph_checks=0,constant_copy_checks=0,scope='All modified attention and state/graph nodes on real live tensors; shared unmodified nondeterministic nodes, no old masks or results reused')
save('result.json',result);active={};layer=[None];rows=[]
def equal(x,y,name):
 if not torch.equal(x,y):
  valid=torch.isfinite(x)&torch.isfinite(y)
  raise AssertionError(f'{name}: max_abs={float((x[valid].float()-y[valid].float()).abs().max())}, unequal={int((x!=y).sum())}')
old_states=shared.states
def states(*args,**kw):
 values=old_states(*args,**kw)
 if active.get('window') is not None:active['window']._check_state=(args,values,args[3]['objective'])
 return values
shared.states=states
old_init=wg.Window.__init__
def init(self,*args,**kw):
 active['window']=self
 try:old_init(self,*args,**kw)
 finally:active['window']=None
wg.Window.__init__=init
old_update=wg.Window.update_constants
def update(self,z,reps,labels,population,geometry):
 old_update(self,z,reps,labels,population,geometry)
 for x,y in zip(self.constants,[z,reps,labels,population]):equal(x,y,'constant copy')
 for k,y in geometry.items():
  if torch.is_tensor(y) and k not in ('drop_values','objective') and k in self.geometry:equal(self.geometry[k],y,'geometry '+k)
 result['constant_copy_checks']+=1
wg.Window.update_constants=update
old_call=wg.Window.__call__
def call(self,prior,pi,old,allow,uniform=None):
 outputs=old_call(self,prior,pi,old,allow,uniform=uniform)
 for x,y in zip(self.inputs,[prior,pi,old,allow,uniform]):equal(x,y,'runtime input copy')
 args,(errors,costs),objective=self._check_state
 u,allowed,population,geometry,context,residuals=args
 native=dict(geometry);native.pop('drop_bin_counts',None);native.pop('bounded_bins',None)
 native['drop_bins']=prepare(population);width=triton.next_power_of_2(int(population.max()));native['width']=width
 for name in ('shared_cells','shared_drop','shared_rep'):native[name]=geometry[name][:,:width].contiguous()
 e,c=ref_states(u,allowed,population,native,context,residuals)
 equal(e,errors,'four-state errors');equal(c,costs,'four-state costs');equal(native['objective'],objective,'four-state objective')
 selected,_=wg.budget(native['objective'],c,e[:,:,0].amax(1)+1e-12,config.retention_budget if hasattr(config,'retention_budget') else outputs[0].shape[1],config.dual_iterations)
 selected=rule.decode(selected)
 expected_ids,real,_=rule.pack(selected,None,self.constants[1],self.constants[2],population,outputs[0].shape[1],dict(context))
 equal(expected_ids,outputs[0],'packed support');equal(real,outputs[1],'valid support slots');equal((expected_ids>=0).to(outputs[2].dtype),outputs[2],'weights')
 result['graph_checks']+=1;rows.append(dict(layer=layer[0],window=result['graph_checks'],state_errors_costs_objectives_bit_equal=True,supports_bit_equal=True,inputs_current=True))
 return outputs
wg.Window.__call__=call
old_attention=attn.direct_sparse_attention_sampled
def attention(*args,**kw):
 x,e=old_attention(*args,**kw)
 ref_kw={k:v for k,v in kw.items() if k not in ('block_queries','split_export','execution_split','scope')}
 ref_kw['num_stages']=2
 y,f=ref_attention(*args,**ref_kw)
 equal(x,y,'attention output')
 for name in ('lse','debiased_log_special_probability','debiased_log_patch_probability'):equal(e[name],f[name],name)
 result['attention_checks']+=1
 return x,e
attn.direct_sparse_attention_sampled=attention
try:
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.pop('patch_module');cfg.pop('sparse_config_path')
 model=build_model(cfg,'cuda');dataset=MV_REGISTRY.get('7scenes')();positions,scene=sample_frames(dataset.get_data(a.scene),a.scene,128,sample_mode='uniform',seed=0);images=load_images(model,'vggt',scene.image_files).cuda()
 config=ReuseExploreConfig(**json.loads((a.config or (ROOT/'configs/evaluation/reuse_shared_value_loss_q256_exact.json')).read_text()));state=install_reuse_residual_exploration(model,config)
 result['deterministic_reductions']=config.deterministic_reductions
 save('input.json',dict(frame_indices=positions,image_paths=scene.image_files,shape=list(images.shape)))
 for i,b in enumerate(model.aggregator.global_blocks):b.attn.register_forward_pre_hook(lambda module,args,i=i:layer.__setitem__(0,i))
 save('status.json',dict(stage='diagnostic_forward'))
 begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
 torch.cuda.synchronize();result['diagnostic_wall_seconds']=time.perf_counter()-begin
 assert result['attention_checks']==24*((state.groups+config.window_groups-1)//config.window_groups)
 assert result['graph_checks']==23*((state.groups+config.window_groups-1)//config.window_groups)
 result.update(passed=True,all_modified_nodes_bit_equal=True,graph_cache=dict(hits=state.graph_cache_hits,created=state.graph_cache_creations,evicted=state.graph_cache_evictions),output_sha256={k:hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for k,v in pred.items() if torch.is_tensor(v)})
finally:
 save('checks.json',rows);save('result.json',result)
save('status.json',dict(stage='diagnostic_complete'));print(json.dumps(result,indent=2),flush=True)
