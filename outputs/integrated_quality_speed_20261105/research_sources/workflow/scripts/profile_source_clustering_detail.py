"""Run the existing full-forward deep audit with opt-in new-clustering scopes."""
import argparse,ast,importlib,sys
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch
from scripts import deep_operation_timing as deep
from scripts import profile_deep_saf3r_query256 as profile
from saf3r.patch.source_local_unanchored import SourceLocalUnanchoredValue as Cls

def install_source(scope):
 def transform(fn,src):
  loop=next(n for n in fn.body if isinstance(n,ast.For))
  position=fn.body.index(loop)
  inner=next(n for n in loop.body if isinstance(n,ast.For))
  cut=loop.body.index(inner)
  loop.body=loop.body[:cut]+[deep.block('source_cluster/member_assignment',[inner]),deep.block('source_cluster/empty_repair',loop.body[cut+1:])]
  tail=fn.body[position+1:]
  assert isinstance(tail[0],ast.Assign) and isinstance(tail[0].value,ast.Call)
  fn.body=[deep.block('source_cluster/feature_centering_padding',fn.body[:position]),loop,tail[0],deep.block('source_cluster/final_real_representatives',tail[1:])]
 deep.scoped_ast(Cls,'update',scope,transform)
 deep.patch_function_scope(Cls,'_centers','source_cluster/centers_population',scope)
 deep.patch_function_scope(Cls,'__init__','source_cluster/initial_source_layout',scope)

# Meaningful numerical check of inserted scopes on CPU before model execution.
torch.set_num_threads(2);torch.manual_seed(27)
x=torch.randn(3*8*8,8);baseline=Cls(3,8,8,37,'cpu',32,2,'spatiotemporal',2)
baseline.update(x,0);expected=(baseline.labels.clone(),baseline.reps.clone())
original_update=Cls.update;original_centers=Cls._centers;original_init=Cls.__init__
install_source(lambda *args:nullcontext())
check=Cls(3,8,8,37,'cpu',32,2,'spatiotemporal',2);check.update(x,0)
assert torch.equal(check.labels,expected[0]) and torch.equal(check.reps,expected[1])
Cls.update=original_update;Cls._centers=original_centers;Cls.__init__=original_init
original_install=deep.install_deep
def install(model,state,scope):
 result=original_install(model,state,scope)
 install_source(scope)
 result[1]['source_clustering_scopes_preserve_outputs_cpu']=True
 return result
deep.install_deep=install
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();a.method='shared_value_q256';a.frames=128;a.source_root=ROOT
profile.run(a)
