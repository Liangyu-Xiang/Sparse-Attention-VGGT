import sys,types
from pathlib import Path
import torch,saf3r
root=Path(__file__).resolve().parents[1];m=types.ModuleType('saf3r.patch');m.__path__=[str(root/'saf3r/patch')];sys.modules['saf3r.patch']=m;saf3r.patch=m
from saf3r.patch.logit_state_loss_models import modeled_errors

def test_history_full_execution_has_zero_modeled_residual():
 z=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[2.,1.]])
 p=torch.tensor([[.1,.2,.3,.4]]);labels=torch.tensor([0,0,1,1]);reps=torch.tensor([0,2]);allkeys=torch.ones_like(p,dtype=torch.bool)
 for name in ['power15','power2','mass_bound','vector','vector_diag1','vector_diag4','drop_center','drop_weighted_center','normalizer_vector','normalizer_diag1','normalizer_diag4']:
  e=modeled_errors(p,labels,reps,z,allkeys,name)
  assert (e>=0).all() and (e[:,:,3]==0).all()
  assert torch.allclose(e[:,:,2],torch.zeros_like(e[:,:,2]),atol=1e-6),name

def test_centered_drop_and_normalizer_losses_ignore_common_value_translation():
 z=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[2.,1.]])
 p=torch.tensor([[.1,.2,.3,.4]]);labels=torch.tensor([0,0,1,1]);reps=torch.tensor([0,2]);some=torch.tensor([[True,False,True,False]])
 for name in ['drop_center','drop_weighted_center','normalizer_vector','normalizer_diag1','normalizer_diag4']:
  a=modeled_errors(p,labels,reps,z,some,name);b=modeled_errors(p,labels,reps,z+torch.tensor([.7,-1.2]),some,name)
  assert torch.allclose(a,b,atol=1e-6,rtol=2e-6),name
