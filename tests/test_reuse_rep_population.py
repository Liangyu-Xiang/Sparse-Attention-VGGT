"""Execution-only ablation must keep support selection fixed at fixed inputs."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration
from saf3r.models.official_vggt.layers.attention import Attention

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_population_ablation_has_same_initial_selection_and_raw_history(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
 torch.manual_seed(77);torch.set_num_threads(1)
 cfg=ReuseExploreConfig(state_loss='cosine_distortion',history_mode='all_relative_logit',drop_multiplier=1.,rep_gain=True,capture=True,source_width=4,window_groups=2)
 states=[ReuseExploreState(cfg),ReuseExploreState(replace(cfg,mass_transfer=False))]
 for state in states:state.reset(1,8,16,device)
 for layer in range(2):
  module=Attention(dim=32,num_heads=2).to(device).eval()
  x=torch.randn(1,133,32,device=device)
  outputs=[]
  for state in states:
   with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
    outputs.append(ReuseResidualExploration(deepcopy(module),state,layer).eval()(x))
  assert all(torch.isfinite(out).all() for out in outputs)
  for a,b in zip(states[0].captured[layer]['windows'],states[1].captured[layer]['windows']):
   assert torch.equal(a['ids'],b['ids'])
  torch.testing.assert_close(states[0].history_relative,states[1].history_relative,atol=2e-4,rtol=2e-3)
