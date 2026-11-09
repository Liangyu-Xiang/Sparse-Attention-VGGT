import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'saf3r/patch/coarse_spatial_layout.py';spec=importlib.util.spec_from_file_location('layout',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_requested_large_groups_have_no_fine_tail_and_match_frames():
 for target,actual,groups in [(256,259,256),(512,518,128),(1024,1036,64)]:
  grid,sizes=m.fixed_blocks(64*1036,1036,target,'cpu')
  assert len(sizes)==groups and torch.all(sizes==actual)
  assert torch.all(grid[1035::1036][:-1]!=grid[1036::1036])
  assert sizes.min()>64


def test_size_limit_preserves_frame_boundaries_and_has_no_tiny_remainder():
 for patches in [925,1036]:
  for target in [128,256,512,1024]:
   grid,sizes=m.fixed_blocks_at_most(3*patches,patches,target,'cpu')
   assert sizes.sum()==3*patches and sizes.max()<=target
   assert sizes.max()-sizes.min()<=1
   blocks_per_frame=len(sizes)//3
   assert len(sizes)==3*blocks_per_frame
   for frame in range(3):
    ids=grid[frame*patches:(frame+1)*patches]
    assert ids.unique().numel()==blocks_per_frame
    assert ids.min()==frame*blocks_per_frame
    assert ids.max()==(frame+1)*blocks_per_frame-1
 # The 1024 ETH3D baseline is unchanged by the new upper-limit convention.
 a,sa=m.fixed_blocks(3*925,925,1024,'cpu')
 b,sb=m.fixed_blocks_at_most(3*925,925,1024,'cpu')
 assert torch.equal(a,b) and torch.equal(sa,sb)
