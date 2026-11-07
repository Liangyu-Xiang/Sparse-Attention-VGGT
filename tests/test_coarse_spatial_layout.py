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
