"""Fixed balanced per-frame spatial strips, without small remainder blocks."""
import torch


def fixed_blocks(n,patches_per_frame,target,device):
 if n%patches_per_frame or target<1:raise ValueError('Invalid fixed frame/block layout')
 per_frame=max(1,patches_per_frame//target)
 positions=torch.arange(n,device=device)
 grid=(positions//patches_per_frame)*per_frame+(positions%patches_per_frame)*per_frame//patches_per_frame
 sizes=torch.bincount(grid).float()
 return grid,sizes
