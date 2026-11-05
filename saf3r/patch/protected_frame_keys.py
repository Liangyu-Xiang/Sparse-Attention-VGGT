"""Reserve the coordinate-anchor frame inside the existing total-Key budget."""
import math
import torch
from .value_evolving_attention import token_layout


def protected_layout(frames,height,width,device,protect_first_frame=False):
    patches,semantic=token_layout(frames,height*width,5,device)
    if not protect_first_frame:return patches,semantic,semantic,frames
    if frames<2:raise ValueError('Protected anchor frame requires at least two frames')
    # Only the remaining rectangular frame domain is clustered/approximated.
    # Protected patches cannot be selected again through the compact patch bank.
    protected=torch.cat((semantic,patches[:height*width])).sort().values
    return patches[height*width:],protected,semantic,frames-1


def check_protected_budget(config,total,protected_count,cluster_frames):
    rates=config.layer_retentions or [config.retention]*24
    minimum=min(math.floor(r*total)-protected_count for r in rates)
    if minimum<cluster_frames:
        raise ValueError('Layer budget cannot accommodate protected anchor keys and remaining frame seeds; retention is not silently increased')
