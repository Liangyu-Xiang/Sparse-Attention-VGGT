"""Static-shape Dense CUDA graph fallback; outputs verified against eager Dense."""
import torch
from torch import nn

class DenseGraph(nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model;self.graph=None;self.static_input=None;self.static_output=None;self.equivalence={}
 def forward(self,images):
  if self.graph is None:
   self.static_input=images.clone()
   with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
    reference=self.model(self.static_input)
   torch.cuda.synchronize()
   from .graph_compatible_vggt import install
   install(self.model,images)
   torch.cuda.empty_cache();self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
    self.static_output=self.model(self.static_input)
   self.graph.replay();torch.cuda.synchronize()
   for key in ('pose_enc','depth','depth_conf'):
    self.equivalence[key]=torch.equal(reference[key],self.static_output[key])
   if not all(self.equivalence.values()):raise RuntimeError('Dense CUDA graph output differs from matched eager initialization')
  else:
   self.static_input.copy_(images);self.graph.replay()
  return self.static_output
