"""Execution graph only: overwrite current-layer geometry and all window data.

The graph retains no probability/state decision from another layer. Binned
capacities are static; actual bin lengths are fresh GPU scalar inputs.
"""
from collections import OrderedDict
import torch,triton
from saf3r.triton import reuse_light_states as rule
from saf3r.triton.reuse_compact import budget

class Window:
 def __init__(self,ids,hp,bg,z,reps,labels,pop,allowed,o0,geo,cap,iterations):
  self.inputs=[torch.empty_like(t) for t in [ids,hp,bg,allowed,o0]]
  self.constants=[t.clone() for t in [z,reps,labels,pop]]
  self.geometry={k:v.clone() if torch.is_tensor(v) else v for k,v in geo.items() if k!='bins'}
  self.geometry['bins']=[(size,torch.empty(triton.next_power_of_2(len(c)),device=c.device,dtype=c.dtype)) for size,c in geo['bins']]
  self.geometry['bin_counts']=[torch.empty(1,device=z.device,dtype=torch.int32) for _ in geo['bins']]
  self.update_geometry(z,reps,labels,pop,geo);self.update_inputs(ids,hp,bg,allowed,o0)
  def run():
   ids,hp,bg,allowed,o0=self.inputs;z,reps,labels,pop=self.constants
   errors,costs,qualified=rule.states(ids,hp,bg,z,reps,labels,pop,allowed,o0,self.geometry)
   selected,_=budget(errors,costs,errors[:,:,0].amax(1)+1.e-12,cap,iterations)
   packed,weight,counts=rule.pack(selected,costs,reps,pop,qualified,self.geometry,cap)
   stats=torch.stack([(selected==i).sum(1) for i in range(4)])
   return packed,weight,counts,stats,errors,selected
  run();self.graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(self.graph):self.outputs=run()
 def update_inputs(self,ids,hp,bg,allowed,o0):
  for dst,src in zip(self.inputs,[ids,hp,bg,allowed,o0]):dst.copy_(src)
 def update_geometry(self,z,reps,labels,pop,geo):
  for dst,src in zip(self.constants,[z,reps,labels,pop]):dst.copy_(src)
  for key in ['members','sums']:self.geometry[key].copy_(geo[key])
  for i,((_,dst),(_,src)) in enumerate(zip(self.geometry['bins'],geo['bins'])):
   dst[:len(src)].copy_(src);self.geometry['bin_counts'][i].fill_(len(src))
 def __call__(self,ids,hp,bg,allowed,o0):
  self.update_inputs(ids,hp,bg,allowed,o0);self.graph.replay()
  packed,weights,counts,stats,errors,selected=self.outputs
  # Persistent history and records must not alias buffers overwritten at replay.
  return packed.clone(),weights,counts.clone(),stats.clone(),errors,selected

def select(s,layer,ids,hp,bg,z,reps,labels,pop,allowed,o0,geo):
 if not hasattr(s,'light_graphs'):s.light_graphs=OrderedDict()
 key=(tuple(ids.shape),len(z),len(reps),geo['width'],tuple((size,triton.next_power_of_2(len(c))) for size,c in geo['bins']))
 tag=(s._forward_serial,layer);graphs=s.light_graphs
 if key not in graphs:
  graphs[key]=Window(ids,hp,bg,z,reps,labels,pop,allowed,o0,geo,s.budget,s.config.dual_iterations);graphs[key].loaded=tag
  if len(graphs)>s.config.graph_capacity:graphs.popitem(last=False)
 w=graphs[key];graphs.move_to_end(key)
 if w.loaded!=tag:w.update_geometry(z,reps,labels,pop,geo);w.loaded=tag
 return w(ids,hp,bg,allowed,o0)
