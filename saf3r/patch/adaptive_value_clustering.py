"""Unanchored centroid Lloyd with local center graph and certified fallback.

Centers serve clustering; actual member representatives serve Attention.
Exact full-search Lloyd is a reference, not a proof of the global SSE optimum.
"""
import torch
from .value_clustering_gpu import centroid_nearest_gpu


def centers_from_labels(x,labels,count):
 population=torch.bincount(labels,minlength=count)
 means=torch.zeros((count,x.shape[-1]),device=x.device,dtype=torch.float32)
 means.index_add_(0,labels,x.float());means/=population.clamp_min(1)[:,None]
 return means,population


def full_assignment(x,centers,old_labels,chunk=1024):
 labels=torch.empty_like(old_labels);cost=torch.empty(len(x),device=x.device)
 cn=centers.square().sum(-1)
 for start in range(0,len(x),chunk):
  xx=x[start:start+chunk].float()
  dist=(xx.square().sum(-1)[:,None]+cn[None]-2*(xx@centers.T)).clamp_min_(0)
  best,idx=dist.min(-1);old=old_labels[start:start+chunk]
  own=dist.gather(1,old[:,None]).squeeze(1)
  keep=own<=best
  labels[start:start+len(xx)]=torch.where(keep,old,idx)
  cost[start:start+len(xx)]=torch.where(keep,own,best)
 return labels,cost


def center_graph(centers,neighbors=32,chunk=512):
 k=len(centers);width=min(neighbors,k)
 ids=torch.empty((k,width),device=centers.device,dtype=torch.long)
 next_distance=torch.full((k,),torch.inf,device=centers.device)
 cn=centers.square().sum(-1)
 for start in range(0,k,chunk):
  cc=centers[start:start+chunk]
  dist=(cc.square().sum(-1)[:,None]+cn[None]-2*(cc@centers.T)).clamp_min_(0)
  # Self is always a candidate, including coincident-center ties.
  own=torch.arange(start,start+len(cc),device=centers.device)
  dist[torch.arange(len(cc),device=centers.device),own]=-1
  v,ix=dist.topk(min(width+1,k),largest=False,sorted=True)
  ids[start:start+len(cc)]=ix[:,:width]
  if width<k:next_distance[start:start+len(cc)]=v[:,width].clamp_min(0).sqrt()
 return ids,next_distance


def local_assignment(x,centers,old_labels,graph,next_distance,chunk=4096):
 labels=torch.empty_like(old_labels);cost=torch.empty(len(x),device=x.device);certified=torch.empty(len(x),device=x.device,dtype=torch.bool)
 for start in range(0,len(x),chunk):
  xx=x[start:start+chunk].float();old=old_labels[start:start+len(xx)];candidate=graph[old]
  # Bounded gathered scratch; never N x K x D.
  selected_centers=centers[candidate]
  dist=(xx[:,None]-selected_centers).square().sum(-1)
  best,idx=dist.min(-1);chosen=candidate.gather(1,idx[:,None]).squeeze(1)
  own=(xx-centers[old]).square().sum(-1)
  keep=own<=best
  labels[start:start+len(xx)]=torch.where(keep,old,chosen);cost[start:start+len(xx)]=torch.where(keep,own,best)
  # Triangle inequality bounds EVERY unseen center. Conservative FP32 slack.
  radius=own.sqrt();lower=(next_distance[old]-radius).clamp_min(0)
  margin=1e-5*(1+radius+next_distance[old].nan_to_num(posinf=0))
  certified[start:start+len(xx)]=(cost[start:start+len(xx)].sqrt()+margin<lower)|torch.isinf(next_distance[old])
 return labels,cost,certified


def repair_empty(x,labels,cost,count):
 population=torch.bincount(labels,minlength=count);empty=torch.where(population==0)[0]
 if len(empty)==0:return labels,0
 # Protect ONE ordinary member per populated cluster solely during repair,
 # not the former Rep. No persistent identity constraint across iterations.
 first=torch.full((count,),len(x),device=x.device,dtype=torch.long)
 first.scatter_reduce_(0,labels,torch.arange(len(x),device=x.device),reduce='amin')
 eligible=torch.arange(len(x),device=x.device)!=first[labels]
 candidates=cost.masked_fill(~eligible,-torch.inf).argsort(descending=True,stable=True)[:len(empty)]
 labels=labels.clone();labels[candidates]=empty
 return labels,len(empty)


def lloyd(x,labels,count,iterations=2,mode='certified',neighbors=32,stop_when_stable=False):
 if mode not in ('local','certified','global'):raise ValueError(mode)
 x=x.float();offset=x.mean(0);x=x-offset;labels=labels.clone();records=[];cached_graph=None
 for iteration in range(iterations):
  centers,population=centers_from_labels(x,labels,count)
  if bool((population==0).any()):raise ValueError('Initial partition must be nonempty')
  if mode=='global':new,cost=full_assignment(x,centers,labels);fallback=len(x);fraction=0.
  else:
   if mode=='certified' or cached_graph is None:cached_graph=center_graph(centers,neighbors)
   graph,bound=cached_graph
   new,cost,certified=local_assignment(x,centers,labels,graph,bound)
   # A reused graph has no valid current-center certificate.
   if mode=='local' and iteration:certified=torch.zeros_like(certified)
   fraction=float(certified.float().mean());uncertain=torch.where(~certified)[0]
   fallback=len(uncertain) if mode=='certified' else 0
   if fallback:
    corrected,exact_cost=full_assignment(x[uncertain],centers,labels[uncertain]);new[uncertain]=corrected;cost[uncertain]=exact_cost
  new,repairs=repair_empty(x,new,cost,count)
  changed=int((new!=labels).sum());labels=new
  records.append(dict(iteration=iteration,changed_tokens=changed,certified_fraction=fraction,fallback_tokens=fallback,empty_reseeds=repairs))
  if stop_when_stable and changed==0:break
 centers,population=centers_from_labels(x,labels,count)
 # Nonempty clusters make every returned representative a distinct real member.
 distance=(x-centers[labels]).square().sum(-1)
 minimum=torch.full((count,),torch.inf,device=x.device).scatter_reduce_(0,labels,distance,reduce='amin')
 candidate=torch.where(distance==minimum[labels],torch.arange(len(x),device=x.device),len(x))
 reps=torch.full((count,),len(x),device=x.device,dtype=torch.long).scatter_reduce_(0,labels,candidate,reduce='amin')
 centers=centers+offset
 return labels,reps,centers,records

class AdaptiveFullCoverageValue:
 def __init__(self,frames,patches_per_frame,base_count,device,width=32,iterations=2,mode='certified',neighbors=32):
  from .fused_shared_av import FullCoverageValue
  initial=FullCoverageValue(frames,patches_per_frame,base_count,device,width,iterations)
  self.labels=initial.labels;self.reps=initial.reps;self.base_count=base_count;self.iterations=iterations;self.mode=mode;self.neighbors=neighbors;self.merges={};self.records=[]
 def update(self,x,layer):
  old_reps=self.reps
  mode='global' if layer==0 else self.mode
  self.labels,self.reps,self.centers,steps=lloyd(x,self.labels,self.base_count,self.iterations,mode,self.neighbors)
  self.records.append(dict(layer=layer,mode=mode,neighbors=self.neighbors,rep_changed_fraction=float((self.reps!=old_reps).float().mean()),steps=steps,anchor_constraint=False,global_scope='whole sequence; no frame/source-cell boundary'))
 def representatives(self,x,count):
  if count==self.base_count:return self.reps,self.labels
  from .fused_shared_av import FullCoverageValue
  return FullCoverageValue.representatives(self,x,count)
