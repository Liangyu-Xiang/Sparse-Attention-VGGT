import torch
class CameraAudit:
 def __init__(self,state):self.s=state;self.rows=[];self.generation=[];self.fidelity=[];self.pending=[];self.camera_rows=[]
 def __enter__(self):
  import torch
  from saf3r.patch import reuse_residual_exploration as mod,reuse_window_graph as window
  if self.s.config.teacher_history_attention_backend=='probability_tiles':
   from saf3r.triton import teacher_history_probability_tiles as av
  elif self.s.config.teacher_history_attention_backend=='raw_masked':
   from saf3r.triton import teacher_history_raw_fast as av
  else:
   from saf3r.triton import teacher_history_attention as av
  from saf3r.triton import teacher_history_probability as probability
  self.update=probability.update;self.prepare=mod.prepare_qkv;self.select=window.select;self.attention=av.attention;self.handles=[]
  def prepare(original,x,pos):
   q,k,v=self.prepare(original,x,pos);self.q=q.to(v.dtype);self.k=k.to(v.dtype);self.v=v;self.original=original;self.start=0;self.samples=[];s=self.s
   with torch.autocast('cuda',enabled=False):
    cameras=s.semantic_specials.reshape(-1,5)[:,0];sample=torch.linspace(0,len(cameras)-1,min(4,len(cameras)),device=q.device).long();camera_query=cameras[sample]
    camera_probability=(self.q[:,camera_query].float()@self.k.float().transpose(1,2)*original.scale).softmax(-1).mean((0,1));patch_mass=float(camera_probability[s.patches].sum())
    for previous in self.pending:
     importance=camera_probability[torch.tensor(previous['positions'],device=q.device)].cpu().tolist();self.camera_rows.append(dict(previous,importance=importance,next_layer=s.previous_layer+1,patch_mass=patch_mass))
    self.pending=[]
   self.groups=torch.linspace(0,s.groups-1,4,device=q.device).long().unique();self.positions=(self.groups[:,None]*256+torch.tensor([32,96,160,224],device=q.device)).clamp_max(s.n-1)
   self.heldout=(self.positions-16).clamp_min(0)
   self.heldout=torch.where(torch.isin(self.heldout,s.specials),(self.heldout+8).clamp_max(s.n-1),self.heldout)
   assert not bool(torch.isin(self.heldout,s.specials).any())
   with torch.autocast('cuda',enabled=False):
    raw=self.q[:,self.positions.flatten()].float()@self.k.float().transpose(1,2)*original.scale;self.truth=raw.softmax(-1).mean(0).reshape(len(self.groups),4,s.n).mean(1)[:,s.patches]
    raw=self.q[:,self.heldout.flatten()].float()@self.k.float().transpose(1,2)*original.scale;self.dense=(raw.softmax(-1)@v.float()).reshape(len(q),len(self.groups),4,v.shape[-1])
   return q,k,v
  def metrics(a,b):return dict(relative_l1=float((a-b).abs().sum()/b.sum().clamp_min(1e-20)),relative_l2=float((a-b).square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-20)),cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=1).mean()))
  def select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs):
   result=self.select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs);start=self.start;g=len(prior);pick=(self.groups>=start)&(self.groups<start+g);groups=self.groups[pick];local=groups-start
   if len(groups):
    from saf3r.patch.current_probe_states import select as direct
    true=self.truth[pick];pred=prior[local];history=old_ids[local].long();valid=history>=0;safe=history.clamp_min(0)
    ref,_,w,_=direct(true,torch.empty(len(groups),len(reps),device=z.device),old_ids[local].contiguous(),allowed[local],z,reps,labels,population,geometry,s.budget,s.config.dual_iterations)
    n=pred.shape[1];ids=result[0][local];a=torch.zeros(len(groups),n+1,device=z.device,dtype=torch.bool).scatter_(1,ids.long().masked_fill(ids<0,n),ids>=0)[:,:n];b=torch.zeros(len(groups),n+1,device=z.device,dtype=torch.bool).scatter_(1,ref.long().masked_fill(ref<0,n),ref>=0)[:,:n]
    coarse=pred.reshape(len(groups),-1) # Metrics are on actual all-Key predicted probability.
    self.rows.append(dict(layer=layer,groups=groups.tolist(),all_patch_prediction=metrics(pred,true),history_prediction=metrics(pred.gather(1,safe)*valid,true.gather(1,safe)*valid),mask_jaccard=float(((a&b).sum(1)/(a|b).sum(1).clamp_min(1)).mean()),captured_mass_vs_reference=float(((true*a).sum(1)/(true*b).sum(1).clamp_min(1e-20)).mean())))
    self.reference=(pick,local,ref,w)
   else:self.reference=None
   self.start+=g;return result
  def attention(q,k,v,actual,specials,weights,scale,probes=4,capture=True,warps=4,stages=1):
   result,e=self.attention(q,k,v,actual,specials,weights,scale,probes,capture,warps,stages)
   if self.reference is not None:
    pick,local,ref,rw=self.reference;first=(self.start-len(actual))*256;positions=self.heldout[pick];patches=self.s.patches
    def manual(ids,ww):
     keys=torch.cat((specials[None].expand(len(ids),-1),patches[ids.clamp_min(0)].masked_fill(ids<0,-1)),1);weight=torch.cat((torch.ones(len(ids),len(specials),device=q.device),ww),1)
     with torch.autocast('cuda',enabled=False):
      raw=torch.einsum('hgpd,hgad->hgpa',self.q[:,positions].float(),k[:,keys.clamp_min(0)].float())*scale+weight.clamp_min(1e-30).log()[None,:,None];raw.masked_fill_((keys<0)[None,:,None],-torch.inf);return torch.einsum('hgpa,hgad->hgpd',raw.softmax(-1),v[:,keys.clamp_min(0)].float())
    target=manual(ref,rw);got=result[:,positions-first].float();dense=self.dense[:,pick];h,g,p,d=got.shape
    def err(a,b):
     with torch.autocast('cuda',enabled=False):
      aa=(a-b).permute(1,2,0,3).reshape(g*p,h*d);bb=b.permute(1,2,0,3).reshape(g*p,h*d);return float(torch.nn.functional.linear(aa,self.original.proj.weight.float()).square().sum()/torch.nn.functional.linear(bb,self.original.proj.weight.float()).square().sum().clamp_min(1e-20))
    with torch.autocast('cuda',enabled=False):
     delta=torch.nn.functional.linear((got-dense).permute(1,2,0,3).reshape(g*p,h*d),self.original.proj.weight.float());reference=torch.nn.functional.linear(dense.permute(1,2,0,3).reshape(g*p,h*d),self.original.proj.weight.float())
     per_query=dict(layer=self.s.previous_layer+1,positions=positions.flatten().cpu().tolist(),squared_error=delta.square().sum(1).cpu().tolist(),reference_energy=reference.square().sum(1).cpu().tolist())
     self.pending.append(per_query)
    self.fidelity.append(dict(layer=self.s.previous_layer+1,projected_error_vs_teacher_mask=err(got,target),projected_error_vs_dense=err(got,dense),teacher_mask_error_vs_dense=err(target,dense)))
   return result,e
  def update(prior,target_mass,ids,evidence,destination,specials,block,eta):
   self.update(prior,target_mass,ids,evidence,destination,specials,block,eta)
   start=self.start-len(ids);pick=(self.groups>=start)&(self.groups<self.start);local=self.groups[pick]-start
   if len(local):self.generation.append(dict(layer=self.s.previous_layer+1,all_patch_prediction=metrics(destination[local],self.truth[pick])))
  mod.prepare_qkv=prepare;window.select=select;av.attention=attention;probability.update=update
  return self
 def __exit__(self,*exc):
  from saf3r.patch import reuse_residual_exploration as mod,reuse_window_graph as window
  if self.s.config.teacher_history_attention_backend=='probability_tiles':
   from saf3r.triton import teacher_history_probability_tiles as av
  elif self.s.config.teacher_history_attention_backend=='raw_masked':
   from saf3r.triton import teacher_history_raw_fast as av
  else:
   from saf3r.triton import teacher_history_attention as av
  from saf3r.triton import teacher_history_probability as probability
  mod.prepare_qkv=self.prepare;window.select=self.select;av.attention=self.attention;probability.update=self.update
