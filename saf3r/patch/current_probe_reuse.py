"""Independent backend: sampled current Dense probabilities + History A states.

No canonical-grid probability estimate,conditional compression,or second selected
QK. Reuse is previous support identity; probabilities are current observations.
"""
import math
import torch,triton
from .fused_shared_av import stage

def main_attention(q,k,v,actual,specials,scale,bias,*,query_group_size=256,block_queries=64,block_keys=128,num_warps=4,num_stages=2):
 from saf3r.triton.reuse_direct_sparse_attention import _attention
 h,r,d=q.shape;g,a=actual.shape;s=len(specials)
 out=torch.empty_like(q);lse=torch.empty((h,r),device=q.device);dummy=torch.empty(1,device=q.device,dtype=torch.int32)
 _attention[(triton.cdiv(r,block_queries),h)](q,k,v,actual,specials,bias,dummy,out,lse,lse,lse,h,r,k.shape[1],a,s,d,q.stride(0),k.stride(0),v.stride(0),scale,triton.next_power_of_2(d),block_keys,True,False,1,1,0,1,0,query_group_size,block_queries,num_warps=num_warps,num_stages=num_stages)
 return out,dict(lse=lse)

def forward(module,x,pos):
 from .reuse_residual_exploration import prepare_qkv
 from .layer_retention_schedule import layer_budget
 from saf3r.triton.current_probe_probability import probability
 from .reuse_window_graph import select
 o,s=module.original,module.state;c=s.config;layer=module.layer
 if module.training or len(x)!=1 or layer!=s.previous_layer+1:raise ValueError('Ordered inference required')
 s.budget=layer_budget(c,layer,s.n,len(s.specials))
 if c.account_dense_special_query_budget:
  from .layer_retention_schedule import output_patch_budget
  dense_queries=s.semantic_specials.numel()//5*c.dense_special_queries
  target=c.retention if c.layer_retentions is None else c.layer_retentions[layer]
  s.budget=output_patch_budget(s.n,len(s.specials),dense_queries,target)
 with stage(s,'shared_qkv_and_rope'):
  q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
 with stage(s,'current_value_geometry'),torch.autocast(x.device.type,enabled=False):
  if c.shared_value_features=='head_sketch':
   from .head_value_sketch import projection,features as sketch
   key=(len(v),v.shape[-1],c.shared_value_head_sketch_dim,v.device)
   if getattr(s,'_value_sketch_shape',None)!=key:s._value_sketch_matrix=projection(*key[:3],key[3]);s._value_sketch_shape=key
   features=sketch(v,s._value_sketch_matrix)[s.patches].contiguous()
  elif c.shared_value_features=='head_mean':features=v.mean(0,dtype=torch.float32)[s.patches]
  else:raise ValueError('Current-probe backend uses head_mean/head_sketch')
  s.fine.update(features,layer);labels,reps=s.fine.labels,s.fine.reps;f=len(reps);n=len(labels)
  population=torch.bincount(labels,minlength=f).float();z=(features/features.square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous()
  norm=z.square().sum(-1).contiguous();distance=(z-z[reps[labels]]).square().sum(-1).contiguous()
  width=triton.next_power_of_2(int(population.max()));assert width<=512
  order=labels.argsort(stable=True);starts=population.long().cumsum(0)-population.long();rank=torch.arange(n,device=x.device)-starts[labels[order]]
  members=torch.full((f,width),-1,device=x.device,dtype=torch.int32);members[labels[order],rank]=order.int()
  geometry=dict(width=1,probe_member_width=width,members=members,norm2=norm,rep_distance=distance,current_dense_prior=True,shared_value_loss=True,fuse_state_kernel=True,rep_history_execution=c.rep_history_execution,drop_bins=[(1,torch.arange(f,device=x.device,dtype=torch.int32))])
 with stage(s,'exact_current_probe_probability'),torch.autocast(x.device.type,enabled=False):
  prior,positions=probability(q,k,s.patches,c.query_group_size,c.current_probability_probes,o.scale)
 output=torch.empty_like(v);history=[];records=[]
 for start in range(0,s.groups,c.window_groups):
  g=min(c.window_groups,s.groups-start);first=start*c.query_group_size;end=min(s.n,(start+g)*c.query_group_size)
  with stage(s,'current_prior_state_selection'),torch.autocast(x.device.type,enabled=False):
   old=torch.full((g,1),-1,device=x.device,dtype=torch.int32) if s.history_ids is None else s.history_ids[start:start+g].contiguous()
   pi=torch.empty((g,f),device=x.device);allowed=torch.ones((g,1),device=x.device,dtype=torch.bool);uniform=torch.zeros((g,1),device=x.device)
   ids,real,weights,record=select(s,layer,prior[start:start+g],pi,old,allowed,z,reps,labels,population,geometry,uniform=uniform)
  actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous();bias=weights.log().contiguous()
  with stage(s,'sparse_attention_only'):
   result,_=main_attention(q[:,first:end],k,v,actual,s.specials,o.scale,bias,query_group_size=c.query_group_size,block_queries=c.attention_block_queries,block_keys=c.attention_block_keys,num_warps=c.attention_warps,num_stages=c.attention_stages)
  with stage(s,'output_and_history_store'):
   output[:,first:end]=result;history.append(ids.clone());records.append(record)
 with stage(s,'history_validation_and_statistics'):
  joined={key:torch.cat([r[key] for r in records]) for key in records[0]}
  if bool(joined['implementation_validation_error'].any()):raise RuntimeError('Current-probe support legality/budget failed')
  s.history_ids=torch.cat(history)
  rows=(s.n-torch.arange(s.groups,device=x.device)*c.query_group_size).clamp_max(c.query_group_size)
  raw=((joined['counts']+len(s.specials))*rows).sum()
  record=dict(layer=layer,query_group_size=c.query_group_size,query_groups=s.groups,fine_clusters=f,patch_budget=s.budget,special_keys=len(s.specials),counts_mean=joined['counts'].float().mean(),unused_budget_mean=(s.budget-joined['counts']).float().mean(),actual_unique_density=raw/s.n**2,final_unique_qk_per_head=raw,
    current_probability_probes=c.current_probability_probes,probability_source='Current exact full-Key softmax,head/probe mean;previous support IDs only',history_evidence_backend='none:no sparse-logit export or grid cache',current_probe_qk_per_head=len(positions)*c.current_probability_probes*(s.n+len(s.patches)),direct_logical_main_qk_per_head=raw,geometry_width=width,rep_history_loss='A',rep_history_execution=c.rep_history_execution,shared_value_features=c.shared_value_features,
    **{key+'_mean':value.float().mean() for key,value in joined.items() if key!='counts'})
 if c.dense_special_queries:
  from .dense_special_queries import correct_output
  with stage(s,'dense_special_query_correction'):
   output,dense_ids=correct_output(q,k,v,output,s.semantic_specials,c.dense_special_queries,o.scale)
  extra=(s.n-joined['counts'][dense_ids//c.query_group_size]-len(s.specials)).sum()
  record.update(dense_special_query_count=len(dense_ids),dense_special_query_physical_qk_per_head=len(dense_ids)*s.n,actual_unique_density=(raw+extra)/s.n**2,final_unique_qk_per_head=raw+extra)
 else:record['dense_special_query_count']=0
 if c.account_dense_special_query_budget:
  record.update(output_retention_target=target,output_unique_qk_cap_per_head=(s.n-dense_queries)*(s.budget+len(s.specials))+dense_queries*s.n,dense_special_queries_in_budget=True)
 s.previous_layer=layer;s.records.append(record)
 return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))
