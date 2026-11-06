"""Temporal full fine-density cache with low-cost current block/anchor correction.

Only L0 uses full-Key probes. Later layers preserve unobserved fine history;
formal multi-head AV supplies selected conditional evidence with no extra QK.
"""
import torch,triton
from .fused_shared_av import stage


def prepare_keys(s,k):
 from .head_value_sketch import features
 c=s.config;block=c.history_key_block_size;m=c.teacher_history_samples
 h,_,d=k.shape;n=len(s.patches);r=triton.cdiv(n,block)
 shape=(n,block,m,k.device)
 if getattr(s,'_teacher_layout_shape',None)!=shape:
  sizes=(n-torch.arange(r,device=k.device)*block).clamp_max(block);effective=sizes.clamp_max(m)
  slot=torch.arange(m,device=k.device);valid=slot[None,:]<effective[:,None]
  index=torch.arange(r,device=k.device)[:,None]*block+((2*slot[None,:]+1)*sizes[:,None]//(2*effective[:,None]))
  index=index.clamp_max(n-1);s._teacher_layout=(sizes.float(),index,valid);s._teacher_layout_shape=shape
 sizes,index,valid=s._teacher_layout
 pad=r*block-n
 if getattr(c,'teacher_history_fused_prepare',False):
  from saf3r.triton.teacher_history_prepare import prepare
  bank,anchor,z=prepare(k,s.patches,s.specials,s._value_sketch_matrix,block,m)
 else:
  patch=k[:,s.patches].float();padded=torch.nn.functional.pad(patch,(0,0,0,pad));pooled=padded.reshape(h,r,block,d).sum(2)/sizes[None,:,None]
  anchor=patch[:,index.flatten()].to(torch.float16).contiguous();z=features(k,s._value_sketch_matrix)[s.patches]
  bank=torch.cat((pooled,k[:,s.specials].float()),1).half().contiguous()
 zpad=torch.nn.functional.pad(z,(0,0,0,pad)).reshape(r,block,-1);az=z[index]
 distance=(zpad.square().sum(-1)[:,:,None]+az.square().sum(-1)[:,None,:]-2*torch.bmm(zpad,az.transpose(1,2)))
 assignment=distance.masked_fill(~valid[:,None,:],float('inf')).argmin(-1).flatten()[:n].int().contiguous()
 bias=torch.cat((sizes.log(),torch.zeros(len(s.specials),device=k.device))).contiguous()
 return dict(bank=bank,bias=bias,anchor_keys=anchor,valid=valid.flatten().contiguous(),regions=r,samples=m,assignment=assignment,sizes=sizes)


def forward(module,x,pos):
 from .reuse_residual_exploration import prepare_qkv
 from .layer_retention_schedule import layer_budget
 from .head_value_sketch import projection,features as sketch
 from .reuse_window_graph import select
 from saf3r.triton.current_probe_probability import probability
 from saf3r.triton import teacher_history_probability as prob
 o,s=module.original,module.state;c=s.config;layer=module.layer
 if module.training or len(x)!=1 or layer!=s.previous_layer+1:raise ValueError('Ordered inference required')
 if c.teacher_history_attention_backend=='probability_tiles':
  from saf3r.triton.teacher_history_probability_tiles import attention
 elif c.teacher_history_attention_backend=='raw_masked':
  from saf3r.triton.teacher_history_raw_fast import attention
 else:
  from saf3r.triton.teacher_history_attention import attention
 s.budget=layer_budget(c,layer,s.n,len(s.specials));probes=c.teacher_history_probes;block=c.history_key_block_size
 with stage(s,'shared_qkv_and_rope'):
  q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
 with stage(s,'current_value_geometry'),torch.autocast(x.device.type,enabled=False):
  key=(len(v),v.shape[-1],c.shared_value_head_sketch_dim,v.device)
  if getattr(s,'_value_sketch_shape',None)!=key:s._value_sketch_matrix=projection(*key[:3],key[3]);s._value_sketch_shape=key
  features=sketch(v,s._value_sketch_matrix)[s.patches].contiguous()
  s.fine.update(features,layer);labels,reps=s.fine.labels,s.fine.reps;f=len(reps);n=len(labels)
  population=torch.bincount(labels,minlength=f).float();z=(features/features.square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous()
  norm=z.square().sum(-1).contiguous();distance=(z-z[reps[labels]]).square().sum(-1).contiguous()
  width=triton.next_power_of_2(int(population.max()));assert width<=512
  order=labels.argsort(stable=True);starts=population.long().cumsum(0)-population.long();rank=torch.arange(n,device=x.device)-starts[labels[order]]
  members=torch.full((f,width),-1,device=x.device,dtype=torch.int32);members[labels[order],rank]=order.int()
  geometry=dict(width=1,probe_member_width=width,members=members,norm2=norm,rep_distance=distance,current_dense_prior=True,shared_value_loss=True,fuse_state_kernel=True,rep_history_execution=c.rep_history_execution,drop_bins=[(1,torch.arange(f,device=x.device,dtype=torch.int32))])
 with stage(s,'history_key_preparation'),torch.autocast(x.device.type,enabled=False):prepared=prepare_keys(s,k)
 shape=(s.groups,n,x.device)
 if getattr(s,'_teacher_buffer_shape',None)!=shape:
  if 2*s.groups*n*4>4*1024**3:raise ValueError('Temporal cache exceeds explicit4GiB limit')
  s._teacher_buffers=[torch.empty((s.groups,n),device=x.device),torch.empty((s.groups,n),device=x.device)];s._teacher_buffer_shape=shape
 destination=s._teacher_buffers[layer%2];r=prepared['regions'];next_mass=torch.empty((s.groups,r),device=x.device)
 if c.teacher_history_layer_observation:
  with stage(s,'current_coarse_observation'),torch.autocast(x.device.type,enabled=False):layer_observation=prob.observe(q,prepared,0,s.groups,c.query_group_size,probes,o.scale)
  next_coarse=layer_observation['coarse'];next_logs=layer_observation['sample_logprob']
 else:next_coarse=torch.empty_like(next_mass);next_logs=torch.empty((s.groups,r,c.teacher_history_samples),device=x.device)
 if layer==0:
  with stage(s,'bootstrap_full_key_probability'),torch.autocast(x.device.type,enabled=False):
   bootstrap,positions=probability(q,k,s.patches,c.query_group_size,4,o.scale);destination.copy_(bootstrap);del bootstrap
 output=torch.empty_like(v);history=[];records=[];scratch=0
 for start in range(0,s.groups,c.window_groups):
  g=min(c.window_groups,s.groups-start);first=start*c.query_group_size;end=min(s.n,(start+g)*c.query_group_size)
  if c.teacher_history_layer_observation:observation={key:value[start:start+g] for key,value in layer_observation.items()}
  else:
   with stage(s,'current_coarse_observation'),torch.autocast(x.device.type,enabled=False):observation=prob.observe(q,prepared,start,g,c.query_group_size,probes,o.scale)
  with stage(s,'history_probability_prediction'),torch.autocast(x.device.type,enabled=False):
   if layer==0:prior=destination[start:start+g];mass=prob.block_mass(prior,block)
   else:
    previous={key:value[start:start+g] for key,value in s.streamed_history_cache.items()}
    prior,mass=prob.predict(previous,observation,prepared['assignment'],prepared['sizes'],block,c.streamed_history_prior_weight,c.teacher_history_anchor_gain)
  with stage(s,'current_prior_state_selection'),torch.autocast(x.device.type,enabled=False):
   old=torch.full((g,1),-1,device=x.device,dtype=torch.int32) if s.history_ids is None else s.history_ids[start:start+g].contiguous()
   pi=torch.empty((g,f),device=x.device);allowed=torch.ones((g,1),device=x.device,dtype=torch.bool);uniform=torch.zeros((g,1),device=x.device)
   ids,real,weights,record=select(s,layer,prior,pi,old,allowed,z,reps,labels,population,geometry,uniform=uniform)
  actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous();capture=0<layer<23
  with stage(s,'sparse_attention_only'):result,evidence=attention(q[:,first:end],k,v,actual,s.specials,weights,o.scale,probes,capture,c.attention_warps,c.attention_stages)
  with stage(s,'teacher_cache_update'),torch.autocast(x.device.type,enabled=False):
   if capture:
    prob.update(prior,mass,ids,evidence,destination[start:start+g],len(s.specials),block,c.teacher_history_update_gain)
    scratch=max(scratch,sum(value.numel()*value.element_size() for key,value in evidence.items() if torch.is_tensor(value)))
   if layer<23:
    next_mass[start:start+g].copy_(mass)
    if not c.teacher_history_layer_observation:next_coarse[start:start+g].copy_(observation['coarse']);next_logs[start:start+g].copy_(observation['sample_logprob'])
  with stage(s,'output_and_history_store'):output[:,first:end]=result;history.append(ids.clone());records.append(record)
 with stage(s,'history_validation_and_statistics'):
  joined={key:torch.cat([rr[key] for rr in records]) for key in records[0]}
  if bool(joined['implementation_validation_error'].any()):raise RuntimeError('Support legality/budget failed')
  s.history_ids=torch.cat(history)
  if layer<23:s.streamed_history_cache=dict(dense_probability=destination,block_mass=next_mass,coarse=next_coarse,sample_logprob=next_logs)
  rows=(s.n-torch.arange(s.groups,device=x.device)*c.query_group_size).clamp_max(c.query_group_size);raw=((joined['counts']+len(s.specials))*rows).sum()
  record=dict(total_edge_retention=c.total_edge_retention,layer=layer,query_group_size=c.query_group_size,query_groups=s.groups,fine_clusters=f,patch_budget=s.budget,special_keys=len(s.specials),counts_mean=joined['counts'].float().mean(),unused_budget_mean=(s.budget-joined['counts']).float().mean(),actual_unique_density=raw/s.n**2,final_unique_qk_per_head=raw,direct_logical_main_qk_per_head=raw,geometry_width=width,rep_history_loss='A',rep_history_execution=c.rep_history_execution,shared_value_features=c.shared_value_features,probability_source='L0 fast4 bootstrap;later persistent historical fine density with current64-Key block and8-anchor change',history_probabilities_used=layer>0,history_cross_layer_head_matching=False,history_key_block_size=block,history_key_blocks=r,current_probability_probes=4 if layer==0 else 0,current_probe_qk_per_head=s.groups*4*(s.n+n) if layer==0 else 0,coarse_observation_qk_per_head=s.groups*probes*(2*r+len(s.specials)+r*c.teacher_history_samples),selected_edge_recompute_qk_per_head=0,history_evidence_backend=c.teacher_history_attention_backend,history_evidence_probes=probes if capture else 0,history_evidence_scratch_peak_bytes=scratch,history_cache_written=layer<23,history_probability_cache_bytes=sum(value.numel()*value.element_size() for value in s.streamed_history_cache.values()),history_cache_allocated_dense_bytes=2*s.groups*n*4,**{key+'_mean':value.float().mean() for key,value in joined.items() if key!='counts'})
 if c.dense_special_queries:
  from .dense_special_queries import correct_output
  with stage(s,'dense_special_query_correction'):output,dense_ids=correct_output(q,k,v,output,s.semantic_specials,c.dense_special_queries,o.scale)
  extra=(s.n-joined['counts'][dense_ids//c.query_group_size]-len(s.specials)).sum();record.update(dense_special_query_count=len(dense_ids),dense_special_query_physical_qk_per_head=len(dense_ids)*s.n,actual_unique_density=(raw+extra)/s.n**2,final_unique_qk_per_head=raw+extra)
 else:record['dense_special_query_count']=0
 s.previous_layer=layer;s.records.append(record)
 return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))
