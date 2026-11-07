"""Fresh previous-layer-only edge history; no inherited unobserved fine density.

Current ordinary QK is coarse-only before state selection. Executed raw QK is
exported inline by the existing AV kernel. No Dense initialization or anchors.
"""
import torch,triton
from .fused_shared_av import stage


def mean_valid(x, valid):
    # x: H,G,P,...; valid: G,P. Normalize before averaging heads/probes.
    weight=valid.float()/valid.sum(1).clamp_min(1)[:,None]
    return (x*weight[None,...,*([None]*(x.ndim-3))]).sum(2).mean(0)


def fresh_conditional(mean_logits,grid,sizes,ids,selected_logits,probe_valid):
    """Uses ONLY this layer's coarse mean and this layer's executed raw edges."""
    h,g,p,r=mean_logits.shape;n=len(grid);valid=ids>=0;safe=ids.clamp_min(0).long()
    index=grid[safe];idx=index[None,:,None].expand_as(selected_logits)
    count=torch.zeros(g,r,device=ids.device).scatter_add_(1,index,valid.float())
    remaining=sizes[None]-count
    if (remaining<0).any():raise ValueError('Duplicated selected support')
    raw=selected_logits.masked_fill(~valid[None,:,None],-torch.inf)
    total_logits=torch.zeros_like(mean_logits).scatter_add_(3,idx,raw.masked_fill(~valid[None,:,None],0.))
    residual_mean=(mean_logits*sizes[None,None,None]-total_logits)/remaining.clamp_min(1)[None,:,None]
    log_remaining=(residual_mean+remaining.clamp_min(1).log()[None,:,None]).masked_fill((remaining==0)[None,:,None],-torch.inf)
    logz=torch.logaddexp(raw.logsumexp(-1),log_remaining.logsumexp(-1))
    selected_prob=mean_valid((raw-logz[...,None]).exp(),probe_valid).masked_fill(~valid,0.)
    unknown_mass=mean_valid((log_remaining-logz[...,None]).exp(),probe_valid)
    uniform=(unknown_mass/remaining.clamp_min(1))[:,grid]
    tombstone=safe.masked_fill(~valid,n)
    posterior=torch.cat((uniform,uniform.new_zeros(g,1)),1).scatter_(1,tombstone,selected_prob)[:,:n]
    block_mass=torch.zeros(g,r,device=ids.device).scatter_add_(1,grid[None].expand(g,-1),posterior)
    phi=posterior/block_mass[:,grid].clamp_min(1e-30)
    phi=torch.where((block_mass==0)[:,grid],1/sizes[grid][None],phi)
    # Correct FP32 reduction drift without retaining any previous fine estimate.
    sums=torch.zeros(g,r,device=ids.device).scatter_add_(1,grid[None].expand(g,-1),phi)
    return phi/sums[:,grid].clamp_min(1e-30)


def decode(evidence,specials,a):
    encoded=evidence['encoded'];g,h,p,total=encoded.shape;tile=evidence['block_keys']
    slots=torch.arange(specials,specials+a,device=encoded.device)
    return (encoded[...,specials:].float()+evidence['shifts'][...,slots//tile]).permute(1,0,2,3).contiguous()


def compensation_diagnostic(s,layer,start,q,k,v,actual,ids,weights,labels,reps,population,result,scale,positions,probe_valid,wo):
    """Fixed support/state; alternate weights never affect model/cache or timing."""
    chosen=s._compensation_groups
    local=chosen[(chosen>=start)&(chosen<start+len(ids))]-start
    if not len(local):return
    for group in local.tolist():
        valid=probe_valid[group];pos=positions[group,valid];qp=q[:,pos].float()
        logits=torch.bmm(qp,k.float().transpose(1,2))*scale
        target=logits.softmax(-1)@v.float()
        support=torch.cat((s.specials,actual[group].long()));good=support>=0;safe=support.clamp_min(0)
        raw=logits[:,:,safe].masked_fill(~good[None,None],-torch.inf)
        oldweights=torch.cat((weights.new_ones(len(s.specials)),weights[group]))
        patchsafe=ids[group].clamp_min(0).long();patchvalid=ids[group]>=0;cluster=labels[patchsafe]
        count=torch.bincount(cluster[patchvalid],minlength=len(reps)).float()
        population_weight=torch.where(patchsafe==reps[cluster],population[cluster]-count[cluster]+1,1.)
        alternatives={'original':oldweights,'unit':torch.ones_like(oldweights),'population_residual':torch.cat((weights.new_ones(len(s.specials)),population_weight))}
        def error(y):
            diff=(y-target).transpose(0,1).reshape(-1,len(q)*q.shape[-1]);base=target.transpose(0,1).reshape_as(diff)
            return float((torch.nn.functional.linear(diff,wo).square().sum()/torch.nn.functional.linear(base,wo).square().sum().clamp_min(1e-30)).sqrt())
        row=dict(layer=layer,query_group=start+group,queries=len(pos),fixed_selected_patch_edges=int(patchvalid.sum()))
        ref=None
        for name,w in alternatives.items():
            y=(raw+w.clamp_min(1e-30).log()[None,None]).softmax(-1)@v[:,safe].float()
            row[name+'_relative_l2']=error(y)
            if name=='original':ref=y
        localpos=pos-start*s.config.query_group_size
        kernel=result[:,localpos].float()
        row['actual_kernel_relative_l2']=error(kernel)
        row['kernel_vs_reconstruction_relative_l2']=float(((kernel-ref).square().sum()/ref.square().sum().clamp_min(1e-30)).sqrt())
        s._compensation_rows.append(row)


def forward(module,x,pos):
    from .reuse_residual_exploration import prepare_qkv
    from .layer_retention_schedule import layer_budget
    from .head_value_sketch import projection,features as sketch
    from .coarse_spatial_layout import fixed_blocks
    from .reuse_window_graph import select
    from saf3r.triton.teacher_history_attention import attention
    o,s=module.original,module.state;c=s.config;layer=module.layer
    if module.training or len(x)!=1 or layer!=s.previous_layer+1:raise ValueError('Ordered inference required')
    s.budget=layer_budget(c,layer,s.n,len(s.specials));probes=4
    if layer==0:
        s._one_layer_cache=None;s._one_layer_source=-1
        if getattr(s,'_compensation_diagnostic',False):
            s._compensation_groups=torch.linspace(0,s.groups-1,8,device=x.device).long().unique();s._compensation_rows=[]
    if s._one_layer_source!=layer-1:raise RuntimeError('History is not exactly the previous layer')
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
    with stage(s,'coarse_key_pooling'),torch.autocast(x.device.type,enabled=False):
        grid,sizes=fixed_blocks(n,s.height*s.width,c.history_key_block_size,x.device);r=len(sizes)
        patch=k[:,s.patches].float();pooled=torch.zeros(len(k),r,k.shape[-1],device=x.device);pooled.index_add_(1,grid,patch);pooled/=sizes[None,:,None]
        is_special=torch.zeros(s.n,device=x.device,dtype=torch.bool);is_special[s.specials]=True
    output=torch.empty_like(v);history=[];records=[];next_cache=[];scratch=0
    # Original windows64 are retained. Fresh updates use only HxGx4xselected scratch.
    for start in range(0,s.groups,c.window_groups):
        g=min(c.window_groups,s.groups-start);first=start*c.query_group_size;end=min(s.n,(start+g)*c.query_group_size)
        with stage(s,'current_coarse_observation'),torch.autocast(x.device.type,enabled=False):
            positions=(torch.arange(start,start+g,device=x.device)[:,None]*256+torch.arange(probes,device=x.device)[None]*64+32).clamp_max(s.n-1)
            eligible=~is_special[positions]
            if not bool(eligible.any(1).all()):raise RuntimeError('No ordinary Query probe in a group')
            qp=q[:,positions].float();mean=torch.einsum('hgpd,hrd->hgpr',qp,pooled)*o.scale
            coarse=mean_valid((mean+sizes.log()[None,None,None]).softmax(-1),eligible)
            phi=(1/sizes[grid])[None].expand(g,-1) if layer==0 else s._one_layer_cache[start:start+g]
            prior=(coarse[:,grid]*phi).contiguous()
        with stage(s,'current_prior_state_selection'),torch.autocast(x.device.type,enabled=False):
            old=torch.full((g,1),-1,device=x.device,dtype=torch.int32) if s.history_ids is None else s.history_ids[start:start+g].contiguous()
            pi=torch.empty((g,f),device=x.device);allowed=torch.ones((g,1),device=x.device,dtype=torch.bool);uniform=torch.zeros((g,1),device=x.device)
            ids,real,weights,record=select(s,layer,prior,pi,old,allowed,z,reps,labels,population,geometry,uniform=uniform)
        actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
        with stage(s,'sparse_attention_only'):
            result,evidence=attention(q[:,first:end],k,v,actual,s.specials,weights,o.scale,probes,layer<23,c.attention_warps,c.attention_stages)
        if layer<23:
            with stage(s,'fresh_previous_layer_cache'),torch.autocast(x.device.type,enabled=False):
                selected_logits=decode(evidence,len(s.specials),ids.shape[1])
                fresh=fresh_conditional(mean,grid,sizes,ids,selected_logits,eligible)
                if not torch.isfinite(fresh).all():raise RuntimeError('Nonfinite fresh cache')
                next_cache.append(fresh);scratch=max(scratch,selected_logits.numel()*4+sum(t.numel()*t.element_size() for t in evidence.values() if torch.is_tensor(t)))
        if getattr(s,'_compensation_diagnostic',False):
            with torch.autocast(x.device.type,enabled=False):compensation_diagnostic(s,layer,start,q,k,v,actual,ids,weights,labels,reps,population,result,o.scale,positions,eligible,o.proj.weight.float())
        output[:,first:end]=result;history.append(ids.clone());records.append(record)
    with stage(s,'history_validation_and_statistics'):
        joined={key:torch.cat([rr[key] for rr in records]) for key in records[0]}
        if bool(joined['implementation_validation_error'].any()):raise RuntimeError('Support legality/budget failed')
        s.history_ids=torch.cat(history);s._one_layer_cache=torch.cat(next_cache) if layer<23 else None;s._one_layer_source=layer
        rows=(s.n-torch.arange(s.groups,device=x.device)*c.query_group_size).clamp_max(c.query_group_size);raw=((joined['counts']+len(s.specials))*rows).sum()
        record=dict(layer=layer,query_group_size=c.query_group_size,query_groups=s.groups,fine_clusters=f,patch_budget=s.budget,special_keys=len(s.specials),actual_unique_density=raw/s.n**2,final_unique_qk_per_head=raw,direct_logical_main_qk_per_head=raw,geometry_width=width,rep_history_loss='A',rep_history_execution=c.rep_history_execution,probability_source='current patch-only pooled blocks x fresh previous-layer selected-aware conditional',history_source_layer=layer-1 if layer else -1,max_direct_history_age=1 if layer else 0,unobserved_fine_history_inherited=False,history_key_block_size=c.history_key_block_size,actual_block_min=int(sizes.min()),actual_block_max=int(sizes.max()),key_blocks_total=r,current_fine_patch_qk_before_mask=0,coarse_qk_per_head=s.groups*probes*r,selected_edge_recompute_qk_per_head=0,history_cache_bytes=0 if s._one_layer_cache is None else s._one_layer_cache.numel()*4,history_evidence_scratch_peak_bytes=scratch,**{key+'_mean':value.float().mean() for key,value in joined.items() if key!='counts'})
    if c.dense_special_queries:
        from .dense_special_queries import correct_output
        with stage(s,'dense_special_query_correction'):output,dense_ids=correct_output(q,k,v,output,s.semantic_specials,c.dense_special_queries,o.scale)
        extra=(s.n-joined['counts'][dense_ids//c.query_group_size]-len(s.specials)).sum();record.update(dense_special_query_count=len(dense_ids),actual_unique_density=(raw+extra)/s.n**2,final_unique_qk_per_head=raw+extra)
    s.previous_layer=layer;s.records.append(record)
    return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install():
    from . import current_probe_reuse
    current_probe_reuse.forward=forward
