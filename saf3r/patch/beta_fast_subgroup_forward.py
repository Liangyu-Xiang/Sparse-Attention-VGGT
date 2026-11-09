"""Global headmean prior with shrunken log-importance history; no block mass lock.

Only previous actual exported QK is cached. No extra QK or kernel changes.
"""
import os
import torch,triton
from .fused_shared_av import stage
from .beta_fast_importance import estimate as model_estimate,observe_subgroup as observe
from saf3r.triton.beta_subgroup_attention import pool_queries
from .one_layer_edge_history import mean_valid,decode,compensation_diagnostic

class UnsafeAttentionGauge(RuntimeError):
    """The fixed attention gauge exceeded its numerical range."""

def forward(module,x,pos):
    from .reuse_residual_exploration import prepare_qkv
    from .layer_retention_schedule import layer_budget
    from .head_value_sketch import projection,features as sketch
    from .coarse_spatial_layout import fixed_blocks,fixed_blocks_at_most
    from .beta_unified_selection import select
    from saf3r.triton.beta_subgroup_reuse_attention import attention
    o,s=module.original,module.state;c=s.config;layer=module.layer
    if module.training or len(x)!=1 or layer!=s.previous_layer+1:raise ValueError('Ordered inference required')
    s.budget=layer_budget(c,layer,s.n,len(s.specials));probes=4
    if layer==0:
        s._beta_observed=None;s._one_layer_source=-1;s._beta_previous_coarse=None;s._quality_layout=None
        if getattr(s,'_compensation_diagnostic',False):
            s._compensation_groups=torch.linspace(0,s.groups-1,8,device=x.device).long().unique();s._compensation_rows=[]
    if s._one_layer_source!=layer-1:raise RuntimeError('History is not exactly the previous layer')
    with stage(s,'shared_qkv_and_rope'):
        position_count=max(s.height,s.width)+1 if os.environ.get("BETA_ROPE_EXTENT")=="1" else None
        q,k,v=prepare_qkv(o,x,pos,position_count=position_count);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
    correction_stream=None
    if c.dense_special_queries and os.environ.get("BETA_OVERLAP_SPECIAL_QUERY")=="1":
        from .dense_special_queries import compute_correction
        correction_main=torch.cuda.current_stream(x.device)
        correction_stream=getattr(s,'_dense_correction_stream',None)
        if correction_stream is None:
            correction_stream=s._dense_correction_stream=torch.cuda.Stream(device=x.device)
        correction_stream.wait_stream(correction_main)
        with torch.cuda.stream(correction_stream),stage(s,'dense_special_query_compute'):
            dense_correction,dense_ids=compute_correction(q,k,v,s.semantic_specials,c.dense_special_queries,o.scale)
        for tensor in (q,k,v,s.semantic_specials):
            tensor.record_stream(correction_stream)
    with stage(s,'current_value_geometry'),torch.autocast(x.device.type,enabled=False):
        key=(len(v),v.shape[-1],c.shared_value_head_sketch_dim,v.device)
        if getattr(s,'_value_sketch_shape',None)!=key:s._value_sketch_matrix=projection(*key[:3],key[3]);s._value_sketch_shape=key
        if os.environ.get("BETA_VALUE_GATHER")=="1":
            from saf3r.triton.value_sketch_gather import gather
            features=gather(v,s._value_sketch_matrix,s.patches)
        else:
            features=sketch(v,s._value_sketch_matrix)[s.patches].contiguous()
        s.fine.update(features,layer);labels,reps=s.fine.labels,s.fine.reps;f=len(reps);n=len(labels)
        population=torch.bincount(labels,minlength=f).float();z=(features/features.square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous()
        norm=z.square().sum(-1).contiguous();distance=(z-z[reps[labels]]).square().sum(-1).contiguous()
        width=triton.next_power_of_2(int(population.max()));assert width<=512
        order=labels.argsort(stable=True);starts=population.long().cumsum(0)-population.long();rank=torch.arange(n,device=x.device)-starts[labels[order]]
        members=torch.full((f,width),-1,device=x.device,dtype=torch.int32);members[labels[order],rank]=order.int()
        geometry=dict(width=1,probe_member_width=width,members=members,norm2=norm,rep_distance=distance,current_dense_prior=True,shared_value_loss=True,fuse_state_kernel=True,rep_history_execution='population',drop_bins=[(1,torch.arange(f,device=x.device,dtype=torch.int32))])
    with stage(s,'coarse_key_pooling'),torch.autocast(x.device.type,enabled=False):
        if layer==0:
            layout=fixed_blocks_at_most if getattr(s,'_group_mean_key_block_rounding','floor')=='ceil' else fixed_blocks
            grid,sizes=layout(n,s.height*s.width,c.history_key_block_size,x.device)
            positions_all=(torch.arange(s.groups,device=x.device)[:,None]*256+torch.arange(probes,device=x.device)[None]*64+32).clamp_max(s.n-1)
            special_mask=torch.zeros(s.n,device=x.device,dtype=torch.bool);special_mask[s.specials]=True
            eligible_all=torch.nn.functional.pad((~special_mask).float(),(0,s.groups*256-s.n)).reshape(s.groups,4,64).sum(2)
            s._ordinary_qcounts_parts=eligible_all.contiguous()
            s._ordinary_qmask=~special_mask
            s._ordinary_qcounts=torch.nn.functional.pad((~special_mask).float(),(0,s.groups*256-s.n)).reshape(s.groups,256).sum(1)
            if not bool(eligible_all.any(1).all()):raise RuntimeError('No ordinary Query probe in a group')
            per_frame=len(sizes)//(n//(s.height*s.width))
            amin=(s.height*s.width)//per_frame;amax=((s.height*s.width)+per_frame-1)//per_frame
            s._quality_layout=(grid,sizes,positions_all,eligible_all,amin,amax)
            s._history16_positions=(torch.arange(s.groups,device=x.device)[:,None]*256+torch.arange(16,device=x.device)[None]*16+8).clamp_max(s.n-1)
            s._history16_eligible=~special_mask[s._history16_positions]
        grid,sizes,positions_all,eligible_all,amin,amax=s._quality_layout;r=len(sizes)
        if os.environ.get("BETA_FUSED_KEY_POOL")=="1":
            from saf3r.triton.fused_coarse_key_pool import pool
            pooled=pool(k,s.patches,sizes,amax)
        else:
            patch=k[:,s.patches].float()
            if amin==amax:
                blocks=patch.reshape(len(k),r,amin,k.shape[-1])
                pooled=blocks.mean(2)
            else:
                pooled=torch.zeros(len(k),r,k.shape[-1],device=x.device);pooled.index_add_(1,grid,patch);pooled/=sizes[None,:,None]
    s._mean_query=pool_queries(q,s._ordinary_qmask,s._ordinary_qcounts_parts)
    use_compact=os.environ.get("BETA_COMPACT_PRIOR")=="1"
    if use_compact:from .beta_unified_importance import estimate_compact
    use_exact_expanded=os.environ.get("BETA_EXACT_EXPANDED")=="1"
    if use_exact_expanded:from .beta_unified_importance import estimate_exact_expanded
    log_sizes=sizes.log()[None,None,None]
    token_major=os.environ.get("BETA_TOKEN_MAJOR_OUTPUT")=="1"
    if token_major:
        output=torch.empty((s.n,v.shape[0],v.shape[2]),device=v.device,dtype=v.dtype).transpose(0,1)
    else:
        output=torch.empty_like(v)
    history=[];records=[];next_cache=[];next_coarse=[];beta_summary=[];update_checks=[];scratch=0
    # Original windows64 are retained. Fresh updates use only HxGx4xselected scratch.
    fuse_support_map=(os.environ.get("BETA_MAP_ALIGNED")=="1"
                      and getattr(s,"_logit_capture",None) is None
                      and not getattr(s,"_compensation_diagnostic",False))
    def select_window(start):
        g=min(c.window_groups,s.groups-start)
        with stage(s,'current_coarse_observation'),torch.autocast(x.device.type,enabled=False):
            positions=positions_all[start:start+g];eligible=eligible_all[start:start+g]
            qp=s._mean_query[:,start:start+g];mean=torch.einsum('hgpd,hrd->hgpr',qp,pooled)*o.scale
            coarse_logits=mean+log_sizes
            coarse=mean_valid(coarse_logits.softmax(-1),eligible)
            previous=None if layer==0 else s._beta_previous_coarse[start:start+g]
            old=torch.full((g,1),-1,device=x.device,dtype=torch.int32) if s.history_ids is None else s.history_ids[start:start+g].contiguous()
            observed=None if layer==0 else s._beta_observed[start:start+g]
            if use_exact_expanded:
                prior,beta=estimate_exact_expanded(coarse,previous,old,observed,grid,sizes,fused_lift=os.environ.get("BETA_FUSED_EXACT_LIFT")=="1")
            elif use_compact:
                prior,beta=estimate_compact(coarse,previous,old,observed,grid,sizes)
            else:
                prior,beta=model_estimate(mean,coarse,previous,old,observed,grid,sizes,layer)
            shift=coarse_logits.logsumexp(-1).contiguous()
            next_coarse.append(coarse);beta_summary.append((beta.sum(),(coarse*beta).sum(),beta.numel(),g))
        with stage(s,'current_prior_state_selection'),torch.autocast(x.device.type,enabled=False):
            pi=torch.empty((g,f),device=x.device);allowed=torch.ones((g,1),device=x.device,dtype=torch.bool);uniform=torch.zeros((g,1),device=x.device)
            ids,real,weights,record=select(s,layer,prior,pi,old,allowed,z,reps,labels,population,geometry)
        if fuse_support_map:
            actual=ids
        elif os.environ.get("BETA_SUPPORT_MAP")=="1":
            from saf3r.triton.support_token_ids import map_support
            actual=map_support(ids,s.patches)
        else:
            actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
        return g,positions,eligible,mean,shift,ids,weights,record,actual,old,real

    overlap=(os.environ.get("BETA_OVERLAP_SELECTION")=="1"
             and getattr(s,'_logit_capture',None) is None
             and not getattr(s,'_compensation_diagnostic',False))
    if overlap:
        main=torch.cuda.current_stream(x.device)
        side=getattr(s,'_selection_stream',None)
        if side is None:
            side=s._selection_stream=torch.cuda.Stream(device=x.device)
        side.wait_stream(main)

        def prefetch(start):
            with torch.cuda.stream(side):
                g,positions,eligible,mean,shift,ids,weights,record,actual,old,real=select_window(start)
                # Graph replay reuses its output buffers. Preserve this window
                # before enqueuing another replay on the selection stream.
                ids,weights=ids.clone(),weights.clone()
                if fuse_support_map:actual=ids
                ready=torch.cuda.Event()
                ready.record()
            return ready,(g,positions,eligible,mean,shift,ids,weights,record,actual)

        pending=prefetch(0)
    for start in range(0,s.groups,c.window_groups):
        if overlap:
            ready,window=pending
            main.wait_event(ready)
            g,positions,eligible,mean,shift,ids,weights,record,actual=window
            for tensor in (mean,shift,ids,weights,actual,*record.values()):
                tensor.record_stream(main)
            upcoming=start+c.window_groups
            if upcoming<s.groups:
                pending=prefetch(upcoming)
        else:
            g,positions,eligible,mean,shift,ids,weights,record,actual,old,real=select_window(start)
        first=start*c.query_group_size
        end=min(s.n,(start+g)*c.query_group_size)
        with stage(s,'sparse_attention_only'):
            tail_args=dict(valid_counts=record["counts"].contiguous()) if getattr(s,"trim_attention_tail",False) else {}
            if fuse_support_map:tail_args["patch_map"]=s.patches
            result,evidence=attention(q[:,first:end],k,v,actual,s.specials,weights,o.scale,s._ordinary_qmask[first:end],s._ordinary_qcounts_parts[start:start+g],layer<23,c.attention_warps,c.attention_stages,shift=shift,output=output[:,first:end],**tail_args)
        if getattr(s,'_logit_capture',None) is not None:s._logit_capture.record(module,locals())
        if layer<23:
            with stage(s,'fresh_previous_layer_cache'),torch.autocast(x.device.type,enabled=False):
                selected_logits=evidence['mean_exp']
                partial_sum=evidence.get('history_sum')
                history_sum=None if partial_sum is None else (partial_sum[0] if len(partial_sum)==1 else partial_sum.sum(0))
                fresh=observe(mean,grid,sizes,ids,selected_logits,eligible,shift=shift,history_sum=history_sum)
                update_checks.append(torch.isfinite(fresh).all())
                next_cache.append(fresh);scratch=max(scratch,selected_logits.numel()*4+sum(t.numel()*t.element_size() for t in evidence.values() if torch.is_tensor(t)))
        if getattr(s,'_compensation_diagnostic',False):
            with torch.autocast(x.device.type,enabled=False):compensation_diagnostic(s,layer,start,q,k,v,actual,ids,weights,labels,reps,population,result,o.scale,positions,eligible,o.proj.weight.float())
        history.append(ids.clone());records.append(record)
    with stage(s,'history_validation_and_statistics'):
        joined={key:torch.cat([rr[key] for rr in records]) for key in records[0]}
        invalid=joined['implementation_validation_error'].any()
        if update_checks:invalid=invalid | ~torch.stack(update_checks).all()
        if getattr(s,'defer_fixed_gauge_check',False):
            bad_output=~torch.isfinite(output).all()
            if bool(invalid|bad_output):
                if bool(bad_output):raise UnsafeAttentionGauge('Fixed attention gauge exceeded its numerical range')
                raise RuntimeError('Support/budget/cache legality or finite check failed')
        elif bool(invalid):raise RuntimeError('Support/budget/cache legality or finite check failed')
        s.history_ids=torch.cat(history);s._beta_observed=torch.cat(next_cache,0) if layer<23 else None;s._one_layer_source=layer;s._beta_previous_coarse=torch.cat(next_coarse) if layer<23 else None
        rows=(s.n-torch.arange(s.groups,device=x.device)*c.query_group_size).clamp_max(c.query_group_size);raw=((joined['counts']+len(s.specials))*rows).sum()
        record=dict(layer=layer,query_group_size=c.query_group_size,query_groups=s.groups,fine_clusters=f,patch_budget=s.budget,special_keys=len(s.specials),actual_unique_density=raw/s.n**2,final_unique_qk_per_head=raw,direct_logical_main_qk_per_head=raw,geometry_width=width,rep_history_loss='A',rep_history_execution='population',execution_compensation='sqrt_population_prior_residual',refinement=s._quality_name,history_gain_confidence=1.,unknown_response_model='none;unobserved keys never enter historical cache',history_cache_kind='previous actual ids and headmean importance normalized against observed plus coarse unknown partition;no crosslayer head axis',probability_source='global current headmean coarse prior times exp(.5*unified beta*log(previous headmean importance/previous coarse));global normalization;no block centering or mass lock',beta_mean=torch.stack([t[0] for t in beta_summary]).sum()/sum(t[2] for t in beta_summary),beta_mass_weighted_mean=torch.stack([t[1] for t in beta_summary]).sum()/sum(t[3] for t in beta_summary),history_source_layer=layer-1 if layer else -1,max_direct_history_age=1 if layer else 0,unobserved_fine_history_inherited=False,history_key_block_size=c.history_key_block_size,actual_block_min=amin,actual_block_max=amax,key_blocks_total=r,current_fine_patch_qk_before_mask=0,validation_query_probes=0,history_observation_probes=0,history_observation_summary='mean exp(raw QK-coarse partition) within each contiguous64Query subgroup;all ordinary Queries covered;perhead/subgroup normalization before weightedheadmean;mask group256 unchanged',validation_source='disabled by user',validation_may_repeat_formal_selected_probe_qk=False,coarse_qk_per_head=s.groups*probes*r,selected_edge_recompute_qk_per_head=0,history_cache_bytes=0 if s._beta_observed is None else (s._beta_observed.numel()+s._beta_previous_coarse.numel()+s.history_ids.numel())*4,history_evidence_scratch_peak_bytes=scratch,**{key+'_mean':value.float().mean() for key,value in joined.items() if key!='counts'})
    if c.dense_special_queries:
        from .dense_special_queries import correct_output
        with stage(s,'dense_special_query_correction'):
            if correction_stream is None:
                output,dense_ids=correct_output(q,k,v,output,s.semantic_specials,c.dense_special_queries,o.scale)
            else:
                correction_main.wait_stream(correction_stream)
                if dense_correction is not None:
                    output[:,dense_ids]=dense_correction[0]
                    dense_correction.record_stream(correction_main)
                dense_ids.record_stream(correction_main)
        extra=(s.n-joined['counts'][dense_ids//c.query_group_size]-len(s.specials)).sum();record.update(dense_special_query_count=len(dense_ids),actual_unique_density=(raw+extra)/s.n**2,final_unique_qk_per_head=raw+extra)
    s.previous_layer=layer;s.records.append(record)
    return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))



def install():
    from . import current_probe_reuse
    def run(module,x,pos):
        module.state._quality_name='beta_unified_subgroup_half'
        module.state._group_mean_key_block_rounding='ceil'
        return forward(module,x,pos)
    current_probe_reuse.forward=run
