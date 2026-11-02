"""Fixed, truth-free guarded approximation of compact reuse allocation."""
import math
import torch
from torch import nn
from .reuse_residual_exploration import ReuseExploreState,top_history,budget_states,pack_support
from .reuse_cosine_distortion import normalized_values,relative_logit_prior,cosine_states,transfer_weights
from .fused_shared_av import stage
from .region_history_attention import prepare_qkv
from saf3r.triton import reuse_compact as compact
from saf3r.triton.reuse_guarded_allocation import budget as guarded_budget
from saf3r.triton import reuse_shared_grid as shared


class GuardedState(ReuseExploreState):
    def reset(self,*args):
        super().reset(*args)
        self.max_budget=self.budget
        self.cache_ids=None;self.cache_weights=None;self.cache_cap=None;self.cache_layer=None
        self.guard_records=[]


@torch.autocast("cuda",enabled=False)
def allocate(s,q,pooled,overlap,population,zvalues,geometry,group_ids,layer,cap,iterations,diagnostics=False):
    """Identical current-input state family; only cap/dual iterations vary."""
    c=s.config;g=len(group_ids);device=q.device;f=s.fine.base_count
    if layer==0 and cap==s.max_budget:
        ids=s.fine.reps[None].expand(g,-1).int().contiguous()
        weights=population[None].expand(g,-1).contiguous()
        counts=torch.full((g,),f,device=device,dtype=torch.int32)
        return ids,weights,dict(counts=counts,reuse_edges=torch.zeros(g,device=device),representatives=counts,
            exact_edges=torch.zeros(g,device=device),dropped_clusters=torch.zeros(g,device=device))
    probes=(group_ids[:,None]*32+torch.tensor([4,12,20,28],device=device)).clamp_max(s.n-1)
    pq=q[:,probes]
    coarse=(pq@pooled[:,None].transpose(-1,-2)).float()*s.scale+s.grid_sizes.log()
    probability=coarse.softmax(-1).mean((0,2))
    if layer:
        previous=s.previous_grid[group_ids].float()
        beta=torch.where(previous>1e-12,(probability/previous.clamp_min(1e-12)).clamp_max(1),0)
        demand=(probability-beta*s.previous_real_grid[group_ids].float()).clamp_min(0)
        residual=torch.sparse.mm(overlap,demand.T).T
        candidates=max(1,math.ceil(f*c.candidate_fraction))
        chosen=residual.topk(candidates,dim=1).indices
        allowed=torch.zeros((g,f),device=device,dtype=torch.bool).scatter_(1,chosen,True)
        old_ids=s.history_ids[group_ids].contiguous();delta=s.history_relative[group_ids].contiguous()
    else:
        # Low-budget seed: Rep/Drop only, with current coarse prior. Keeping F
        # fixed at the original20% does not imply that all Reps fit a15% cap.
        old_ids=torch.empty((g,0),device=device,dtype=torch.int32);delta=torch.empty((g,0),device=device)
        allowed=torch.zeros((g,f),device=device,dtype=torch.bool)
    prior,pi=relative_logit_prior(probability,s.grid,s.grid_sizes,old_ids,delta,s.fine.labels,f)
    if geometry is not None:
        hids,discarded=compact.top_history(old_ids,prior,s.fine.reps,geometry,c.history_per_cluster)
    else:
        observed=prior.gather(1,old_ids.long().clamp_min(0)).masked_fill(old_ids<0,0)
        hids,_,discarded=top_history(old_ids,observed,torch.ones_like(probability),s.grid,s.fine.labels,s.fine.reps,c.history_per_cluster)
    errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry=geometry)
    objective=geometry['objective'] if geometry is not None else errors-errors[:,:,1:2]
    upper=errors[:,:,0].amax(1)+1e-12
    if geometry is not None:
        selected,count=guarded_budget(objective,costs,upper,cap,iterations,tighten=iterations<c.reference_dual_iterations)
        ids,_,record=compact.pack(selected,hids,s.fine.reps,s.fine.labels,population,cap)
        weights=compact.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry)
        if diagnostics:record.update(compact.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps))
    else:
        selected,count=budget_states(objective,costs,cap,iterations,lambda_upper=upper)
        ids,_,_,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,cap)
        weights,_=transfer_weights(selected,zvalues,s.fine.reps,s.fine.labels,hids,prior,ids)
    record['history_cap_discarded']=discarded
    return ids,weights,record


def guard_groups(groups,layer,count,device):
    count=min(groups,count)
    # Deterministic rotating groups; within-group offset16 is fixed.
    return ((torch.arange(count,device=device)*max(1,groups//count)+layer*131)%groups).long()


def guard_decision(s,q,k,v,pooled,overlap,population,zvalues,geometry,layer,requested_cap,forced):
    c=s.config;group_ids=guard_groups(s.groups,layer,c.guard_groups,q.device)
    # Counterfactual state selection uses CURRENT Q/K/V and the incoming actual
    # observed bank, never GT, a second network, or future-layer information.
    with stage(s,'guard_reference_allocation'):
        rid,rw,_=allocate(s,q,pooled,overlap,population,zvalues,geometry,group_ids,layer,s.max_budget,c.reference_dual_iterations)
    with stage(s,'guard_candidate_allocation'):
        if forced:
            cid,cw,_=allocate(s,q,pooled,overlap,population,zvalues,geometry,group_ids,layer,requested_cap,c.dual_iterations)
        else:
            cid=s.cache_ids[group_ids].contiguous();cw=s.cache_weights[group_ids].contiguous()
    with stage(s,'guard_sample_attention'):
        positions=(group_ids*32+16).clamp_max(s.n-1)
        sq=q[:,positions].contiguous()
        actual_ref=s.patches[rid.long().clamp_min(0)].masked_fill(rid<0,-1).int().contiguous()
        actual_candidate=s.patches[cid.long().clamp_min(0)].masked_fill(cid<0,-1).int().contiguous()
        ref=shared.sample(sq,k,v,actual_ref,rw,s.specials,s.scale)
        candidate=shared.sample(sq,k,v,actual_candidate,cw,s.specials,s.scale)
        # One global layer decision; all sampled heads and queries contribute.
        # Zero/near-zero teacher output uses a fixed numerical denominator.
        per_group=(candidate-ref).square().sum((0,2)).sqrt()/ref.square().sum((0,2)).sqrt().clamp_min(1e-8)
        maximum=per_group.max()
        passed=bool(torch.isfinite(maximum)&(maximum<=c.guard_rel_l2))
    s.guard_candidate_width=cid.shape[1]
    s.guard_reference_edges=(rid>=0).sum()+len(group_ids)*len(s.specials)
    s.guard_candidate_edges=(cid>=0).sum()+len(group_ids)*len(s.specials)
    return passed,maximum,per_group,group_ids


class GuardedReuse(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config
        if self.training or len(x)!=1 or self.layer!=s.previous_layer+1:raise ValueError('Ordered batch-one inference required')
        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
        if not q.is_cuda:raise ValueError('Shared-grid guarded backend requires CUDA')
        s.scale=o.scale
        with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches];s.fine.update(features,self.layer)
        with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):
            population,_,pooled,overlap=s.prepare(k,features)
            zvalues=normalized_values(features)
        with stage(s,'geometry_prepare'),torch.autocast(x.device.type,enabled=False):
            geometry=compact.prepare_geometry(zvalues,s.fine.labels,s.fine.reps,population)
            if geometry is not None:geometry['skip_invalid_states']=True
        requested=c.retention if c.layer_retentions is None else c.layer_retentions[self.layer]
        requested_cap=math.floor(requested*s.n)-len(s.specials)
        boundary=self.layer==0 or (c.layer_retentions is not None and requested!=c.layer_retentions[self.layer-1])
        forced=(boundary or s.cache_layer is None or self.layer-s.cache_layer>=c.refresh_interval or s.cache_cap>requested_cap)
        passed,error,group_errors,group_ids=guard_decision(s,q,k,v,pooled,overlap,population,zvalues,geometry,self.layer,requested_cap,forced)
        fallback=not passed
        cap=s.max_budget if fallback else requested_cap
        iterations=c.reference_dual_iterations if fallback else c.dual_iterations
        refresh=forced or fallback
        with stage(s,'shared_bank_prepare'):
            bank=shared.prepare_bank(k,v,s.fine.reps,s.patches,s.specials,s.grid,s.regions)
        output=torch.empty_like(v);ids_all=[];weight_all=[];delta_all=[];grids=[];units=[];stats=[];capture=[]
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*32;end=min(s.n,(start+g)*32)
            with stage(s,'reuse_and_budget'),torch.autocast(x.device.type,enabled=False):
                if refresh:
                    ids,weights,record=allocate(s,q,pooled,overlap,population,zvalues,geometry,torch.arange(start,start+g,device=x.device),self.layer,cap,iterations,diagnostics=True)
                else:
                    ids=s.cache_ids[start:start+g].contiguous();weights=s.cache_weights[start:start+g].contiguous()
                    # Cached support has logical IDs and population multipliers;
                    # current Q/K/V and evidence are always recomputed.
                    counts=(ids>=0).sum(1)
                    record=dict(counts=counts)
            with stage(s,'execution_and_evidence'):
                result,delta,grid,unit=shared.attention(q[:,first:end],k,v,ids,weights,s.patches,s.fine.labels,s.fine.reps,s.specials,s.grid,pooled,bank,o.scale)
            output[:,first:end]=result
            ids_all.append(ids);weight_all.append(weights);delta_all.append(delta);grids.append(grid.half());units.append(unit.half());stats.append(record)
            if c.capture:capture.append(dict(ids=s.patches[ids.long().clamp_min(0)].masked_fill(ids<0,-1).int(),population=weights.clone(),genuine=ids>=0,delta=delta.clone(),grid=grid.clone(),unit=unit.clone()))
        with stage(s,'history_summary'):
            s.history_ids=torch.cat(ids_all);s.history_relative=torch.cat(delta_all)
            s.history_weights=None # Relative-logit mode never consumes probability evidence.
            s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(units)
            if refresh:
                s.cache_ids=s.history_ids;s.cache_weights=torch.cat(weight_all);s.cache_cap=cap;s.cache_layer=self.layer
        counts=torch.cat([r['counts'] for r in stats])
        if bool((counts>cap).any()):raise RuntimeError('Guarded support exceeds actual layer budget')
        rows=(s.n-torch.arange(s.groups,device=x.device)*32).clamp_max(32)
        diagnostics={}
        if refresh:
            common=set.intersection(*(set(r) for r in stats))
            diagnostics={key+'_mean':torch.cat([r[key] for r in stats]).float().mean() for key in common}
        record=dict(layer=self.layer,backend=c.attention_backend,requested_retention=requested,actual_retention_cap=(cap+len(s.specials))/s.n,
            patch_budget=cap,special_keys=len(s.specials),fine_clusters=s.fine.base_count,canonical_regions=s.regions,
            actual_unique_density=((counts+len(s.specials))*rows).sum()/s.n**2,
            unused_budget_mean=(cap-counts).float().mean(),support_count_max=counts.max(),
            refresh=refresh,forced_refresh=bool(forced),fallback=fallback,guard_passed=passed,guard_max_relative_l2=error,
            guard_relative_l2_by_group=list(group_errors.unbind()),guard_group_ids=list(group_ids.unbind()),guard_queries_per_group=1,
            iterations=iterations if refresh else 0,cache_source_layer=s.cache_layer,evidence_extra_qk_per_head=0,
            evidence='single-QK grid log normalizers, raw relative logits in logical support order',
            guard_reference_qk_per_head=s.guard_reference_edges,
            guard_candidate_qk_per_head=s.guard_candidate_edges,
            geometry_width=0 if geometry is None else geometry['width'],**diagnostics)
        s.records.append(record);s.guard_records.append({key:record[key] for key in ('layer','refresh','fallback','guard_max_relative_l2','requested_retention','actual_retention_cap','iterations')})
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        s.previous_layer=self.layer
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))

