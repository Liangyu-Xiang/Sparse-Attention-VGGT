"""Shared Value support + confidence-Mask/residual hierarchical Attention.

Correctness-oriented GPU implementation. No Dense oracle, random exploration,
head-Q/K averaging, persistent Attention weights, or repeated final QK pass.
Actual explored leaf coverage is recorded; no fixed search-rate speed claim.
"""
import math
import torch
from torch import nn

from .layer_shared_av import LayerSharedAVState
from .hierarchical_value_attention import sizes
from .value_clustering_gpu import stable_cluster_mean
from .value_evolving_attention import token_layout


def allocate_variable(weights,totals,capacities):
    """Per-row capacity-capped Hamilton allocation, including zero budgets."""
    if weights.shape!=capacities.shape or totals.shape!=(len(weights),):
        raise ValueError('invalid allocation shapes')
    if bool((totals<0).any()) or bool((totals>capacities.sum(1)).any()):
        raise ValueError('budget exceeds available capacity')
    result=torch.zeros_like(capacities)
    remaining=totals.clone()
    while bool((remaining>0).any()):
        available=capacities-result
        active=available>0
        w=weights.float().clamp_min(0)*active
        denom=w.sum(1,keepdim=True)
        fallback=active.float()/active.sum(1,keepdim=True).clamp_min(1)
        p=torch.where(denom>0,w/denom.clamp_min(1e-30),fallback)
        desired=p*remaining[:,None]
        base=torch.minimum(desired.floor().long(),available)
        result+=base;remaining-=base.sum(1)
        truncated=(desired.floor().long()>base).any(1)
        available=capacities-result
        fractions=(desired-desired.floor()).masked_fill(available==0,-torch.inf)
        order=fractions.argsort(dim=1,descending=True,stable=True)
        rank=torch.empty_like(order).scatter_(1,order,
            torch.arange(order.shape[1],device=order.device).expand_as(order))
        extra=(rank<remaining[:,None])&(available>0)&~truncated[:,None]
        result+=extra;remaining-=extra.sum(1)
    return result


def residual_allocation(probabilities,total,history,capacities):
    """Total desired quota minus occupancy, then normalize the remaining deficit."""
    target=allocate_variable(probabilities,total,capacities)
    deficit=(target-history).clamp_min(0)
    remaining=total-history.sum(1)
    if bool((remaining<0).any()) or bool((history>capacities).any()):
        raise ValueError('history exceeds final budget/capacity')
    residual=allocate_variable(deficit.float(),remaining,capacities-history)
    return history+residual,residual


def pack_mask(mask):
    """Pack row-wise True positions in token order, padded with -1."""
    counts=mask.sum(1)
    width=int(counts.max()) if len(counts) else 0
    output=torch.full((len(mask),width),-1,device=mask.device,dtype=torch.long)
    if width:
        row,col=mask.nonzero(as_tuple=True)
        starts=counts.cumsum(0)-counts
        slot=torch.arange(len(row),device=mask.device)-starts[row]
        output[row,slot]=col
    return output


def coarse_probabilities(q,pooled):
    # Pooling never mixes heads. Query positions also remain independent.
    return ((q@pooled.transpose(-1,-2))*(q.shape[-1]**-.5)).softmax(-1).mean(0)


def history_from_weights(mean_prob,patch_indices,max_fraction,gamma):
    """Concentration over actually executed full support, incl mandatory specials.

    Return only indices (a sparse binary Mask), not weights or ranking metadata.
    The caller aggregates entropy across query chunks before choosing width.
    """
    entropy=-(mean_prob*mean_prob.clamp_min(1e-30).log()).sum(1)
    concentration=(1-entropy/math.log(mean_prob.shape[1])).clamp(0,1) if mean_prob.shape[1]>1 else torch.ones_like(entropy)
    width=math.ceil(max_fraction*patch_indices.shape[1])
    ranked=mean_prob[:,:patch_indices.shape[1]].argsort(dim=1,descending=True,stable=True)[:,:width]
    return concentration,patch_indices.gather(1,ranked)


def assemble_history(chunks,keep):
    """Trim ranked chunks before joining; sort each row without a full-width copy."""
    if not keep:return None
    if any(chunk.shape[1]<keep for chunk in chunks):
        raise ValueError('history chunk has insufficient ranked candidates')
    result=torch.empty((sum(len(chunk) for chunk in chunks),keep),
        device=chunks[0].device,dtype=torch.int32)
    offset=0
    for chunk in chunks:
        result[offset:offset+len(chunk)]=chunk[:,:keep].to(torch.int32).sort(1).values
        offset+=len(chunk)
    return result


def qk_logits(q,k,indices,scale):
    if q.is_cuda:
        from saf3r.triton.residual_attention import indexed_qk
        return indexed_qk(q,k,indices.to(torch.int32).contiguous(),scale)
    selected=k[:,indices.clamp_min(0)]
    z=(q[:,:,None].float()*selected.float()).sum(-1)*scale
    return z.masked_fill(indices[None]<0,-torch.inf)


def logits_av(logits,v,indices):
    if v.is_cuda:
        from saf3r.triton.residual_attention import cached_logits_av
        return cached_logits_av(logits.contiguous(),v,indices.to(torch.int32).contiguous())
    values=v[:,indices.clamp_min(0)].float()
    return (logits.softmax(-1)[...,None]*values).sum(-2).to(v.dtype)


def select_per_fine(scores,fine_ids,eligible,quota):
    """Select each fine group's exact quota by current mean-head probabilities."""
    rows,cols=scores.shape;nf=quota.shape[1]
    score_order=scores.masked_fill(~eligible,-torch.inf).argsort(dim=1,descending=True,stable=True)
    groups=fine_ids.gather(1,score_order).masked_fill(~eligible.gather(1,score_order),nf)
    by_group=groups.argsort(dim=1,stable=True)
    ordered=score_order.gather(1,by_group)
    group_ids=groups.gather(1,by_group)
    position=torch.arange(cols,device=scores.device).expand(rows,-1)
    changed=torch.cat((torch.ones((rows,1),device=scores.device,dtype=torch.bool),group_ids[:,1:]!=group_ids[:,:-1]),1)
    start=torch.where(changed,position,0).cummax(1).values
    keep=(group_ids<nf)&((position-start)<quota.gather(1,group_ids.clamp_max(nf-1)))
    return torch.zeros_like(eligible).scatter_(1,ordered,keep)


class ResidualHistoryAVState(LayerSharedAVState):
    def __init__(self,retention=.1,fine_ratio=.15,coarse_size=128,cluster_iterations=2,
                 query_chunk=32,coarse_fraction=.5,history_max_fraction=.5,
                 confidence_gamma=1.,attention_ratios=None,fine_pair_batch=128,
                 capture_masks=False):
        if not 0<coarse_fraction<=1 or not 0<=history_max_fraction<=1 or confidence_gamma<=0:
            raise ValueError('invalid coarse/history parameters')
        if query_chunk<1 or fine_pair_batch<1:
            raise ValueError('invalid chunk/batch size')
        ratios=[0.]*10+[1.]*7+[0.]*7 if attention_ratios is None else list(attention_ratios)
        if len(ratios)!=24 or any(not math.isfinite(a) or not 0<=a<=1 for a in ratios):
            raise ValueError('attention_ratios must contain24 values in[0,1]')
        self.attention_ratios=ratios
        self.coarse_fraction,self.history_max_fraction=coarse_fraction,history_max_fraction
        self.confidence_gamma,self.fine_pair_batch=confidence_gamma,fine_pair_batch
        self.capture_masks=capture_masks
        # Base state supplies ONLY unchanged Value hierarchy maintenance.
        super().__init__(retention=retention,fine_ratio=fine_ratio,coarse_size=coarse_size,
            cluster_iterations=cluster_iterations,query_chunk=128)
        self.query_chunk=query_chunk

    def reset(self,frames,height,width):
        super().reset(frames,height,width)
        self.captured_masks=[]
        self.captured_histories=[]
        self._previous_history_width=0


def hierarchy_tables(labels,reps,mapping,value_ids):
    nf=len(reps);nc=int(mapping.max())+1
    value_mask=torch.zeros(len(labels),device=labels.device,dtype=torch.bool)
    value_mask[value_ids]=True
    capacity=sizes(labels[~value_mask],nf)
    coarse_capacity=torch.zeros(nc,device=labels.device,dtype=torch.long).index_add_(0,mapping,capacity)
    count=sizes(mapping,nc);order=mapping.argsort(stable=True)
    offsets=count.cumsum(0)-count
    children=torch.full((nc,int(count.max())),-1,device=labels.device,dtype=torch.long)
    children[mapping[order],torch.arange(nf,device=labels.device)-offsets[mapping[order]]]=order
    return value_mask,capacity,coarse_capacity,children


def residual_search(q,kpatch,hierarchy,value_ids,history,budget,state,prepared=None):
    labels,reps,mapping=hierarchy
    rows=len(q[0]);nf=len(reps);nc=int(mapping.max())+1
    if prepared is None:
        value_mask,fcap,ccap,children=hierarchy_tables(labels,reps,mapping,value_ids)
        pooled=torch.stack([stable_cluster_mean(kh[reps],mapping,nc) for kh in kpatch])
    else:
        value_mask,fcap,ccap,children,pooled=prepared
    p=coarse_probabilities(q,pooled)
    order=p.argsort(dim=1,descending=True,stable=True)
    rank=torch.empty_like(order).scatter_(1,order,torch.arange(nc,device=q.device).expand(rows,-1))
    # Only widen the top-region gate if it cannot supply the requested UNIQUE
    # Attention budget after protected Value edges have been excluded.
    cumulative=ccap[order].cumsum(1)
    required=(cumulative<budget).sum(1)+1
    base=max(1,math.ceil(state.coarse_fraction*nc))
    width=torch.maximum(required,torch.full_like(required,base))
    gate=rank<width[:,None]
    kept=torch.zeros((rows,len(labels)),device=q.device,dtype=torch.bool)
    if history is not None and history.shape[1]:
        ids=history.long();valid=ids>=0;safe=ids.clamp_min(0)
        valid&=gate.gather(1,mapping[labels[safe]])&~value_mask[safe]
        priority=p.gather(1,mapping[labels[safe]]).masked_fill(~valid,-torch.inf)
        pick=priority.argsort(dim=1,descending=True,stable=True)[:,:min(budget,ids.shape[1])]
        chosen=safe.gather(1,pick);ok=valid.gather(1,pick)
        r,c=ok.nonzero(as_tuple=True)
        kept[r,chosen[r,c]]=True
    occupancy_f=torch.zeros((rows,nf),device=q.device,dtype=torch.long)
    occupancy_f.scatter_add_(1,labels.expand(rows,-1),kept.long())
    occupancy_c=torch.zeros((rows,nc),device=q.device,dtype=torch.long)
    occupancy_c.scatter_add_(1,mapping.expand(rows,-1),occupancy_f)
    total=torch.full((rows,),budget,device=q.device,dtype=torch.long)
    coarse_total,coarse_residual=residual_allocation(p*gate,total,occupancy_c,ccap.expand(rows,-1)*gate)
    fine_residual=torch.zeros_like(occupancy_f)
    pairs=(coarse_residual>0).nonzero()
    for begin in range(0,len(pairs),state.fine_pair_batch):
        pair=pairs[begin:begin+state.fine_pair_batch]
        r,c=pair.unbind(1);child=children[c];safe=child.clamp_min(0)
        valid=child>=0
        capacity=fcap[safe]*valid
        old=occupancy_f[r[:,None],safe]*valid
        representatives=kpatch[:,reps[safe]]
        z=(q[:,r,None]*representatives).sum(-1)*(q.shape[-1]**-.5)
        z=z.masked_fill(~valid[None],-torch.inf)
        prob=z.softmax(-1).mean(0)
        # Pass the TOTAL parent quota, not its residual: subtract history once
        # at this level, avoiding the reference pseudocode's double deduction.
        _,missing=residual_allocation(prob,coarse_total[r,c],old,capacity)
        a,b=valid.nonzero(as_tuple=True)
        fine_residual[r[a],child[a,b]]=missing[a,b]
    leaf=fine_residual[:,labels]>0
    protected=kept|value_mask[None]
    candidates=leaf|protected
    stats=dict(history_reused=int(kept.sum()),coarse_qk_edges=rows*nc,
        fine_representative_qk_edges=int((children[pairs[:,1]]>=0).sum()) if len(pairs) else 0,
        expanded_queries=int((width>base).sum()),active_parents=len(pairs))
    return candidates,protected,fine_residual,stats


class ResidualHistoryAVAttention(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        if self.training or x.shape[0]!=1:
            raise ValueError('batch-one inference required')
        o,s=self.original,self.state
        b,n,c=x.shape
        q,k,v=o.qkv(x).reshape(b,n,3,o.num_heads,o.head_dim).permute(2,0,3,1,4).unbind(0)
        patches,specials=token_layout(s.frames,s.height*s.width,5,x.device)
        with torch.autocast(x.device.type,enabled=False):
            labels,reps,mapping,count=s.update(v[0,:,patches].float().mean(0),self.layer)
        q,k=o.q_norm(q),o.k_norm(k)
        if o.rope is not None:q,k=o.rope(q,pos),o.rope(k,pos)
        q,k,v=q[0].contiguous(),k[0].contiguous(),v[0].contiguous()
        total=math.floor(s.retention*n);patch_budget=total-len(specials)
        if not 0<patch_budget<=len(reps):raise ValueError('incompatible sparse budget')
        ka=round(s.attention_ratios[self.layer]*patch_budget);kv=patch_budget-ka
        priority=count.argsort(descending=True,stable=True)[:kv]
        if self.layer==0 and kv:priority=torch.arange(kv,device=x.device)*len(reps)//kv
        value_ids=reps[priority]
        fine_q=q.to(v.dtype)
        fine_k=k.to(v.dtype).contiguous()
        prepared=None;kpatch=None
        if ka:
            with torch.autocast(x.device.type,enabled=False):
                kpatch=k[:,patches].float().contiguous()
                tables=hierarchy_tables(labels,reps,mapping,value_ids)
                pooled=torch.stack([stable_cluster_mean(kh[reps],mapping,int(mapping.max())+1) for kh in kpatch])
                prepared=(*tables,pooled)
        output=torch.empty_like(v)
        pending=[];concentrations=[];captured=[]
        counters=dict(history_reused=0,coarse_qk_edges=0,fine_representative_qk_edges=0,
            expanded_queries=0,active_parents=0,leaf_candidate_edges=0,final_patch_edges=0)
        for start in range(0,n,s.query_chunk):
            end=min(n,start+s.query_chunk);rows=end-start
            with torch.autocast(x.device.type,enabled=False):
                if ka:
                    history=None if s.history is None else s.history[start:end]
                    candidate,protected,quota,stats=residual_search(q[:,start:end].float(),kpatch,
                        (labels,reps,mapping),value_ids,history,ka,s,prepared)
                    for key,value in stats.items():counters[key]+=value
                else:
                    candidate=torch.zeros((rows,len(patches)),device=x.device,dtype=torch.bool)
                    candidate[:,value_ids]=True;protected=candidate
                packed=pack_mask(candidate)
                global_keys=patches[packed.clamp_min(0)].masked_fill(packed<0,-1)
                all_keys=torch.cat((global_keys,specials.expand(rows,-1)),1)
                # Match autocast SDPA operands; accumulate the actual indexed QK
                # once, then reuse its selected logits for AV and confidence.
                logits=qk_logits(fine_q[:,start:end],fine_k,all_keys,o.scale)
                mean=logits.softmax(-1).mean(0)
                if ka:
                    valid=packed>=0;safe=packed.clamp_min(0)
                    fixed=protected.gather(1,safe)&valid
                    selected=select_per_fine(mean[:,:packed.shape[1]],labels[safe],valid&~fixed,quota)|fixed
                    final_mask=torch.cat((selected,torch.ones((rows,len(specials)),device=x.device,dtype=torch.bool)),1)
                    if not bool((final_mask.sum(1)==total).all()):raise RuntimeError('final unique budget mismatch')
                    slots=pack_mask(final_mask)
                    final_keys=all_keys.gather(1,slots)
                    selected_logits=logits.gather(2,slots[None].expand(o.num_heads,-1,-1))
                    final_patch=packed.gather(1,slots[:,:patch_budget])
                else:
                    final_keys,selected_logits,final_patch=all_keys,logits,packed
                output[:,start:end]=logits_av(selected_logits,v,final_keys)
                final_mean=selected_logits.softmax(-1).mean(0)
                confidence,seeds=history_from_weights(final_mean,final_patch,s.history_max_fraction,s.confidence_gamma)
                concentrations.append(confidence);pending.append(seeds.to(torch.int32))
                counters['leaf_candidate_edges']+=int(candidate.sum())
                counters['final_patch_edges']+=rows*patch_budget
                if s.capture_masks:captured.append(final_keys.detach().cpu())
        concentration=float(torch.cat(concentrations).mean())
        keep=math.floor(s.history_max_fraction*(concentration**s.confidence_gamma)*patch_budget)
        # Sorting numeric IDs encodes only a binary support, not previous scores.
        s.history=assemble_history(pending,keep)
        if s.capture_masks:
            s.captured_masks.append(torch.cat(captured))
            s.captured_histories.append(s.history.cpu() if s.history is not None else torch.empty((n,0),dtype=torch.int32))
        s.records.append(dict(layer=self.layer,attention_keys=ka,value_keys=kv,special_keys=len(specials),
            total_keys=total,actual_density=total/n,fine_clusters=len(reps),coarse_groups=int(mapping.max())+1,
            concentration=concentration,history_keys=keep,confidence_mapping='max_fraction*C**gamma',
            value_source_layer=self.layer-1,execution_mode='value' if not ka else 'attention' if not kv else 'mixed',
            candidate_patch_density=counters['leaf_candidate_edges']/(n*len(patches)),
            history_input_keys=0 if self.layer==0 else getattr(s,'_previous_history_width',0),**counters))
        s._previous_history_width=keep
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(b,n,c)))


def install_residual_history_av(model,**kwargs):
    state=ResidualHistoryAVState(**kwargs)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=ResidualHistoryAVAttention(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0];state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14)
    state.reset_hook_handle=model.aggregator.register_forward_pre_hook(reset)
    model.residual_history_av_state=state
    return state
