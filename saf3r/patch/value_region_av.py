"""Whole Value-cluster region search; opt-in successor to fused_shared_av.

Parent and child membership is a hierarchy of the maintained Value fine labels,
not another spatial grid. Only the budget-crossing region is refined. A chosen
fine cluster is never cut, and no terminal real-Key QK is performed.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .fused_shared_av import (FusedSharedAVConfig, FullCoverageValue, patch_budgets,
    shared_scores, pack_disjoint_history, reference_attention, stage)
from .region_history_attention import prepare_qkv
from .value_evolving_attention import token_layout


@dataclass(frozen=True)
class ValueRegionAVConfig(FusedSharedAVConfig):
    region_child_width:int=8

    def __post_init__(self):
        super().__post_init__()
        if (not 1<=self.region_child_width<=16 or self.source_width%self.region_child_width
                or self.source_width//self.region_child_width>16):
            raise ValueError('Child width must divide source width; both boundary fanouts must be <=16')


class ValueRegionAVState:
    def __init__(self,config):self.config=config;self.profiling=False

    def reset(self,frames,height,width,device):
        c=self.config;self.frames,self.height,self.width=frames,height,width
        self.patches,self.specials=token_layout(frames,height*width,5,device)
        self.n=frames*(height*width+5)
        self.total,self.ka,self.kv=patch_budgets(self.n,len(self.specials),c)
        self.value=FullCoverageValue(frames,height*width,max(self.kv),device,c.source_width,c.local_iterations)
        self.parent_count=self.value.groups
        self.children_per_parent=c.source_width//c.region_child_width
        self.child_count=self.parent_count*self.children_per_parent
        self.padded_fine=self.parent_count*c.source_width
        fine=torch.arange(self.value.base_count,device=device)
        self.parent=fine//c.source_width;self.child=fine//c.region_child_width
        self.parent_fine_count=torch.bincount(self.parent,minlength=self.parent_count).float()
        self.child_fine_count=torch.bincount(self.child,minlength=self.child_count).float()
        self.capacity=self.value.table.shape[1]
        self.history=None;self.previous_layer=-1;self.records=[];self.captured={};self.profile_events=[]

    def refresh_hierarchy(self,keys,value_ids):
        """Small local member-table sort, not per-Query global Token ranking."""
        c=self.config;labels=self.value.labels;table=self.value.table
        local_labels=labels[table.clamp_min(0)].masked_fill(table<0,self.padded_fine)
        order=local_labels.argsort(dim=1,stable=True)
        self.member_bank=table.gather(1,order).int().flatten().contiguous()
        sizes=torch.bincount(labels,minlength=self.padded_fine)
        self.fine_sizes=sizes
        matrix=sizes.reshape(self.parent_count,c.source_width)
        self.parent_sizes=matrix.sum(1)
        self.child_sizes=matrix.reshape(self.child_count,c.region_child_width).sum(1)
        base=torch.arange(self.parent_count,device=keys.device)*self.capacity
        self.parent_starts=base
        self.fine_starts=(base[:,None]+matrix.cumsum(1)-matrix).flatten()
        child_matrix=self.child_sizes.reshape(self.parent_count,self.children_per_parent)
        self.child_starts=(base[:,None]+child_matrix.cumsum(1)-child_matrix).flatten()
        self.value_fine_counts=torch.bincount(labels[value_ids],minlength=self.padded_fine)
        # One real representative per Value fine cluster, shared across heads.
        fine=keys[:,self.patches[self.value.reps]].contiguous()
        def pool(mapping,count,counts):
            out=torch.zeros((keys.shape[0],count,keys.shape[-1]),device=keys.device,dtype=torch.float32)
            out.index_add_(1,mapping,fine.float())
            return (out/counts.clamp_min(1)[None,:,None]).to(keys.dtype)
        self.pooled_parent=pool(self.parent,self.parent_count,self.parent_fine_count)
        self.pooled_child=pool(self.child,self.child_count,self.child_fine_count)
        self.fine_keys=F.pad(fine,(0,0,0,self.padded_fine-self.value.base_count))
        self.parent_log_population=self.parent_sizes.clamp_min(1).float().log()
        # Last parent/child can contain padded, nonexistent descendants.
        # They must have ZERO probability, not population=1 in head softmax.
        self.child_log_population=self.child_sizes.float().log()
        self.fine_log_population=sizes.float().log()


def boundary_scores(q,keys,log_population,candidates,active,start_group,scale):
    """Score ONLY one boundary's descendants per Query group, never all children."""
    if q.is_cuda:
        from saf3r.triton.value_region_selection import boundary_probabilities
        return boundary_probabilities(q,keys,log_population,candidates,active,start_group,scale)
    probes=(torch.arange(len(candidates),device=q.device)[:,None]+start_group)*32+torch.tensor([4,12,20,28],device=q.device)
    probes=probes.clamp_max(q.shape[1]-1)
    logits=torch.einsum('hgtd,hgcd->hgtc',q[:,probes].float(),keys[:,candidates].float())*scale
    logits+=log_population[candidates][None,:,None]
    return logits.softmax(-1).mean((0,2))*active[:,None]


def ranked_prefix(scores,counts,target,include_boundary=False):
    """Whole-region prefix. Return at most ONE crossing region to refine."""
    order=scores.masked_fill(counts==0,-torch.inf).argsort(dim=1,descending=True,stable=True)
    ordered=counts.gather(1,order);ends=ordered.cumsum(1);before=ends-ordered
    full=(ordered>0)&(before<target[:,None])
    if not include_boundary:full&=ends<=target[:,None]
    crossing=(ordered>0)&(before<target[:,None])&(ends>target[:,None])
    boundary=order.gather(1,crossing.long().argmax(1)[:,None]).squeeze(1)
    remainder=(target-(ordered*full).sum(1)).clamp_min(0)
    return order,full,boundary,crossing.any(1),remainder


def expand_regions(clean,starts,lengths,state,is_value,ka):
    """Unique history first, then all members of complete selected regions."""
    width=ka+state.capacity-1
    if starts.is_cuda:
        from saf3r.triton.value_region_selection import expand_member_regions
        return expand_member_regions(clean,starts,lengths,state.member_bank,is_value,width,
            ka+len(state.current_value_ids)+state.capacity-1)
    result=torch.full((len(starts),width),-1,dtype=torch.int32,device=starts.device)
    counts=torch.zeros(len(starts),dtype=torch.long,device=starts.device)
    for row in range(len(starts)):
        ids=clean[row][clean[row]>=0].long().tolist();seen=set(ids)
        for start,length in zip(starts[row].tolist(),lengths[row].tolist()):
            for idx in state.member_bank[start:start+length].tolist():
                if idx>=0 and not is_value[idx] and idx not in seen:ids.append(idx);seen.add(idx)
        if len(ids)>width:raise RuntimeError('Whole-region support overflow')
        result[row,:len(ids)]=torch.tensor(ids,dtype=torch.int32)
        counts[row]=len(ids)
    return result,counts


def select_value_regions(q,state,is_value,history,value_ids,ka,start_group,groups,scale):
    c=state.config;device=q.device;state.current_value_ids=value_ids
    clean,kept=pack_disjoint_history(history,is_value)
    if clean is None:
        clean=torch.empty((groups,0),device=device,dtype=torch.int32)
        kept=torch.zeros(groups,device=device,dtype=torch.long)
    missing=ka-kept
    # A representative does NOT cover every other member of its cluster.
    # Exclude individual selected Keys; skip a region only when fully covered.
    available=(state.fine_sizes-state.value_fine_counts)[None].expand(groups,-1).clone()
    if clean.shape[1]:
        valid=clean>=0;label=state.value.labels[clean.long().clamp_min(0)]
        available.scatter_add_(1,label,-valid.long())
    parent_available=available.reshape(groups,state.parent_count,c.source_width).sum(2)
    with stage(state,'region_parent'):
        score=shared_scores(q,state.pooled_parent,scale,state.parent_log_population,start_group,groups)
        po,pfull,pboundary,pactive,remaining=ranked_prefix(score,parent_available,missing)
    # Fully chosen parents are final supports. Refine only the crossing parent.
    child_ids=pboundary[:,None]*state.children_per_parent+torch.arange(state.children_per_parent,device=device)
    child_available=available.reshape(groups,state.child_count,c.region_child_width).sum(2).gather(1,child_ids)
    child_available*=pactive[:,None]
    with stage(state,'region_child_boundary'):
        score=boundary_scores(q,state.pooled_child,state.child_log_population,child_ids,pactive,start_group,scale)
        co,cfull,cboundary,cactive,remaining=ranked_prefix(score,child_available,remaining)
    ordered_child=child_ids.gather(1,co)
    boundary_child=child_ids.gather(1,cboundary[:,None]).squeeze(1)
    fine_ids=boundary_child[:,None]*c.region_child_width+torch.arange(c.region_child_width,device=device)
    fine_available=available.gather(1,fine_ids)*cactive[:,None]
    with stage(state,'region_fine_boundary'):
        score=boundary_scores(q,state.fine_keys,state.fine_log_population,fine_ids,cactive,start_group,scale)
        fo,ffull,_,_,_=ranked_prefix(score,fine_available,remaining,include_boundary=True)
    ordered_fine=fine_ids.gather(1,fo)
    starts=torch.cat((state.parent_starts[po],state.child_starts[ordered_child],state.fine_starts[ordered_fine]),1)
    lengths=torch.cat((state.parent_sizes[po]*pfull,state.child_sizes[ordered_child]*cfull,
        state.fine_sizes[ordered_fine]*ffull),1)
    with stage(state,'region_expand'):
        attention,counts=expand_regions(clean,starts,lengths,state,is_value,ka)
    return attention,counts,dict(incoming_history=kept.sum(),parent_regions=pfull.sum(),
        child_regions=cfull.sum(),fine_regions=ffull.sum(),child_refined_groups=pactive.sum(),
        fine_refined_groups=cactive.sum(),terminal_member_qk_per_head=0), (starts,lengths)


def ragged_history(e,lse,common,attention,counts,value_count,history_count):
    """Same-pass tile-mass history, excluding variable-length padding and specials."""
    groups=len(attention);tile=64
    mass=(e-lse[:,:,None]).exp().sum(0)
    mass=F.pad(mass,(0,0,0,groups*32-len(mass))).reshape(groups,32,-1).sum(1)
    ps=math.ceil(len(common)/tile);ats=math.ceil(attention.shape[1]/tile)
    common_counts=(value_count-torch.arange(ps,device=attention.device)*tile).clamp(0,tile)
    att_counts=(counts[:,None]-torch.arange(ats,device=attention.device)*tile).clamp(0,tile)
    sizes=torch.cat((common_counts[None].expand(groups,-1),att_counts),1)
    order=mass.masked_fill(sizes==0,-1).argsort(dim=1,descending=True,stable=True)
    ends=sizes.gather(1,order).cumsum(1)
    ranks=torch.arange(history_count,device=attention.device)[None].expand(groups,-1).contiguous()
    which=torch.searchsorted(ends.contiguous(),ranks,right=True)
    slots=order.gather(1,which);starts=F.pad(ends,(1,0))[:,:-1].gather(1,which)
    within=ranks-starts
    common_col=(slots*tile+within).clamp_max(len(common)-1)
    if attention.shape[1]:
        att_col=((slots-ps)*tile+within).clamp(0,attention.shape[1]-1)
        return torch.where(slots<ps,common[common_col],attention.gather(1,att_col)).int()
    return common[common_col].int()


class ValueRegionAVAttention(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config
        if self.training or len(x)!=1:raise ValueError('Batch-one inference only')
        if self.layer!=s.previous_layer+1:raise ValueError('Invalid layer state order')
        q,k,v=prepare_qkv(o,x,pos);q,k=q.to(v.dtype).contiguous(),k.to(v.dtype).contiguous()
        with stage(s,'value'),torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches]
            s.value.update(features,self.layer)
            value_ids,merged=s.value.representatives(features,s.kv[self.layer])
        common=torch.cat((s.patches[value_ids],s.specials));kv=len(value_ids);ka=s.ka[self.layer]
        with stage(s,'pack'):pk=k[:,common].contiguous();pv=v[:,common].contiguous()
        is_value=torch.zeros(len(s.patches),device=x.device,dtype=torch.bool);is_value[value_ids]=True
        if ka:
            with stage(s,'value_region_hierarchy'):s.refresh_hierarchy(k,value_ids)
        groups=math.ceil(s.n/32);output=torch.empty_like(v)
        next_history=min(s.total[self.layer],math.floor(c.history_fraction*s.ka[self.layer+1])) if self.layer<23 else 0
        histories=[];captured=[];region_capture=[];all_counts=[];stats=[]
        for g in range(0,groups,c.window_groups):
            length=min(c.window_groups,groups-g);start=g*32;end=min(s.n,(g+length)*32)
            if ka:
                old=None if s.history is None else s.history[g:g+length]
                with stage(s,'selection_including_expansion'):
                    attention,counts,record,regions=select_value_regions(q,s,is_value,old,value_ids,ka,g,length,o.scale)
                stats.append(record)
                if c.capture:region_capture.append(regions)
            else:
                attention=torch.empty((length,0),device=x.device,dtype=torch.int32)
                counts=torch.zeros(length,device=x.device,dtype=torch.long)
            all_counts.append(counts)
            actual=s.patches[attention.long().clamp_min(0)].masked_fill(attention<0,-1).int()
            if q.is_cuda:
                from saf3r.triton.fused_shared_av import shared_av
                with stage(s,'execution'):
                    result,e,lse=shared_av(q[:,start:end],k,v,pk,pv,actual,kv,scale=o.scale,evidence=bool(next_history))
            else:result,e,lse=reference_attention(q[:,start:end],k,v,common,actual,kv,o.scale)
            output[:,start:end]=result
            if next_history:
                with stage(s,'history'):indices=ragged_history(e,lse,common,actual,counts,kv,next_history)
                histories.append((indices.long()-(indices.long()//(s.height*s.width+5)+1)*5).int())
            if c.capture:captured.append(actual.clone())
        s.history=torch.cat(histories) if next_history else None;s.previous_layer=self.layer
        counts=torch.cat(all_counts);rows=(s.n-torch.arange(groups,device=x.device)*32).clamp_max(32)
        unique=(counts+len(common))*rows
        record=dict(layer=self.layer,attention_edges_target=ka,value_edges=kv,special_keys=len(s.specials),
            total_edges_target=ka+len(common),attention_capacity=attention.shape[1],
            actual_attention_min=counts.min(),actual_attention_max=counts.max(),
            overshoot_max=(counts-ka).max(),actual_density=unique.sum()/s.n**2,
            final_unique_qk_per_head=unique.sum(),
            final_padded_qk_per_head=groups*32*(math.ceil(len(common)/64)*64+math.ceil(attention.shape[1]/64)*64),
            attention_ratio=c.attention_ratios[self.layer],history_out=next_history,
            base_clusters=s.value.base_count,query_groups=groups,source='whole_Value_regions_boundary_refinement',
            terminal_member_qk_per_head=0)
        if ka:
            record['max_fine_population']=s.fine_sizes.max()
            for name in stats[0]:record[name]=sum(z[name] for z in stats)
            record['routing_qk_per_head']=4*(groups*s.parent_count+
                record['child_refined_groups']*s.children_per_parent+record['fine_refined_groups']*c.region_child_width)
        else:record['routing_qk_per_head']=0
        s.records.append(record)
        if c.capture:
            s.captured[self.layer]=dict(common=common.clone(),attention=torch.cat(captured),
                base_labels=s.value.labels.clone(),value_labels=None if merged is None else merged.clone(),
                value_representatives=value_ids.clone(),regions=region_capture,
                member_bank=None if not ka else s.member_bank.clone())
        if ka:
            for name in ('pooled_parent','pooled_child','fine_keys','member_bank'):
                setattr(s,name,None)
        del q,k,v,pk,pv,features
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install_value_region_av(model,config=None,**kwargs):
    if hasattr(model,'value_region_av_state'):raise ValueError('Already installed')
    config=ValueRegionAVConfig(**kwargs) if config is None else config
    state=ValueRegionAVState(config)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=ValueRegionAVAttention(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one image sequence required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_hook=model.aggregator.register_forward_pre_hook(reset)
    model.value_region_av_state=state
    return state
