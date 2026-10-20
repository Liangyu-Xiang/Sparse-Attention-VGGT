"""Independent causal budget -> Attention -> residual-Value sparse attention.

No legacy A/V controller is installed or called. Only common QKV, local Value
clustering, coarse scoring and the verified final attention kernel are reused.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from .fused_shared_av import FullCoverageValue,shared_scores
from .region_history_attention import prepare_qkv
from .value_evolving_attention import token_layout


@dataclass(frozen=True)
class AttentionFirstConfig:
    retentions:tuple=(.2,)*24
    history_fraction:float=.8
    history_weight:float=.5
    value_weight:float=1.
    source_width:int=32
    local_iterations:int=2
    gate_groups:int=32
    window_groups:int=8
    capture:bool=False

    def __post_init__(self):
        if len(self.retentions)!=24 or any(not math.isfinite(r) or not 0<r<=1 for r in self.retentions):
            raise ValueError('Provide24 finite retention fractions in (0,1]')
        if not math.isfinite(self.history_fraction) or not 0<=self.history_fraction<=1:
            raise ValueError('Invalid history fraction')
        if any(not math.isfinite(w) or w<0 for w in (self.history_weight,self.value_weight)):
            raise ValueError('Signal weights must be finite and nonnegative')
        if not 1<=self.source_width<=32 or self.local_iterations<1 or self.gate_groups<1 or self.window_groups<1:
            raise ValueError('Invalid clustering/window settings')


def allocate_budget(patch_budget,concentration,history_lift,value_error,previous,config):
    """Heuristic causal allocation, no layer table or future-layer signal."""
    if patch_budget<1:raise ValueError('Patch budget must contain a Value representative')
    signals=(concentration,history_lift,value_error)+(() if previous is None else tuple(previous))
    if any(not math.isfinite(float(signal)) for signal in signals):
        raise ValueError('Nonfinite budget evidence')
    c=max(0.,min(1.,float(concentration)))
    h=max(0.,min(1.,float(history_lift)))
    d=max(0.,min(1.,float(value_error)))
    old_c,old_d=(c,d) if previous is None else previous
    a_need=c+config.history_weight*h*old_c
    v_need=(1-c)+config.value_weight*(d+old_d)*.5
    alpha=a_need/max(a_need+v_need,1e-12)
    # At least one residual Value group is always reserved. History never
    # forces this allocation to expand when attention becomes flat again.
    ka=max(0,min(patch_budget-1,round(alpha*patch_budget)))
    return ka,patch_budget-ka,dict(requested_alpha=alpha,effective_alpha=ka/patch_budget,
        concentration=c,history_lift=h,value_error=d,attention_demand=a_need,value_demand=v_need)


def choose_attention(scores,table,parent_sizes,parent_of_patch,history,ka,npatch):
    """History first; fill exact remaining budget from current ranked regions.

    Boundary Parent uses uniform member positions, no terminal real-key QK.
    Returns patch IDs and a bounded group x patch workspace, not an NxN mask.
    """
    groups=len(scores);device=scores.device
    history=history if history is not None else torch.empty((groups,0),device=device,dtype=torch.int32)
    used=history.shape[1]
    if used>ka:raise ValueError('History must be capped AFTER current budget allocation')
    seen=torch.zeros((groups,npatch),device=device,dtype=torch.bool)
    if used:seen.scatter_(1,history.long(),True)
    hist_counts=torch.zeros((groups,len(table)),device=device,dtype=torch.long)
    if used:hist_counts.scatter_add_(1,parent_of_patch[history.long()],torch.ones_like(history,dtype=torch.long))
    available=parent_sizes[None]-hist_counts
    # Probability per real Key: mass alone unfairly rewards large regions.
    relevance=scores/parent_sizes.clamp_min(1)[None]
    order=relevance.masked_fill(available==0,-torch.inf).argsort(dim=1,descending=True,stable=True)
    counts=available.gather(1,order)
    before=counts.cumsum(1)-counts
    result=torch.full((groups,ka),-1,device=device,dtype=torch.int32)
    if used:result[:,:used]=history
    if ka>used:
        if device.type=='cuda':
            from saf3r.triton.attention_first_selection import fill_attention
            fill_attention(result,seen,table,order,counts,before,used)
        else:
            for group in range(groups):
                cursor=used
                for parent in order[group].tolist():
                    if cursor==ka:break
                    members=table[parent];members=members[members>=0]
                    members=members[~seen[group,members]]
                    take=min(len(members),ka-cursor)
                    if take:
                        positions=(torch.arange(take,device=device)*len(members)+take-1)//take
                        result[group,cursor:cursor+take]=members[positions].int();cursor+=take
                if cursor!=ka:raise RuntimeError('Insufficient unique Attention candidates')
    if ka:seen.scatter_(1,result.long(),True)
    return result,seen


def residual_value(features,labels,fine_reps,fine_sums,fine_sizes,attention,kv):
    """Cover untouched Value fine clusters AFTER Attention has selected Keys.

    One Attention member marks that fine Value structure covered (an
    approximation, NOT exact equality of its members' V). Untouched fine groups
    are merged in fine-ID order into exactly KV nonempty groups. Representatives
    are real base-fine representatives nearest each merged residual mean.
    """
    groups=len(attention);nf=len(fine_reps);device=features.device
    hit=torch.zeros((groups,nf),device=device,dtype=torch.long)
    if attention.shape[1]:
        hit.scatter_add_(1,labels[attention.long()],torch.ones_like(attention,dtype=torch.long))
    uncovered=hit==0;ordinal=uncovered.long().cumsum(1);remaining=ordinal[:,-1]
    # F>=total patch budget and unique A count=KA imply F-covered>=KV.
    if device.type=='cpu' and (remaining<kv).any():raise ValueError('Not enough untouched fine clusters')
    bins=((ordinal-1).clamp_min(0)*kv//remaining[:,None].clamp_min(1)).clamp_max(kv-1)
    boundaries=(torch.arange(kv+1,device=device)[None]*remaining[:,None]+kv-1)//kv
    starts=torch.searchsorted(ordinal.contiguous(),boundaries.contiguous(),right=True)
    sums=F.pad((fine_sums[None]*uncovered[:,:,None]).cumsum(1),(0,0,1,0))
    populations=F.pad((fine_sizes[None]*uncovered).cumsum(1),(1,0))
    at=sums.gather(1,starts[:,:,None].expand(-1,-1,features.shape[1]))
    count=populations.gather(1,starts)
    population=count[:,1:]-count[:,:-1]
    centers=(at[:,1:]-at[:,:-1])/population.clamp_min(1)[:,:,None]
    del sums,at
    candidate=features[fine_reps].contiguous()
    if device.type=='cuda':
        from saf3r.triton.attention_first_selection import representative_costs
        costs=representative_costs(candidate,centers.contiguous(),bins,uncovered)
    else:
        costs=(candidate[None]-centers.gather(1,bins[:,:,None].expand(-1,-1,candidate.shape[1]))).square().sum(-1)
        costs=costs.masked_fill(~uncovered,torch.inf)
    minimum=torch.full((groups,kv),torch.inf,device=device)
    minimum.scatter_reduce_(1,bins,costs,reduce='amin',include_self=True)
    ids=torch.where(uncovered & (costs==minimum.gather(1,bins)),fine_reps[None],len(features))
    reps=torch.full((groups,kv),len(features),device=device,dtype=torch.long)
    reps.scatter_reduce_(1,bins,ids,reduce='amin',include_self=True)
    return reps.int(),dict(uncovered=uncovered,merged_labels=bins,populations=population,
                           attention_fine_counts=hit,uncovered_fine_count=remaining)


def compress_history(e,lse,patch_ids,special_count,capacity):
    """Same-pass conditional sparse tile-mass evidence; no Dense mass claim."""
    groups=len(patch_ids);rows=lse.shape[1];tiles=math.ceil(patch_ids.shape[1]/64)
    mass=(e-lse[:,:,None]).exp().sum(0)[:,math.ceil(special_count/64):]
    mass=F.pad(mass,(0,0,0,groups*32-rows)).reshape(groups,32,tiles).sum(1)
    sizes=(patch_ids.shape[1]-torch.arange(tiles,device=patch_ids.device)*64).clamp(0,64)
    order=mass.argsort(dim=1,descending=True,stable=True)
    ends=sizes[order].cumsum(1)
    ranks=torch.arange(capacity,device=patch_ids.device)[None].expand(groups,-1).contiguous()
    which=torch.searchsorted(ends.contiguous(),ranks,right=True)
    tile=order.gather(1,which);before=F.pad(ends,(1,0))[:,:-1].gather(1,which)
    return patch_ids.gather(1,tile*64+ranks-before).int()


class AttentionFirstState:
    def __init__(self,config):self.config=config

    def reset(self,frames,height,width,device):
        c=self.config;self.frames,self.height,self.width=frames,height,width
        self.patches,self.specials=token_layout(frames,height*width,5,device)
        self.n=frames*(height*width+5)
        self.budgets=[math.floor(r*self.n)-len(self.specials) for r in c.retentions]
        if min(self.budgets)<1:raise ValueError('Retention does not accommodate specials plus residual Value')
        if max(self.budgets)<frames:raise ValueError('Fine hierarchy requires at least one cluster per frame')
        self.fine=FullCoverageValue(frames,height*width,max(self.budgets),device,c.source_width,c.local_iterations)
        self.parent_count=self.fine.groups
        self.fine_parent=torch.arange(self.fine.base_count,device=device)//c.source_width
        self.parent_fine_count=torch.bincount(self.fine_parent,minlength=self.parent_count).float()
        self.parent_sizes=(self.fine.table>=0).sum(1)
        self.history=None;self.previous_signals=None;self.previous_layer=-1
        self.records=[];self.captured={}

    def pool(self,keys):
        fine_keys=keys[:,self.patches[self.fine.reps]].float()
        pooled=torch.zeros((len(keys),self.parent_count,keys.shape[-1]),device=keys.device)
        pooled.index_add_(1,self.fine_parent,fine_keys)
        return (pooled/self.parent_fine_count[None,:,None]).to(keys.dtype)

    def gate(self,q,pooled,features,layer,scale):
        groups=math.ceil(self.n/32);ng=min(groups,self.config.gate_groups)
        gs=torch.linspace(0,groups-1,ng,device=q.device).long()
        probes=(gs[:,None]*32+torch.tensor([4,12,20,28],device=q.device)).clamp_max(self.n-1)
        compact=q[:,probes].reshape(len(q),ng*4,-1)
        # shared_scores expects 4 real probes in each compact32-row group.
        z=compact@pooled.transpose(-1,-2)
        p=(z.float()*scale+self.parent_sizes.float().log()).softmax(-1).reshape(len(q),ng,4,-1).mean((0,2))
        prior=self.parent_sizes.float()/len(features)
        concentration=(p-prior).abs().sum(-1).mean()*.5
        lift=torch.zeros((),device=q.device)
        if self.history is not None and self.history.shape[1]:
            ids=self.history[gs].long()
            parent=self.fine.labels[ids]//self.config.source_width
            counts=torch.zeros((ng,self.parent_count),device=q.device)
            counts.scatter_add_(1,parent,torch.ones_like(parent,dtype=torch.float32))
            proxy=(p*counts/self.parent_sizes[None]).sum(1)
            uniform=ids.shape[1]/len(features)
            lift=((proxy-uniform)/max(1-uniform,1e-12)).clamp(0,1).mean()
        variance=(features-features.mean(0)).square().sum(-1).mean()
        distortion=(features-features[self.fine.reps[self.fine.labels]]).square().sum(-1).mean()
        relative=torch.where(variance>1e-12,distortion/variance.clamp_min(1e-12),torch.zeros_like(variance)).clamp(0,1)
        # One explicit scalar-vector synchronization per layer, counted in time.
        current=tuple(torch.stack((concentration,lift,relative)).tolist())
        ka,kv,signals=allocate_budget(self.budgets[layer],*current,self.previous_signals,self.config)
        self.previous_signals=(current[0],current[2])
        signals['gate_probe_groups']=ng
        return ka,kv,signals


def reference_union(q,k,v,specials,patch_keys,scale):
    """CPU reference: one softmax across specials AND the A/V union."""
    h,rows,d=q.shape;output=torch.empty_like(q)
    slots=math.ceil(len(specials)/64)+math.ceil(patch_keys.shape[1]/64)
    e=torch.full((h,rows,slots),-torch.inf,device=q.device);lse=torch.empty((h,rows),device=q.device)
    for group,ids in enumerate(patch_keys):
        start=group*32;stop=min(rows,start+32)
        all_ids=torch.cat((specials,ids.long()))
        z=(q[:,start:stop].float()@k[:,all_ids].float().transpose(-1,-2))*scale
        output[:,start:stop]=(z.softmax(-1)@v[:,all_ids].float()).to(q.dtype)
        lse[:,start:stop]=z.logsumexp(-1)
        offset=math.ceil(len(specials)/64)
        for tile in range(math.ceil(len(ids)/64)):
            e[:,start:stop,offset+tile]=z[:,:,len(specials)+tile*64:len(specials)+(tile+1)*64].logsumexp(-1)
    return output,e,lse


class AttentionFirstResidualValue(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config;layer=self.layer
        if self.training or len(x)!=1:raise ValueError('Batch-one inference only')
        if layer!=s.previous_layer+1:raise ValueError('Invalid causal layer order')
        q,k,v=prepare_qkv(o,x,pos);q,k=q.to(v.dtype).contiguous(),k.to(v.dtype).contiguous()
        with torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches]
            s.fine.update(features,layer)
            sums=torch.zeros((s.fine.base_count,features.shape[1]),device=x.device)
            sums.index_add_(0,s.fine.labels,features)
            sizes=torch.bincount(s.fine.labels,minlength=s.fine.base_count)
        pooled=s.pool(k)
        ka,kv,signals=s.gate(q,pooled,features,layer,o.scale)
        incoming=0 if s.history is None else s.history.shape[1]
        keep=min(incoming,math.floor(c.history_fraction*ka))
        groups=math.ceil(s.n/32)
        capacity=min(s.budgets[layer],math.floor(c.history_fraction*(s.budgets[layer+1]-1))) if layer<23 else 0
        output=torch.empty_like(v);histories=[];captures=[];uncovered_counts=[]
        pk,pv=k[:,s.specials].contiguous(),v[:,s.specials].contiguous()
        for start_group in range(0,groups,c.window_groups):
            length=min(c.window_groups,groups-start_group)
            start=start_group*32;stop=min(s.n,(start_group+length)*32)
            score=shared_scores(q,pooled,o.scale,s.parent_sizes.float().log(),start_group,length)
            history=None if not keep else s.history[start_group:start_group+length,:keep]
            attention,seen=choose_attention(score,s.fine.table,s.parent_sizes,s.fine.labels//c.source_width,
                history,ka,len(features))
            del seen
            with torch.autocast(x.device.type,enabled=False):
                value,coverage=residual_value(features,s.fine.labels,s.fine.reps,sums,sizes,attention,kv)
            union=torch.cat((attention,value),1)
            actual=s.patches[union.long()].int()
            if q.is_cuda:
                from saf3r.triton.fused_shared_av import shared_av
                result,e,lse=shared_av(q[:,start:stop],k,v,pk,pv,actual,0,scale=o.scale,evidence=bool(capacity))
            else:result,e,lse=reference_union(q[:,start:stop],k,v,s.specials,actual,o.scale)
            output[:,start:stop]=result
            if capacity:histories.append(compress_history(e,lse,union,len(s.specials),capacity))
            uncovered_counts.append(coverage['uncovered_fine_count'])
            if c.capture:captures.append(dict(attention=attention.clone(),value=value.clone(),
                uncovered=coverage['uncovered'].clone(),merged_labels=coverage['merged_labels'].clone(),
                populations=coverage['populations'].clone(),union=actual.clone()))
        s.history=torch.cat(histories) if capacity else None;s.previous_layer=layer
        s.records.append(dict(layer=layer,**signals,attention_keys=ka,value_keys=kv,special_keys=len(s.specials),
            patch_budget=s.budgets[layer],history_candidates=incoming,history_used=keep,
            history_discarded=incoming-keep,exploration_keys=ka-keep,history_sent_capacity=capacity,
            fine_clusters=s.fine.base_count,parents=s.parent_count,query_groups=groups,
            actual_unique_density=(s.budgets[layer]+len(s.specials))/s.n,
            final_unique_qk_per_head=s.n*(s.budgets[layer]+len(s.specials)),
            final_padded_qk_per_head=groups*32*(math.ceil(len(s.specials)/64)+math.ceil(s.budgets[layer]/64))*64,
            coarse_routing_qk_per_head=4*s.parent_count*(groups+signals['gate_probe_groups']),
            uncovered_fine_min=torch.cat(uncovered_counts).min(),
            phase_order=['budget','history','exploration','residual_value','unified_attention']))
        if c.capture:s.captured[layer]=dict(windows=captures,specials=s.specials.clone(),
            labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install_attention_first_residual_value(model,config=None,**kwargs):
    if any(hasattr(model,name) for name in ('attention_first_residual_value_state','value_region_av_state','fused_shared_av_state')):
        raise ValueError('Install on a fresh unpatched model, not on another A/V variant')
    config=AttentionFirstConfig(**kwargs) if config is None else config
    state=AttentionFirstState(config)
    if len(model.aggregator.global_blocks)!=24:raise ValueError('This entry supports24-global-layer VGGT')
    required=('qkv','q_norm','k_norm','rope','scale','head_dim','num_heads','proj','proj_drop')
    if any(not all(hasattr(block.attn,name) for name in required) for block in model.aggregator.global_blocks):
        raise ValueError('Expected original unpatched VGGT attention modules')
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=AttentionFirstResidualValue(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one image sequence required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_handle=model.aggregator.register_forward_pre_hook(reset)
    model.attention_first_residual_value_state=state
    return state
