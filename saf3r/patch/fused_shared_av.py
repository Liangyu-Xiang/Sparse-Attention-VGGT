"""All24 global layers: full-coverage Value representatives + evolving Attention.

New opt-in approximation:32 queries share support, never shared output/weights.
No dense seed, no Dense fallback, no per-token N*N masks or retained logits.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .region_history_attention import prepare_qkv,spatial_search_layout,table_from_labels
from .value_evolving_attention import token_layout
from .value_clustering_gpu import centroid_nearest_gpu


DEFAULT_ALPHA=(0.,)+(.2,)*9+(.8,)*7+(.2,)*7


@contextmanager
def stage(state,name):
    if not state.profiling:
        yield
        return
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record()
    yield
    b.record();state.profile_events.append((state.previous_layer+1,name,a,b))


@dataclass(frozen=True)
class FusedSharedAVConfig:
    attention_ratios:tuple=DEFAULT_ALPHA
    retentions:tuple=(.2,)*24
    query_block:int=32
    key_tile:int=64
    window_groups:int=128
    history_fraction:float=.5
    local_iterations:int=2
    source_width:int=32
    capture:bool=False

    def __post_init__(self):
        if len(self.attention_ratios)!=24 or len(self.retentions)!=24:raise ValueError('24layer schedules required')
        if any(not math.isfinite(x) or not 0<=x<=1 for x in self.attention_ratios):raise ValueError('Invalid alpha')
        if any(not math.isfinite(x) or not .05<=x<=.2 for x in self.retentions):raise ValueError('Final budget must be5–20%')
        if self.attention_ratios[0]!=0:raise ValueError('L0 is Value-only, no Dense seed')
        if (self.query_block,self.key_tile)!=(32,64) or self.window_groups<1:raise ValueError('Unsupported tile/window')
        if not 0<=self.history_fraction<=1 or self.local_iterations<1:raise ValueError('Invalid history/update settings')
        if self.source_width<1:raise ValueError('Invalid local candidate width')


def patch_budgets(tokens,specials,config):
    total=[math.floor(r*tokens)-specials for r in config.retentions]
    if min(total)<1:raise ValueError('Budget cannot contain specials and representatives')
    attention=[round(p*a) for p,a in zip(total,config.attention_ratios)]
    return total,attention,[p-a for p,a in zip(total,attention)]


def means(x,labels,count):
    out=torch.zeros((count,x.shape[-1]),device=x.device,dtype=torch.float32)
    out.index_add_(0,labels,x.float())
    size=torch.zeros(count,device=x.device,dtype=torch.float32)
    size.index_add_(0,labels,torch.ones(len(labels),device=x.device))
    return out/size.clamp_min(1)[:,None]


class FullCoverageValue:
    """Fixed spatial-source membership, anchored synchronous fine updates.

    The requested Value partition is a MERGE of all base fine cells. Every
    token remains covered, unlike taking a high-population representative subset.
    Source groups have a static table, eliminating token/chunk Python loops.
    """
    def __init__(self,frames,patches_per_frame,base_count,device,width=32,iterations=2):
        if not frames<=base_count<=frames*patches_per_frame:raise ValueError('Invalid frame-local base count')
        self.base_count,self.width,self.iterations=base_count,width,iterations
        quotas=torch.full((frames,),base_count//frames,device=device,dtype=torch.long)
        quotas+=torch.arange(frames,device=device)<base_count%frames
        offsets=quotas.cumsum(0)-quotas
        labels=(torch.arange(patches_per_frame,device=device)[None]*quotas[:,None]//patches_per_frame+offsets[:,None]).flatten()
        self.labels=labels
        first=torch.full((base_count,),len(labels),device=device,dtype=torch.long)
        first.scatter_reduce_(0,labels,torch.arange(len(labels),device=device),reduce='amin')
        self.reps=first
        self.groups=math.ceil(base_count/width)
        source=labels//width
        counts=torch.bincount(source,minlength=self.groups)
        starts=counts.cumsum(0)-counts
        capacity=width*math.ceil(patches_per_frame/(base_count//frames))
        self.table=torch.full((self.groups,capacity),-1,device=device,dtype=torch.long)
        self.table[source,torch.arange(len(labels),device=device)-starts[source]]=torch.arange(len(labels),device=device)
        self.candidates=torch.arange(self.groups*width,device=device).reshape(self.groups,width)
        self.valid=self.candidates<base_count
        self.merges={}

    def update(self,x,layer):
        if layer:
            safe=self.table.clamp_min(0);group_x=x[safe].float()
            own=torch.arange(self.base_count,device=x.device)
            anchors=self.reps.clone()
            for _ in range(self.iterations):
                representatives=x[self.reps[self.candidates.clamp_max(self.base_count-1)]].float()
                # Full FP32 local squared distances; bounded by static source tables.
                distances=(group_x.square().sum(-1)[:,:,None]+representatives.square().sum(-1)[:,None]
                    -2*torch.bmm(group_x,representatives.transpose(1,2)))
                choice=distances.masked_fill(~self.valid[:,None],torch.inf).argmin(-1)
                assigned=self.candidates.gather(1,choice)
                labels=self.labels.clone()
                # Padding targets sentinel N, then discard it (no dynamic nonzero).
                padded=F.pad(labels,(0,1))
                padded.scatter_(0,torch.where(self.table>=0,self.table,len(labels)).flatten(),assigned.flatten())
                labels=padded[:-1];labels[anchors]=own
                self.labels=labels
                self.reps=centroid_nearest_gpu(x,labels,self.reps)
        else:self.reps=centroid_nearest_gpu(x,self.labels,self.reps)

    def representatives(self,x,count):
        if not 0<=count<=self.base_count:raise ValueError('Value count exceeds base hierarchy')
        if count==0:return self.reps[:0],None
        if count==self.base_count:return self.reps,self.labels
        mapping=self.merges.get(count)
        if mapping is None:
            mapping=torch.arange(self.base_count,device=x.device)*count//self.base_count
            self.merges[count]=mapping
        labels=mapping[self.labels]
        first=torch.full((count,),len(x),device=x.device,dtype=torch.long)
        first.scatter_reduce_(0,mapping,self.reps,reduce='amin')
        return centroid_nearest_gpu(x,labels,first),labels


def shared_scores(q,pooled,scale,log_population,start_group,groups,block=32,query_offset=0):
    offsets=torch.tensor([4,12,20,28],device=q.device)
    query=(torch.arange(start_group,start_group+groups,device=q.device)[:,None]*block+offsets).clamp_max(q.shape[1]-1)
    if query_offset:query=(query+query_offset)%q.shape[1]
    probe=q[:,query].reshape(q.shape[0],groups*4,q.shape[-1])
    scores=probe@pooled.transpose(-1,-2)
    if q.is_cuda:
        from saf3r.triton.shared_av_routing import routing_probabilities
        return routing_probabilities(scores,log_population,groups,scale)
    logits=scores.float()*scale+log_population
    return logits.softmax(-1).reshape(q.shape[0],groups,4,-1).mean((0,2))


def pack_disjoint_history(history,is_value):
    if history is None:return None,None
    valid=(history>=0)&~is_value[history.long().clamp_min(0)]
    return history.masked_fill(~valid,-1),valid.sum(1)


def compact_prefix(first,second,width):
    """Keep valid first IDs, then valid second IDs, without token-wide sorting."""
    ids=torch.cat((first,second),1);valid=ids>=0
    positions=valid.long().cumsum(1)-1
    # A dedicated sentinel column receives all padding writes.
    out=torch.full((len(ids),width+1),-1,device=ids.device,dtype=torch.int32)
    target=torch.where(valid&(positions<width),positions,width)
    out.scatter_(1,target,ids.int())
    return out[:,:width].contiguous()


def history_from_mass(e,lse,common,attention,value_count,history_count,block=32,tile=64):
    """Rank support tiles by exact mean probability MASS from this same QK pass.

    Retaining complete/partial tiles is an explicit approximation to token TopK.
    No old attention values are carried; return only distinct real patch indices.
    """
    groups=len(attention)
    if history_count==0:return torch.empty((groups,0),device=common.device,dtype=torch.int32)
    mass=(e-lse[:,:,None]).exp().sum(0)
    mass=F.pad(mass,(0,0,0,groups*block-len(mass))).reshape(groups,block,-1).sum(1)
    ps=math.ceil(len(common)/tile);ats=math.ceil(attention.shape[1]/tile)
    counts=torch.cat(((value_count-torch.arange(ps,device=common.device)*tile).clamp(0,tile),
        (attention.shape[1]-torch.arange(ats,device=common.device)*tile).clamp(0,tile)))
    order=mass.masked_fill(counts[None]==0,-1).argsort(dim=1,descending=True,stable=True)
    ordered_counts=counts[order];ends=ordered_counts.cumsum(1)
    ranks=torch.arange(history_count,device=common.device).expand(groups,-1).contiguous()
    which=torch.searchsorted(ends.contiguous(),ranks,right=True)
    slots=order.gather(1,which)
    starts=F.pad(ends,(1,0))[:,:-1].gather(1,which)
    within=ranks-starts
    common_col=(slots*tile+within).clamp_max(len(common)-1)
    if attention.shape[1]:
        att_col=((slots-ps)*tile+within).clamp(0,attention.shape[1]-1)
        return torch.where(slots<ps,common[common_col],attention.gather(1,att_col)).int()
    return common[common_col].int()


class FusedSharedAVState:
    def __init__(self,config):self.config=config;self.profiling=False

    def reset(self,frames,height,width,device):
        c=self.config;self.frames,self.height,self.width=frames,height,width
        self.patches,self.specials=token_layout(frames,height*width,5,device)
        self.n=frames*(height*width+5)
        self.total,self.ka,self.kv=patch_budgets(self.n,len(self.specials),c)
        self.value=FullCoverageValue(frames,height*width,max(self.kv),device,c.source_width,c.local_iterations)
        self.layout=spatial_search_layout(frames,height,width,device)
        labels,reps,parent,child=self.layout
        self.members=table_from_labels(labels)
        self.child_members=table_from_labels(child)
        self.parent,self.child=parent,child
        self.parent_count=int(parent.max())+1;self.child_count=int(child.max())+1
        population=torch.bincount(labels,minlength=len(reps)).float()
        self.parent_population=torch.zeros(self.parent_count,device=device).index_add_(0,parent,population)
        self.child_population=torch.zeros(self.child_count,device=device).index_add_(0,child,population)
        self.parent_log_population=self.parent_population.clamp_min(1).log()
        self.child_log_population=self.child_population.clamp_min(1).log()
        self.history=None;self.records=[];self.captured={};self.previous_layer=-1;self.profile_events=[]
        self.pooled_parent=None;self.pooled_child=None


def pooled_keys(keys,reps,mapping,count):
    # Cached mapping; pooled once/layer, shared by all Query groups.
    out=torch.zeros((keys.shape[0],count,keys.shape[-1]),device=keys.device,dtype=torch.float32)
    out.index_add_(1,mapping,keys[:,reps].float())
    sizes=torch.zeros(count,device=keys.device).index_add_(0,mapping,torch.ones_like(mapping,dtype=torch.float32))
    return (out/sizes.clamp_min(1)[None,:,None]).to(keys.dtype)


def select_attention(q,keys,state,is_value,history,value_ids,ka,start_group,groups,scale):
    """Bounded group window; pools remain global and are never rebuilt per Query."""
    labels,reps,parent,child=state.layout;nf=len(reps)
    clean,kept=pack_disjoint_history(history,is_value)
    if clean is None:
        clean=torch.empty((groups,0),device=q.device,dtype=torch.int32)
        kept=torch.zeros(groups,device=q.device,dtype=torch.long)
    missing=ka-kept
    used=torch.zeros((groups,nf),device=q.device,dtype=torch.int32)
    if len(value_ids):used[:,labels[value_ids]]=1
    if history is not None and history.shape[1]:
        valid=history>=0;safe=history.long().clamp_min(0)
        used.scatter_add_(1,labels[safe],valid.int())
    eligible=used==0
    p1=shared_scores(q,state.pooled_parent,scale,state.parent_log_population,start_group,groups)
    counts=torch.zeros((groups,state.parent_count),device=q.device,dtype=torch.long)
    counts.scatter_add_(1,parent.expand(groups,-1),eligible.long())
    order=p1.argsort(dim=1,descending=True,stable=True);ordered=counts.gather(1,order)
    desired=torch.maximum((eligible.sum(1)+1)//2,missing)
    chosen=torch.zeros_like(counts,dtype=torch.bool).scatter_(1,order,(ordered.cumsum(1)-ordered)<desired[:,None])
    eligible&=chosen[:,parent]
    p2=shared_scores(q,state.pooled_child,scale,state.child_log_population,start_group,groups)
    # Normalize child evidence only inside selected parents; common denominator
    # does not change ordering. No fake exact-member mass claim.
    # Rank small child groups, then linearly compact their eligible fine cells.
    # Sorting most of all33k fine scores per row was a major avoidable cost.
    child_count=torch.zeros((groups,state.child_count),device=q.device,dtype=torch.long)
    child_count.scatter_add_(1,child.expand(groups,-1),eligible.long())
    child_order=p2.masked_fill(child_count==0,-1).argsort(dim=1,descending=True,stable=True)
    candidates=state.child_members[child_order].flatten(1)
    valid=(candidates>=0)&eligible.gather(1,candidates.clamp_min(0))
    selected=compact_prefix(torch.empty((groups,0),device=q.device,dtype=torch.int32),candidates.masked_fill(~valid,-1),min(ka,nf)).long()
    if q.is_cuda:
        from saf3r.triton.shared_av_terminal import terminal_keys_tensorcore
        with stage(state,'terminal'):
            picked=terminal_keys_tensorcore(q[:,start_group*32:min(q.shape[1],(start_group+groups)*32)],keys,state.patches,selected,state.members,missing,scale)
    else:
        picked=torch.full_like(selected,-1,dtype=torch.int32)
        for row in range(groups):
            qr=(torch.tensor([4,12,20,28])+32*(start_group+row)).clamp_max(q.shape[1]-1)
            for col in range(int(missing[row])):
                ids=state.members[selected[row,col]];ids=ids[ids>=0]
                z=torch.einsum('hpd,hmd->hpm',q[:,qr].float(),keys[:,state.patches[ids]].float())*scale
                picked[row,col]=ids[z.softmax(-1).mean((0,1)).argmax()]
    active=torch.arange(selected.shape[1],device=q.device)[None]<missing[:,None]
    member_count=(state.members>=0).sum(1)
    return compact_prefix(clean,picked,ka),dict(terminal_fine_pairs=missing.sum(),
        incoming_history=kept.sum(),terminal_capacity=selected.numel(),
        terminal_member_qk_per_head=4*(member_count[selected]*active).sum())


class FusedSharedAVAttention(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config
        if self.training or len(x)!=1:raise ValueError('Batch-one inference only')
        if self.layer!=s.previous_layer+1:raise ValueError('State must advance one layer at a time')
        q,k,v=prepare_qkv(o,x,pos)
        # RoPE/normalization can promote Q/K to FP32 under BF16 autocast.
        # The deployed tensor-core attention consumes the original V dtype.
        q,k=q.to(v.dtype).contiguous(),k.to(v.dtype).contiguous()
        patches,specials=s.patches,s.specials
        with stage(s,'value'),torch.autocast(x.device.type,enabled=False):
            # FP32 accumulation WITHOUT an H*N*D FP32 intermediate or full
            # patch-only V gather. This avoids hundreds of MiB at128frames.
            features=v.mean(0,dtype=torch.float32)[patches]
            s.value.update(features,self.layer)
            value_ids,merged=s.value.representatives(features,s.kv[self.layer])
        common=torch.cat((patches[value_ids],specials))
        with stage(s,'pack'):
            pk=k[:,common].contiguous();pv=v[:,common].contiguous()
        ka=s.ka[self.layer];kv=len(value_ids)
        is_value=torch.zeros(len(patches),device=x.device,dtype=torch.bool);is_value[value_ids]=True
        if ka:
            with stage(s,'pool'):
                global_reps=patches[s.layout[1]]
                s.pooled_parent=pooled_keys(k,global_reps,s.parent,s.parent_count)
                s.pooled_child=pooled_keys(k,global_reps,s.child,s.child_count)
        groups=math.ceil(s.n/c.query_block)
        next_history=min(s.total[self.layer],math.floor(c.history_fraction*s.ka[self.layer+1])) if self.layer<23 else 0
        output=torch.empty_like(v);histories=[];captured=[];stats=[]
        for g in range(0,groups,c.window_groups):
            length=min(c.window_groups,groups-g);start=g*32;end=min(s.n,(g+length)*32)
            if ka:
                old=None if s.history is None else s.history[g:g+length]
                with stage(s,'selection_including_terminal'):
                    attention,record=select_attention(q,k,s,is_value,old,value_ids,ka,g,length,o.scale)
                stats.append(record)
            else:attention=torch.empty((length,0),device=x.device,dtype=torch.int32)
            # Convert patch IDs only once for the final kernel and history.
            actual=patches[attention.long().clamp_min(0)].masked_fill(attention<0,-1).int()
            if q.is_cuda:
                from saf3r.triton.fused_shared_av import shared_av
                with stage(s,'execution'):
                    result,e,lse=shared_av(q[:,start:end],k,v,pk,pv,actual,kv,scale=o.scale,evidence=bool(next_history))
            else:
                result,e,lse=reference_attention(q[:,start:end],k,v,common,actual,kv,o.scale)
            output[:,start:end]=result
            if next_history:
                with stage(s,'history'):
                    indices=history_from_mass(e,lse,common,actual,kv,next_history)
                # Token-layout inverse, global real IDs -> patch-local IDs.
                local=indices.long()-(indices.long()//(s.height*s.width+5)+1)*5
                histories.append(local.int())
            if c.capture:captured.append(actual.clone())
        s.history=torch.cat(histories) if next_history else None
        # Routing banks are ephemeral: do not keep the last layer's pools alive
        # through the DPT head or into the next input sequence.
        s.pooled_parent=None;s.pooled_child=None
        s.previous_layer=self.layer
        record=dict(layer=self.layer,attention_edges=ka,value_edges=kv,special_keys=len(specials),
            total_edges=ka+kv+len(specials),density=(ka+kv+len(specials))/s.n,
            attention_ratio=c.attention_ratios[self.layer],history_out=next_history,
            base_clusters=s.value.base_count,query_groups=groups,
            padded_final_qk_ratio=(math.ceil(len(common)/64)*64+math.ceil(ka/64)*64)/s.n,
            routing_qk_per_head=4*groups*(s.parent_count+s.child_count) if ka else 0,
            final_unique_qk_per_head=s.n*(ka+kv+len(specials)),
            final_padded_qk_per_head=groups*32*(math.ceil(len(common)/64)*64+math.ceil(ka/64)*64),
            source='full_coverage_Value_plus_history_exploration',cold_seed=self.layer==0)
        if stats:
            for name in stats[0]:record[name]=sum(z[name] for z in stats)
        s.records.append(record)
        if c.capture:s.captured[self.layer]=dict(common=common.clone(),attention=torch.cat(captured),
            base_labels=s.value.labels.clone(),value_labels=None if merged is None else merged.clone(),value_representatives=value_ids.clone())
        # Release projection-independent banks BEFORE the output projection.
        # Keeping all three contiguous Q/K/V copies alive through reshape/proj
        # created a higher transient peak than fused Dense attention.
        del q,k,v,pk,pv,features
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def reference_attention(q,k,v,common,attention,value_count,scale):
    """Small CPU reference; never used by the CUDA inference path."""
    h,r,d=q.shape;ps=math.ceil(len(common)/64);slots=ps+math.ceil(attention.shape[1]/64)
    out=torch.empty_like(q);e=torch.full((h,r,slots),-torch.inf,device=q.device);lse=torch.empty((h,r),device=q.device)
    for row in range(r):
        ids=torch.cat((common,attention[row//32].long()));valid=ids>=0;safe=ids.clamp_min(0)
        z=(q[:,row,None].float()*k[:,safe].float()).sum(-1)*scale
        z=z.masked_fill(~valid[None],-torch.inf);lse[:,row]=z.logsumexp(-1)
        out[:,row]=(z.softmax(-1)[:,:,None]*v[:,safe].float()).sum(1)
        for slot in range(ps):e[:,row,slot]=z[:,slot*64:min((slot+1)*64,value_count)].logsumexp(-1)
        for slot in range(slots-ps):e[:,row,ps+slot]=z[:,len(common)+slot*64:len(common)+(slot+1)*64].logsumexp(-1)
    return out,e,lse


def install_fused_shared_av(model,config=None,**kwargs):
    if hasattr(model,'fused_shared_av_state'):raise ValueError('Already installed')
    config=FusedSharedAVConfig(**kwargs) if config is None else config
    state=FusedSharedAVState(config)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=FusedSharedAVAttention(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one image sequence required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_hook=model.aggregator.register_forward_pre_hook(reset)
    model.fused_shared_av_state=state
    return state
