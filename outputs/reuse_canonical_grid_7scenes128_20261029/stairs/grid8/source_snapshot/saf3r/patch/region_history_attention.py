"""Opt-in middle-layer region search: Dense0–9 / sparse10–16 / Dense17–23.

Spatial fine cells are the terminal search units. Search evaluates only pooled
intermediate Keys; it never evaluates Q against all fine representatives or
expands fine members for ranking. All filtered history edges are protected.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .layer_shared_av import LayerSharedAVState, spatial_hierarchy
from .value_clustering_gpu import stable_cluster_mean
from .value_evolving_attention import token_layout
from .residual_history_av import pack_mask, qk_logits, logits_av, coarse_probabilities


def table_from_labels(labels):
    groups=int(labels.max())+1
    count=torch.bincount(labels,minlength=groups)
    order=labels.argsort(stable=True);offset=count.cumsum(0)-count
    table=torch.full((groups,int(count.max())),-1,device=labels.device,dtype=torch.long)
    table[labels[order],torch.arange(len(labels),device=labels.device)-offset[labels[order]]]=order
    return table


def spatial_search_layout(frames,height,width,device,fine_ratio=.25):
    labels,reps,_=spatial_hierarchy(frames,height,width,fine_ratio,128,device)
    fh=max(1,min(height,round(height*math.sqrt(fine_ratio))))
    fw=max(1,min(width,round(height*width*fine_ratio/fh)))
    yy,xx=torch.meshgrid(torch.arange(fh,device=device),torch.arange(fw,device=device),indexing='ij')
    per_frame=math.ceil(fh/2)*math.ceil(fw/5)
    parent=(yy//2*math.ceil(fw/5)+xx//5)[None]+torch.arange(frames,device=device)[:,None,None]*per_frame
    # A2x5 parent has four connected children containing2–3 fine cells each.
    child=parent*4+(yy%2*2+(xx%5>=3).long())[None]
    _,child=torch.unique(child.flatten(),sorted=True,return_inverse=True)
    return labels,reps,parent.flatten(),child


def concentration(probability,tau=.8):
    covered=probability.sort(1,descending=True).values.cumsum(1)
    m=(covered<tau).sum(1).add(1).clamp_max(probability.shape[1])
    return 1-m.float()/probability.shape[1]


def pooled_group_probabilities(q,kfine,children,eligible):
    """Pool unresolved representative KEYS first, then perform group-level QK."""
    safe=children.clamp_min(0)
    active=eligible[:,safe]&(children[None]>=0)
    count=active.sum(2)
    pooled=torch.einsum('rgc,hgcd->hrgd',active.float(),kfine[:,safe].float())
    pooled=pooled/count.clamp_min(1)[None,:,:,None]
    z=(q.float()[:,:,None]*pooled).sum(-1)*q.shape[-1]**-.5
    z=z.masked_fill(count[None]==0,-torch.inf)
    return z.softmax(-1).nan_to_num().mean(0),count


def select_by_parent(probabilities,parent,eligible,quota):
    # Fine cells inherit their parent's score; no fine-representative QK ranking.
    score=probabilities[:,parent].masked_fill(~eligible,-torch.inf)
    order=score.argsort(dim=1,descending=True,stable=True)
    rank=torch.empty_like(order).scatter_(1,order,torch.arange(len(parent),device=order.device).expand_as(order))
    return eligible&(rank<quota[:,None])


def select_regions(q,kpatch,layout,coarse_prob,token_coarse,history,budget,variant='A'):
    labels,reps,parent,child=layout;rows=q.shape[1];nf=len(reps)
    nc=coarse_prob.shape[1]
    gate=torch.zeros_like(coarse_prob,dtype=torch.bool)
    gate.scatter_(1,coarse_prob.argsort(dim=1,descending=True,stable=True)[:,:math.ceil(nc*.5)],True)
    retained=torch.zeros((rows,len(labels)),device=q.device,dtype=torch.bool)
    if history is not None and history.numel():
        valid=history>=0;safe=history.long().clamp_min(0)
        valid=valid&gate.gather(1,token_coarse[safe])
        rr,cc=valid.nonzero(as_tuple=True)
        retained[rr,safe[rr,cc]]=True
    used=torch.zeros((rows,nf),device=q.device,dtype=torch.long)
    used.scatter_add_(1,labels.expand(rows,-1),retained.long())
    inherited=retained.sum(1)
    target=torch.maximum(inherited,torch.full_like(inherited,budget))
    missing=target-inherited
    eligible=(used==0)&(missing[:,None]>0)
    if bool((eligible.sum(1)<missing).any()):
        raise ValueError('Not enough unresolved fine cells; cannot fill budget with one edge per cell')
    p1,c1=pooled_group_probabilities(q,kpatch[:,reps],table_from_labels(parent),eligible)
    probes=c1.gt(0).sum(1)
    if variant=='A':
        selected=select_by_parent(p1,parent,eligible,missing)
    elif variant=='B':
        order=p1.argsort(dim=1,descending=True,stable=True)
        counts=c1.gather(1,order)
        desired=torch.maximum((eligible.sum(1).float()*.5).ceil().long(),missing)
        before=counts.cumsum(1)-counts
        chosen=torch.zeros_like(c1,dtype=torch.bool).scatter_(1,order,(before<desired[:,None])&(counts>0))
        second=eligible&chosen[:,parent]
        p2,c2=pooled_group_probabilities(q,kpatch[:,reps],table_from_labels(child),second)
        probes+=c2.gt(0).sum(1)
        selected=select_by_parent(p2,child,second,missing)
    else:raise ValueError('variant must be A or B')
    final=retained.clone()
    rr,ff=selected.nonzero(as_tuple=True)
    final[rr,reps[ff]]=True
    assert bool((final.sum(1)==target).all())
    assert not bool((selected&(used>0)).any())
    return pack_mask(final),dict(history_retained=inherited,explored=missing,
        intermediate_probes=probes,history_covered_fine=(used>0).sum(1))


def transfer_history(scores,patch_ids,coarse_p,budget,ratio,adaptive=False):
    widths=(concentration(coarse_p)*budget).floor().long() if adaptive else torch.full(
        (len(scores),),math.floor(ratio*budget),device=scores.device,dtype=torch.long)
    widths=torch.minimum(widths,(patch_ids>=0).sum(1))
    width=int(widths.max())
    order=scores.masked_fill(patch_ids<0,-torch.inf).argsort(dim=1,descending=True,stable=True)[:,:width]
    ids=patch_ids.gather(1,order)
    ids=ids.masked_fill(torch.arange(width,device=scores.device)[None]>=widths[:,None],-1)
    # Numeric order, not a carried-over score ranking; -1 is only padding.
    return ids.sort(1).values.to(torch.int32)


def prepare_qkv(original,x,pos):
    b,n,c=x.shape
    q,k,v=original.qkv(x).reshape(b,n,3,original.num_heads,original.head_dim).permute(2,0,3,1,4).unbind(0)
    q,k=original.q_norm(q),original.k_norm(k)
    if original.rope is not None:q,k=original.rope(q,pos),original.rope(k,pos)
    return tuple(t[0].contiguous() for t in (q,k,v))


class RegionHistoryState:
    def __init__(self,variant='A',transfer_ratio=.5,adaptive=False,retention=.1,query_chunk=128,capture=False):
        if variant not in ['A','B'] or not 0<=transfer_ratio<=1:raise ValueError('Invalid search settings')
        self.variant,self.transfer_ratio,self.adaptive=variant,transfer_ratio,adaptive
        self.retention,self.query_chunk,self.capture=retention,query_chunk,capture
        self.value=LayerSharedAVState()

    def reset(self,frames,height,width,device):
        self.value.reset(frames,height,width)
        self.frames,self.height,self.width=frames,height,width
        self.layout=spatial_search_layout(frames,height,width,device)
        self.history=None;self.records=[];self.captured={}


class RegionHistoryAttention(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        if self.training or x.shape[0]!=1:raise ValueError('Batch-one inference only')
        o,s=self.original,self.state;b,n,c=x.shape
        q,k,v=prepare_qkv(o,x,pos)
        patches,specials=token_layout(s.frames,s.height*s.width,5,x.device)
        with torch.autocast(x.device.type,enabled=False):
            labels,reps,mapping,_=s.value.update(v[:,patches].float().mean(0),self.layer)
        sparse=10<=self.layer<=16
        if not sparse:
            attended=F.scaled_dot_product_attention(q[None],k[None],v[None],scale=o.scale)[0]
        if 9<=self.layer<=16:
            with torch.autocast(x.device.type,enabled=False):
                kpatch=k[:,patches].float().contiguous()
                pooled=torch.stack([stable_cluster_mean(kh[reps],mapping,int(mapping.max())+1) for kh in kpatch])
            budget=math.floor(s.retention*n)-len(specials)
            if budget<=0:raise ValueError('Specials exceed budget')
            if sparse:attended=torch.empty_like(v)
            histories=[];indices=[];stats=[];concentrations=[]
            old=s.history
            for start in range(0,n,s.query_chunk):
                end=min(n,start+s.query_chunk)
                with torch.autocast(x.device.type,enabled=False):
                    if getattr(s,'coarse_size_bias',False):
                        population=torch.bincount(mapping[labels],minlength=pooled.shape[1]).float()
                        z=(q[:,start:end].float()@pooled.transpose(-1,-2))*o.scale
                        p=(z+population.clamp_min(1).log()).softmax(-1).mean(0)
                    else:p=coarse_probabilities(q[:,start:end].float(),pooled)
                    concentrations.append(concentration(p))
                    if sparse:
                        selector=getattr(s,'selector',select_regions)
                        patch_ids,record=selector(q[:,start:end].float(),kpatch,s.layout,p,mapping[labels],
                            None if old is None else old[start:end],budget,s.variant)
                        keys=torch.cat((patches[patch_ids.clamp_min(0)].masked_fill(patch_ids<0,-1),
                            specials.expand(end-start,-1)),1)
                        logits=qk_logits(q[:,start:end].to(v.dtype),k.to(v.dtype).contiguous(),keys,o.scale)
                        attended[:,start:end]=logits_av(logits,v,keys)
                        mean=logits.softmax(-1).mean(0)[:,:patch_ids.shape[1]]
                        stats.append(record)
                        if s.capture:indices.append(keys.cpu())
                    else:
                        # Dense L9 really executes Dense attention. Its already
                        # available true weights define a10%-budget seed mask.
                        dense=((q[:,start:end].float()@k.float().transpose(-1,-2))*o.scale).softmax(-1).mean(0)
                        rank=dense[:,patches].argsort(dim=1,descending=True,stable=True)[:,:budget]
                        patch_ids=rank;mean=dense[:,patches].gather(1,rank)
                    histories.append(transfer_history(mean,patch_ids,p,budget,s.transfer_ratio,s.adaptive))
            width=max(h.shape[1] for h in histories)
            s.history=torch.cat([F.pad(h,(0,width-h.shape[1]),value=-1) for h in histories])
            rec=dict(layer=self.layer,mode='sparse' if sparse else 'dense_seed',fine_cells=len(s.layout[1]),
                coarse_groups=pooled.shape[1],mean_concentration=float(torch.cat(concentrations).mean()),
                history_out=float((s.history>=0).sum(1).float().mean()))
            if sparse:
                for name in stats[0]:rec[name]=float(torch.cat([r[name] for r in stats]).float().mean())
                patch_counts=torch.cat([record['history_retained']+record['explored'] for record in stats])
                rec['density']=float((patch_counts+len(specials)).float().mean())/n
                if s.capture:
                    masks=torch.cat([F.pad(ids,(0,max(i.shape[1] for i in indices)-ids.shape[1]),value=-1) for ids in indices])
                    s.captured[self.layer]=dict(indices=masks,
                        q=q.cpu(),k=k.cpu(),v=v.cpu(),output=attended.cpu())
            s.records.append(rec)
        return o.proj_drop(o.proj(attended.transpose(0,1).reshape(b,n,c)))


def install_region_history_attention(model,**kwargs):
    state=RegionHistoryState(**kwargs)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=RegionHistoryAttention(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0];state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_hook=model.aggregator.register_forward_pre_hook(reset)
    model.region_history_state=state
    return state
