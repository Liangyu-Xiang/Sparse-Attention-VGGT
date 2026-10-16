"""Dense 0:2, representative-token Value 3:9/17:23, exact-QK TopK 10:16."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .hierarchical_value_attention import HierarchicalValueState
from .value_evolving_attention import token_layout


def layer_mode(layer):
    if not 0<=layer<24:
        raise ValueError('Expected zero-based global layer 0..23')
    return 'dense' if layer<3 else 'attention_topk' if 10<=layer<=16 else 'value_representatives'


def exact_topk_attention(q,k,v,patches,specials,total_budget,scale,query_chunk=64):
    """Full FP32 QK, chunked only on queries; exact token-wise TopK then AV.

    All special Keys are mandatory and COUNT toward total_budget. Select the
    remaining highest scoring patch Keys per head/query. Renormalize softmax.
    No dense AV, no repeated selected QK, no [H,N,N] retained allocation.
    """
    h,n,d=q.shape
    count=total_budget-len(specials)
    if not 0<=count<=len(patches) or query_chunk<1:
        raise ValueError('Invalid budget/chunk')
    if total_budget==k.shape[1]:
        return F.scaled_dot_product_attention(q[None],k[None],v[None],scale=scale)[0]
    out=torch.empty((h,n,v.shape[-1]),device=v.device,dtype=v.dtype)
    keys=k.float().transpose(1,2)
    for start in range(0,n,query_chunk):
        stop=min(n,start+query_chunk)
        # Disable outer BF16 autocast: preserve FP32 scores before TopK.
        with torch.autocast(device_type=q.device.type,enabled=False):
            scores=(q[:,start:stop].float()@keys)*scale
            best,ix=scores[:,:,patches].topk(count,dim=-1,sorted=False)
            ix=patches[ix]
            sp=specials[None,None].expand(h,stop-start,-1)
            selected=torch.cat((ix,sp),-1)
            logits=torch.cat((best,scores[:,:,specials]),-1)
            probabilities=logits.softmax(-1)
            if q.is_cuda:
                from saf3r.triton.indexed_weighted_value import indexed_weighted_value
                out[:,start:stop]=indexed_weighted_value(probabilities,selected,v)
            else:
                for head in range(h):
                    out[head,start:stop]=(probabilities[head,:,:,None]*v[head,selected[head]].float()).sum(1).to(v.dtype)
    return out


class HybridState:
    def __init__(self,retention=.10,query_chunk=64,coarse_size=128):
        if not 0<retention<=1 or query_chunk<1 or coarse_size<1:
            raise ValueError('Invalid hybrid parameters')
        self.retention,self.query_chunk,self.coarse_size=retention,query_chunk,coarse_size
        self.reset(1,1,1)

    def reset(self,frames,height,width):
        self.frames,self.height,self.width=frames,height,width
        npatch=frames*height*width;nspecial=5*frames
        self.total_budget=round(self.retention*(npatch+nspecial))
        self.rep_count=self.total_budget-nspecial
        # Tiny synthetic geometries may not fit mandatory specials at 10%.
        self.value=None
        if self.rep_count>0:
            ratio=self.rep_count/npatch
            self.value=HierarchicalValueState(retention=ratio,fine_ratio=ratio,
                coarse_size=self.coarse_size,initialization='lightweight',deterministic=True)
            self.value.reset(frames,height,width)
        self.records=[]


class HybridValueTopK(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        if self.training or len(x)!=1:
            raise ValueError('Inference batch one only')
        o,s=self.original,self.state
        if s.value is None:
            raise ValueError('10% budget cannot accommodate specials plus representatives')
        _,n,c=x.shape
        if n!=s.frames*(s.height*s.width+5):
            raise ValueError('Token layout/sequence reset mismatch')
        q,k,v=o.qkv(x).reshape(1,n,3,o.num_heads,o.head_dim).permute(2,0,3,1,4).unbind(0)
        patches,specials=token_layout(s.frames,s.height*s.width,5,x.device)
        hierarchy=None
        if self.layer>=2:
            with torch.autocast(device_type=x.device.type,enabled=False):
                hierarchy=s.value.update(v[0,:,patches].float().mean(0),self.layer+1)
        q,k=o.q_norm(q),o.k_norm(k)
        if o.rope is not None:q,k=o.rope(q,pos),o.rope(k,pos)
        q,k,v=q[0].to(v.dtype).contiguous(),k[0].to(v.dtype).contiguous(),v[0].contiguous()
        mode=layer_mode(self.layer)
        if mode=='dense':
            out=F.scaled_dot_product_attention(q[None],k[None],v[None],scale=o.scale)[0]
            kept=n
        elif mode=='value_representatives':
            reps=hierarchy[1]
            if len(reps)!=s.rep_count:raise RuntimeError('Representative count changed')
            selected=torch.cat((patches[reps],specials)).sort().values
            out=F.scaled_dot_product_attention(q[None],k[:,selected][None],v[:,selected][None],scale=o.scale)[0]
            kept=len(selected)
        else:
            out=exact_topk_attention(q,k,v,patches,specials,s.total_budget,o.scale,s.query_chunk)
            kept=s.total_budget
        s.records.append(dict(layer_zero_based=self.layer,mode=mode,total_tokens=n,
            retained_keys_per_query=kept,actual_density=kept/n,representative_count=s.rep_count if hierarchy else 0,
            value_updated=hierarchy is not None,special_keys=len(specials),
            seed=self.layer==2,query_chunk=s.query_chunk))
        return o.proj_drop(o.proj(out.transpose(0,1).reshape(1,n,c)))


def install_hybrid_value_topk(model,**kwargs):
    if hasattr(model,'hybrid_value_topk_state'):raise ValueError('Already installed')
    state=HybridState(**kwargs)
    for layer,b in enumerate(model.aggregator.global_blocks):
        b.attn=HybridValueTopK(b.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Expected batch-one images')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14)
    state.reset_hook_handle=model.aggregator.register_forward_pre_hook(reset)
    model.hybrid_value_topk_state=state
    return state
