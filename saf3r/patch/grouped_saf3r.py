"""Batch heads with identical static SAF3R patterns; preserve scalar dynamic heads.

No profile, Top-K, mask, scale or retained Key is changed. Query-probe and DINO
heads retain the original implementation and lazy correspondence scheduling.
"""
from collections import OrderedDict
from types import MethodType
import torch
import torch.nn.functional as F
from .wrapper import BaseHeadwiseSparseAttentionWrapper

def _head_view(value,heads):
    if heads==list(range(heads[0],heads[-1]+1)):return value[:,heads[0]:heads[-1]+1]
    return value[:,heads]

def _static_bank(module,n,per,config,head,device):
    stride=config.get('stride',4);start=head%stride if config.get('shift',False) else 0
    include=config.get('include_as',True);key=(n,per,stride,start,include,str(device))
    cache=module.grouped_static_indices
    if key not in cache:
        ids=torch.arange(start,n,stride,device=device)
        if include:
            specials=(torch.arange(n//per,device=device)[:,None]*per+torch.arange(5,device=device)[None]).flatten()
            ids=torch.cat((ids,specials,torch.arange(per,device=device))).unique(sorted=True)
        cache[key]=ids
        while len(cache)>module.grouped_geometry_cache_capacity:cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]

def grouped_compute(self,q,k,v):
    b,h,n,d=q.shape;per=self.patch_height*self.patch_width+5;frames=n//per
    output=torch.empty(v.shape,device=v.device,dtype=v.dtype);groups=OrderedDict()
    for head,config in enumerate(self.head_cfg):
        mode=config['mode']
        if mode=='stride':
            stride=config.get('stride',4);start=head%stride if config.get('shift',False) else 0
            key=(mode,stride,start,config.get('include_as',True))
        elif mode in ('all_to_first','global_to_frame','full'):key=(mode,)
        else:key=('scalar',head)
        groups.setdefault(key,[]).append(head)
    pairs=0;probe_pairs=0;lazy_pairs=0
    for signature,heads in groups.items():
        config=self.head_cfg[heads[0]];mode=config['mode']
        if signature[0]=='scalar':
            head=heads[0];first_lazy=(mode=='dino_topk' and self.lazy_dino_topk and config.get('is_first_dino',False) and not self.first_dino_flag)
            result=self.sparse_attention(q[:,head:head+1],k[:,head:head+1],v[:,head:head+1],config,head)
            if mode=='q_probe_topk':
                if config.get('include_as',False):raise ValueError('Grouped execution accounting requires the frozen probe profile without anchor union')
                count=max(1,n//config['stride']) if 'stride' in config else min(config.get('topk',1024),n)
                pairs+=n*count;probe_pairs+=n
            elif mode=='dino_topk':
                special=frames*5;patch=n-special;count=self.est_topk_idx.shape[-1]
                pairs+=special*len(range(0,n,config.get('dino_sp2sp_stride',1)))+patch*count
                if first_lazy:lazy_pairs+=patch*patch
            else:raise ValueError('Unsupported frozen profile mode: '+mode)
            output[:,head:head+1]=result
            continue
        qg=_head_view(q,heads)
        if mode=='all_to_first':
            kg=_head_view(k[:,:,:per],heads);vg=_head_view(v[:,:,:per],heads);count=per
            result=F.scaled_dot_product_attention(qg,kg,vg)
        elif mode=='global_to_frame':
            shape=(b,len(heads),frames,per,d)
            qg=qg.reshape(shape).permute(0,2,1,3,4).reshape(b*frames,len(heads),per,d)
            kg=_head_view(k,heads).reshape(shape).permute(0,2,1,3,4).reshape_as(qg)
            vg=_head_view(v,heads).reshape(b,len(heads),frames,per,v.shape[-1]).permute(0,2,1,3,4).reshape(b*frames,len(heads),per,v.shape[-1])
            result=F.scaled_dot_product_attention(qg,kg,vg).reshape(b,frames,len(heads),per,v.shape[-1]).permute(0,2,1,3,4).reshape(b,len(heads),n,v.shape[-1]);count=per
        elif mode=='stride':
            ids=_static_bank(self,n,per,config,heads[0],q.device);count=len(ids)
            head_ids=torch.tensor(heads,device=q.device)
            kg=k[:,head_ids[:,None],ids[None,:]];vg=v[:,head_ids[:,None],ids[None,:]]
            result=F.scaled_dot_product_attention(qg,kg,vg)
        else:
            count=n;result=F.scaled_dot_product_attention(qg,_head_view(k,heads),_head_view(v,heads))
        if heads==list(range(heads[0],heads[-1]+1)):output[:,heads[0]:heads[-1]+1]=result
        else:output[:,heads]=result
        pairs+=len(heads)*n*count
    self.grouped_execution_record=dict(main_qk_pairs=pairs,mean_unique_density=pairs/(h*n*n),
        query_probe_qk_pairs=probe_pairs,lazy_correspondence_qk_pairs=lazy_pairs,
        execution_groups=len(groups),heads=h,geometry_cache_banks=len(self.grouped_static_indices),
        geometry_cache_bytes=sum(t.numel()*t.element_size() for t in self.grouped_static_indices.values()))
    return output

def install_grouped_saf3r(model,geometry_cache_capacity=4,release_cuda_cache_each_block=False):
    # Cache eviction is allocator housekeeping, not an attention/quality gate.
    # The original defaults remain True; this explicit execution variant lets
    # PyTorch reuse freed blocks instead of purging them24times per forward.
    model.aggregator.release_cuda_cache_each_block=release_cuda_cache_each_block
    originals=[]
    for block in model.aggregator.global_blocks:
        module=block.attn
        if not isinstance(module,BaseHeadwiseSparseAttentionWrapper):raise ValueError('Install grouped execution on the original SAF3R-patched model')
        if hasattr(module,'grouped_static_indices'):raise ValueError('Grouped execution already installed')
        originals.append(module.compute_multihead_attention)
        module.grouped_static_indices=OrderedDict();module.grouped_geometry_cache_capacity=geometry_cache_capacity
        module.compute_multihead_attention=MethodType(grouped_compute,module)
    return originals
