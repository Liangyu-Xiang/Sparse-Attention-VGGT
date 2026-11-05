"""Explicit hybrid: original early head profile, then independent Value routing.

The prefix is an algorithm change, not an equivalent kernel optimization. It
uses no DINO/correspondence Top-K head (the frozen profile starts those later).
The tail starts fresh clustering/history and counts all of that work in forward.
"""
import copy,json,math
from pathlib import Path
import torch
from .patch_vggt import HeadwiseSparseAttentionWrapper
from .reuse_residual_exploration import ReuseExploreState,ReuseResidualExploration

PROFILE=Path(__file__).resolve().parents[2]/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json'

def prefix_density(groups,frames,patches):
    per=patches+5;n=frames*per;special={f*per+i for f in range(frames) for i in range(5)}
    total=0;heads=0
    for cfg in groups:
        for h in cfg['head_ids']:
            mode=cfg['mode']
            if mode in ('all_to_first','global_to_frame'):keys=per
            elif mode=='stride':
                stride=cfg.get('stride',4);start=h%stride if cfg.get('shift',False) else 0
                ids=set(range(start,n,stride))
                if cfg.get('include_as',True):ids.update(special);ids.update(range(per))
                keys=len(ids)
            elif mode=='q_probe_topk':
                if cfg.get('include_as',False):raise ValueError('Prefix density requires explicit probe/anchor union counting')
                keys=max(1,n//cfg['stride']) if 'stride' in cfg else min(cfg.get('topk',1024),n)
            else:raise ValueError('Unsupported early head profile: '+mode)
            total+=keys;heads+=1
    if heads!=16:raise ValueError('Expected all16 profiled heads')
    return total/(heads*n)

class Prefix(HeadwiseSparseAttentionWrapper):
    def __init__(self,original,state,layer,groups):
        super().__init__(original,config=copy.deepcopy(groups),lazy_dino_topk=False)
        self.state=state;self.layer=layer;self.groups=groups
    @property
    def original(self):return self.orig_module
    def forward(self,x,pos=None):
        if self.layer!=self.state.prefix_previous_layer+1:raise ValueError('Ordered native prefix required')
        result=super().forward(x,pos);s=self.state;s.prefix_previous_layer=self.layer
        if s.config.collect_statistics:
            density=prefix_density(self.groups,s.frames,s.height*s.width)
            s.records.append(dict(layer=self.layer,native_prefix=True,profile=str(PROFILE),head_profile=self.groups,
                actual_unique_density=density,final_unique_qk_per_head=density*s.n*s.n,
                protect_first_frame=False,protected_patch_keys=None,semantic_special_keys=s.frames*5,
                dense_special_query_count=0,scope='Original frozen early SAF3R head patterns; no lazy/DINO correspondence Top-K in this prefix'))
        return result

class Tail(ReuseResidualExploration):
    def __init__(self,original,state,local_layer,offset):
        super().__init__(original,state,local_layer);self.offset=offset
    def forward(self,x,pos=None):
        result=super().forward(x,pos)
        if self.state.config.collect_statistics:
            row=self.state.records[-1];row['tail_layer']=self.layer;row['layer']=self.layer+self.offset
            row['native_prefix']=False
        return result

def install_native_prefix(model,config):
    if hasattr(model,'reuse_explore_state'):raise ValueError('Fresh unpatched model required')
    if config.deterministic_reductions:raise ValueError('Set deterministic mode explicitly in the diagnostic harness for this hybrid')
    profile=json.loads(PROFILE.read_text());count=config.native_prefix_layers
    state=ReuseExploreState(config);state.last_layer=23-count
    prefix=[]
    for layer,block in enumerate(model.aggregator.global_blocks):
        if layer<count:
            block.attn=Prefix(block.attn,state,layer,profile[str(layer)]).eval();prefix.append(block.attn)
        else:block.attn=Tail(block.attn,state,layer-count,count).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one sequences required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
        state.prefix_previous_layer=-1
        for attention in prefix:
            attention.patch_height=state.height;attention.patch_width=state.width
            attention.est_topk_idx=torch.empty(0,device=images.device);attention.first_dino_flag=False
    state.reset_handle=model.aggregator.register_forward_pre_hook(reset)
    model.reuse_explore_state=state
    return state
