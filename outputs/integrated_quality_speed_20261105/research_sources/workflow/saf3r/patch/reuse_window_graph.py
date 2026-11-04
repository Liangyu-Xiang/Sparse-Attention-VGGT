"""Capture execution of unchanged window selection; no cached input values.

Geometry is copied anew once per current layer; priors, history, eligibility
are overwritten before EVERY replay. Reusable graphs cache topology only.
"""
import torch
from . import reuse_inference_ops as rule
from saf3r.triton.reuse_compact import budget
from saf3r.triton.reuse_drop_moment import drop_norm

def eager(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,uniform=None):
    context=rule.gate(prior,pi,population,old_ids,labels,reps)
    if geometry.get('shared_value_loss',False):
        if uniform is None:raise RuntimeError('Missing per-Grid uniform probabilities for shared Value loss')
        from saf3r.triton import reuse_shared_value_states as shared
        residuals=shared.corrections(prior,old_ids,uniform,labels,geometry,context)
        errors,costs=shared.states(uniform,allowed,population,geometry,context,residuals)
    elif geometry.get('fuse_state_kernel',False) and geometry['width']<=64:
        if geometry.get("p0_loss_pack",False):
            from saf3r.triton.reuse_p0_states import states
        else:
            from saf3r.triton.reuse_fused_states import states
        errors,costs=states(prior,z,reps,labels,population,allowed,geometry,context)
    else:
        geometry['drop_values']=drop_norm(prior,z,geometry,len(reps),geometry['drop_bins'])
        errors,costs=rule.states(pi,z,reps,labels,population,None,allowed,prior,drop_multiplier,geometry,context)
    selected,count=budget(geometry['objective'],costs,errors[:,:,0].amax(1)+1e-12,cap,iterations)
    selected=rule.decode(selected)
    if geometry.get("p0_loss_pack",False):
        from saf3r.triton.reuse_p0_pack import pack
        ids,real,record=pack(selected,reps,labels,population,cap,geometry,context)
    else:
        ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context)
    if geometry.get('shared_value_loss',False):
        # No population or nearest-owner compensation: each selected Key has unit weight.
        weights=(ids>=0).to(dtype=prior.dtype)
        context['mass_error']=torch.zeros(len(prior),device=prior.device)
    else:
        weights=rule.transfer(selected,labels,reps,None,prior,pi,ids,geometry,context)
    record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context))
    record['history_cap_discarded']=torch.zeros(len(prior),device=prior.device)
    return ids,real,weights,record

def _signature(value):
    """Include every captured structural value; tensor contents remain runtime inputs."""
    if torch.is_tensor(value):
        return (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        return tuple((k, _signature(v)) for k, v in sorted(value.items())
                     if k not in ('objective', 'drop_values', 'drop_bin_counts'))
    if isinstance(value, (tuple, list)):return tuple(_signature(v) for v in value)
    return value


class Window:
    def __init__(self,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,bucket_bins=False,uniform=None):
        if geometry.get('shared_value_loss',False) and uniform is None:raise ValueError('Missing shared Value uniform input')
        sources=[prior,pi,old_ids,allowed]+([uniform] if uniform is not None else [])
        self.inputs=[torch.empty_like(t) for t in sources]
        for dest,src in zip(self.inputs,sources):dest.copy_(src)
        self.constants=[t.clone() for t in [z,reps,labels,population]]
        self.geometry={k:(v.clone() if torch.is_tensor(v) else v) for k,v in geometry.items() if k not in ['drop_values','objective','drop_bins']}
        self.geometry['drop_bins']=[(size,ids.clone()) for size,ids in geometry['drop_bins']]
        if bucket_bins:
            import triton
            self.geometry['drop_bins']=[(size,torch.empty(triton.next_power_of_2(len(ids)),device=ids.device,dtype=ids.dtype)) for size,ids in geometry['drop_bins']]
            self.geometry['drop_bin_counts']=[torch.empty(1,device=z.device,dtype=torch.int32) for _ in geometry['drop_bins']]
            self.update_constants(z,reps,labels,population,geometry)
        def run():
            return eager(*self.inputs[:4],*self.constants,self.geometry,cap,iterations,drop_multiplier,
                         uniform=self.inputs[4] if len(self.inputs)>4 else None)
        run()  # Compile, initialize constants, allocate outside capture.
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.ids,self.real,self.weights,record=run()
            self.keys={}
            for k,v in record.items():self.keys.setdefault(v.dtype,[]).append(k)
            self.records={dtype:torch.stack([record[k] for k in keys]) for dtype,keys in self.keys.items()}
    def update_constants(self,z,reps,labels,population,geometry):
        for dst,src in zip(self.constants,[z,reps,labels,population]):dst.copy_(src)
        for k,v in geometry.items():
            if k not in ('drop_values','objective') and k in self.geometry and torch.is_tensor(v):self.geometry[k].copy_(v)
        for i,((size,dst),(_,src)) in enumerate(zip(self.geometry['drop_bins'],geometry['drop_bins'])):
            dst[:len(src)].copy_(src)
            if 'drop_bin_counts' in self.geometry:self.geometry['drop_bin_counts'][i].fill_(len(src))
    def __call__(self,prior,pi,old_ids,allowed,uniform=None):
        sources=[prior,pi,old_ids,allowed]+([uniform] if uniform is not None else [])
        if len(sources)!=len(self.inputs):raise ValueError('Mismatched shared Value graph inputs')
        for dest,src in zip(self.inputs,sources):dest.copy_(src)
        self.graph.replay()
        record={}
        for dtype,static in self.records.items():
            values=static.clone().unbind(0)
            record.update(zip(self.keys[dtype],values))
        return self.ids,self.real,self.weights,record

def select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
    geometry["p0_loss_pack"]=getattr(state.config,"p0_loss_pack",False)
    if getattr(state.config,'reuse_window_graphs',False) and geometry.get('fuse_state_kernel',False) and geometry['width']<=64:
        import triton
        from collections import OrderedDict
        if not hasattr(state,'_reused_window_graphs'):state._reused_window_graphs=OrderedDict()
        if getattr(state.config,'static_shared_graph',False):
            key=_signature((prior,pi,old_ids,allowed,uniform,z,reps,labels,population,geometry,
                            state.budget,state.config.dual_iterations,state.config.drop_multiplier))
        else:
            key=(tuple(prior.shape),len(reps),geometry['width'],tuple((size,triton.next_power_of_2(len(ids))) for size,ids in geometry['drop_bins']),bool(geometry.get('shared_value_loss',False)))
        tag=(state._forward_serial,layer)
        graphs=state._reused_window_graphs
        if key not in graphs:
            state.graph_cache_misses+=1
            # Evict before allocating the next private graph pool to bound peak usage.
            if getattr(state.config,'static_shared_graph',False):
                while graphs and (len(graphs)>=state.config.graph_cache_capacity or
                       sum(getattr(w,'cache_bytes',0) for w in graphs.values())>=state.config.graph_cache_max_mib*2**20):
                    graphs.popitem(last=False);state.graph_cache_evictions+=1
            before=torch.cuda.memory_allocated(prior.device)
            graphs[key]=Window(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier,True,uniform=uniform)
            graphs[key].cache_bytes=max(0,torch.cuda.memory_allocated(prior.device)-before)
            state.graph_cache_creations=getattr(state,'graph_cache_creations',0)+1
            graphs[key].loaded=tag
            if len(graphs)>getattr(state.config,'graph_cache_capacity',24):
                graphs.popitem(last=False);state.graph_cache_evictions+=1
        else:state.graph_cache_hits+=1
        window=graphs[key];graphs.move_to_end(key)
        if window.loaded!=tag:
            window.update_constants(z,reps,labels,population,geometry);window.loaded=tag;state.graph_constant_updates+=1
        if getattr(state.config,'static_shared_graph',False):
            limit=state.config.graph_cache_max_mib*2**20
            while graphs and sum(getattr(w,'cache_bytes',0) for w in graphs.values())>limit:
                graphs.popitem(last=False);state.graph_cache_evictions+=1
            state.graph_cache_bytes=sum(getattr(w,'cache_bytes',0) for w in graphs.values())
        return window(prior,pi,old_ids,allowed,uniform=uniform)
    if getattr(state,'_window_graph_layer',None)!=layer:
        state._window_graphs={};state._window_graph_layer=layer
    key=(len(prior),geometry['width'],bool(geometry.get('shared_value_loss',False)))
    if key not in state._window_graphs:
        state._window_graphs[key]=Window(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier,uniform=uniform)
    return state._window_graphs[key](prior,pi,old_ids,allowed,uniform=uniform)
