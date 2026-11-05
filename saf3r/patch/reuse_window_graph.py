"""Capture execution of unchanged window selection; no cached input values.

Geometry is copied anew once per current layer; priors, history, eligibility
are overwritten before EVERY replay. Reusable graphs cache topology only.
"""
import torch
from . import reuse_inference_ops as rule
from saf3r.triton.reuse_compact import budget
from saf3r.triton.reuse_drop_moment import drop_norm

def eager(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,uniform=None,special_p=None):
    if geometry.get('state_loss_variant','legacy')!='legacy':
        if special_p is None:raise ValueError('Consistent state loss requires current coarse Special probabilities')
        from .state_loss_controller import select_window
        return select_window(prior,pi,old_ids,allowed,reps,labels,population,geometry,cap,iterations,special_p)
    context=rule.gate(prior,pi,population,old_ids,labels,reps)
    if geometry.get('shared_value_loss',False):
        if geometry.get('shared_value_allow_observed_rep',False):
            # Rep and History are alternative states, so this cannot duplicate
            # a Key. Historical observation is evidence, not disqualification.
            context['remove_rep']=torch.zeros_like(context['remove_rep'])
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
        from .shared_value_repair import execution_weights
        weights=execution_weights(geometry.get('shared_value_weighting','unit'),selected,ids,reps,labels,population,prior,pi,context['eligible_count'])
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


def _bin_capacity(size, clusters, tokens):
    # A bin (size/2, size] cannot contain more than N/(size/2+1) clusters.
    # This bound depends only on structural sizes, never on data values.
    import triton
    minimum=1 if size==1 else size//2+1
    return triton.next_power_of_2(max(1,min(clusters,tokens//minimum)))


class Window:
    def __init__(self,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,bucket_bins=False,uniform=None,special_p=None):
        if geometry.get('shared_value_loss',False) and geometry.get('state_loss_variant','legacy')=='legacy' and uniform is None:raise ValueError('Missing shared Value uniform input')
        self.has_uniform=uniform is not None;self.has_special=special_p is not None
        sources=[prior,pi,old_ids,allowed]+([uniform] if self.has_uniform else [])+([special_p] if self.has_special else [])
        self.inputs=[torch.empty_like(t) for t in sources]
        for dest,src in zip(self.inputs,sources):dest.copy_(src)
        # Shared-Value selection consumes the already reduced norm/distance
        # statistics in geometry, not the original N x feature_dim matrix.
        # Retaining a private copy per graph needlessly thrashes the cache for
        # full-head features. All non-shared loss paths still receive z.
        if geometry.get('shared_value_loss',False):z=z.new_empty((0,))
        self.constants=[t.clone() for t in [z,reps,labels,population]]
        self.geometry={k:(v.clone() if torch.is_tensor(v) else v) for k,v in geometry.items() if k not in ['drop_values','objective','drop_bins']}
        self.geometry['drop_bins']=[(size,ids.clone()) for size,ids in geometry['drop_bins']]
        if bucket_bins:
            import triton
            self.geometry['drop_bins']=[(size,torch.empty((_bin_capacity(size,len(reps),len(labels)) if bucket_bins=='stable' else triton.next_power_of_2(max(1,len(ids)))),device=ids.device,dtype=ids.dtype)) for size,ids in geometry['drop_bins']]
            self.geometry['drop_bin_counts']=[torch.empty(1,device=z.device,dtype=torch.int32) for _ in geometry['drop_bins']]
            self.update_constants(z,reps,labels,population,geometry)
        def run():
            return eager(*self.inputs[:4],*self.constants,self.geometry,cap,iterations,drop_multiplier,
                         uniform=self.inputs[4] if self.has_uniform else None,
                         special_p=self.inputs[4+int(self.has_uniform)] if self.has_special else None)
        run()  # Compile, initialize constants, allocate outside capture.
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.ids,self.real,self.weights,record=run()
            self.keys={}
            for k,v in record.items():self.keys.setdefault(v.dtype,[]).append(k)
            self.records={dtype:torch.stack([record[k] for k in keys]) for dtype,keys in self.keys.items()}
    def update_constants(self,z,reps,labels,population,geometry):
        for index,(dst,src) in enumerate(zip(self.constants,[z,reps,labels,population])):
            if index==0 and dst.numel()==0:continue
            dst.copy_(src)
        for k,v in geometry.items():
            if k not in ('drop_values','objective') and k in self.geometry and torch.is_tensor(v):self.geometry[k].copy_(v)
        for i,((size,dst),(source_size,src)) in enumerate(zip(self.geometry['drop_bins'],geometry['drop_bins'])):
            if size!=source_size or len(src)>len(dst):raise ValueError('Graph bin structure/capacity mismatch; recapture required')
            dst[:len(src)].copy_(src)
            if 'drop_bin_counts' in self.geometry:self.geometry['drop_bin_counts'][i].fill_(len(src))
    def __call__(self,prior,pi,old_ids,allowed,uniform=None,special_p=None):
        sources=[prior,pi,old_ids,allowed]+([uniform] if uniform is not None else [])+([special_p] if special_p is not None else [])
        if len(sources)!=len(self.inputs):raise ValueError('Mismatched shared Value graph inputs')
        for dest,src in zip(self.inputs,sources):dest.copy_(src)
        self.graph.replay()
        record={}
        for dtype,static in self.records.items():
            values=static.clone().unbind(0)
            record.update(zip(self.keys[dtype],values))
        return self.ids,self.real,self.weights,record

def select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None,special_p=None):
    geometry["p0_loss_pack"]=getattr(state.config,"p0_loss_pack",False)
    if getattr(state.config,'reuse_window_graphs',False) and geometry.get('fuse_state_kernel',False) and (geometry['width']<=64 or (getattr(state.config,'stable_binned_graph',False) and geometry.get('shared_value_loss',False) and geometry['width']<=512)):
        import triton
        from collections import OrderedDict
        if not hasattr(state,'_reused_window_graphs'):state._reused_window_graphs=OrderedDict()
        bounded=getattr(state.config,'stable_binned_graph',False)
        if bounded:
            signature_geometry=dict(geometry,drop_bins=[(size,_bin_capacity(size,len(reps),len(labels)),str(ids.dtype),str(ids.device)) for size,ids in geometry['drop_bins']])
        else:signature_geometry=geometry
        key=_signature((prior,pi,old_ids,allowed,uniform,special_p,z,reps,labels,population,signature_geometry,
                        state.budget,state.config.dual_iterations,state.config.drop_multiplier))
        tag=(state._forward_serial,layer)
        graphs=state._reused_window_graphs
        if key not in graphs:
            state.graph_cache_misses+=1
            # Evict before allocating the next private graph pool to bound peak usage.
            if getattr(state.config,'static_shared_graph',False) or bounded:
                while graphs and (len(graphs)>=state.config.graph_cache_capacity or
                       sum(getattr(w,'cache_bytes',0) for w in graphs.values())>=state.config.graph_cache_max_mib*2**20):
                    graphs.popitem(last=False);state.graph_cache_evictions+=1
            before=torch.cuda.memory_allocated(prior.device)
            graphs[key]=Window(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier,'stable' if bounded else True,uniform=uniform,special_p=special_p)
            graphs[key].cache_bytes=max(0,torch.cuda.memory_allocated(prior.device)-before)
            state.graph_cache_creations=getattr(state,'graph_cache_creations',0)+1
            graphs[key].loaded=tag
            if len(graphs)>getattr(state.config,'graph_cache_capacity',24):
                graphs.popitem(last=False);state.graph_cache_evictions+=1
        else:state.graph_cache_hits+=1
        window=graphs[key];graphs.move_to_end(key)
        if window.loaded!=tag:
            window.update_constants(z,reps,labels,population,geometry);window.loaded=tag;state.graph_constant_updates+=1
        if getattr(state.config,'static_shared_graph',False) or bounded:
            limit=state.config.graph_cache_max_mib*2**20
            while graphs and sum(getattr(w,'cache_bytes',0) for w in graphs.values())>limit:
                graphs.popitem(last=False);state.graph_cache_evictions+=1
            state.graph_cache_bytes=sum(getattr(w,'cache_bytes',0) for w in graphs.values())
        return window(prior,pi,old_ids,allowed,uniform=uniform,special_p=special_p)
    if getattr(state,'_window_graph_layer',None)!=layer:
        state._window_graphs={};state._window_graph_layer=layer
    key=_signature((prior,pi,old_ids,allowed,uniform,special_p,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier))
    if key not in state._window_graphs:
        state._window_graphs[key]=Window(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier,uniform=uniform,special_p=special_p)
    return state._window_graphs[key](prior,pi,old_ids,allowed,uniform=uniform,special_p=special_p)
