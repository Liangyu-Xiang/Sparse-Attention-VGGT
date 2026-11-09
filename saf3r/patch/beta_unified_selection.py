"""Replay identical four-state selection, refreshing every numerical input."""
from .history_transfer_states import select as reference
ENABLED=True
_INSTALLED=False

def select(state,layer,prior,pi,old,allowed,z,reps,labels,population,geometry):
    global _INSTALLED
    c=state.config
    if not ENABLED:return reference(prior,pi,old,allowed,z,reps,labels,population,geometry,state.budget,c.dual_iterations)
    from . import reuse_window_graph as graph
    if not _INSTALLED:
        original=graph.eager
        def eager(prior,pi,old,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,uniform=None,special_p=None):
            if geometry.get('unified_history_sqrt',False):return reference(prior,pi,old,allowed,z,reps,labels,population,geometry,cap,iterations)
            return original(prior,pi,old,allowed,z,reps,labels,population,geometry,cap,iterations,drop_multiplier,uniform,special_p)
        graph.eager=eager;_INSTALLED=True
    geo=dict(geometry,shared_value_loss=True,unified_history_sqrt=True)
    return graph.select(state,layer,prior,pi,old,allowed,z,reps,labels,population,geo,uniform=prior.new_zeros(len(prior),1))
