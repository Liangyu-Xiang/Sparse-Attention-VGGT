"""Layer budgets independent of the fixed Value-cluster initialization count.

Retention counts all keys, including the always-retained special keys. Sparsity
is one minus retention. Profiles are declared before geometric quality is read.
"""
import math


PROFILES = {
    "uniform16": [0.16] * 24,
    "uniform18": [0.18] * 24,
    "middle22_mean16": [0.10] * 6 + [0.22] * 12 + [0.10] * 6,
    "middle24_mean17": [0.10] * 6 + [0.24] * 12 + [0.10] * 6,
    "middle24_mean18": [0.12] * 6 + [0.24] * 12 + [0.12] * 6,
    "uniform20": [0.20] * 24,
    "middle24": [0.16] * 6 + [0.24] * 12 + [0.16] * 6,
    "middle28": [0.12] * 6 + [0.28] * 12 + [0.12] * 6,
    "middle30": [0.15] * 8 + [0.30] * 8 + [0.15] * 8,
    "middle26_mean21": [0.16] * 6 + [0.26] * 12 + [0.16] * 6,
}


def validate_schedule(retention, layers):
    if not math.isfinite(retention) or not 0.05 <= retention <= 0.5:
        raise ValueError("Base retention must be finite and within [0.05, 0.5]")
    if layers is not None and (
        len(layers) != 24
        or any(not math.isfinite(r) or not 0.05 <= r <= 0.5 for r in layers)
    ):
        raise ValueError("Exactly 24 finite layer retentions in [0.05, 0.5] required")


def patch_budget(total_keys, special_keys, retention):
    if not 0 <= special_keys < total_keys:
        raise ValueError("Invalid key geometry")
    if not math.isfinite(retention) or not 0 < retention <= 1:
        raise ValueError("Invalid retention")
    budget = math.floor(total_keys * retention) - special_keys
    if budget < 1:
        raise ValueError("Retention cannot accommodate the required special keys")
    return budget


def total_edge_patch_budget(total_keys,special_keys,dense_queries,retention):
    """Exact integer cap after the existing full-Key special-Query correction."""
    if not 0<=dense_queries<total_keys:raise ValueError('Invalid Dense special-query geometry')
    edge_cap=math.floor(retention*total_keys*total_keys)
    remaining=edge_cap-dense_queries*total_keys
    if remaining<0:raise ValueError('Special queries alone exceed the requested total-edge cap')
    keys=remaining//(total_keys-dense_queries)
    result=keys-special_keys
    if result<1:raise ValueError('Total-edge cap cannot accommodate required Special Keys')
    return result

def layer_budget(config, layer, total_keys, special_keys):
    if not 0 <= layer < 24:
        raise ValueError("Layer must be in [0, 23]")
    rates = config.layer_retentions
    rate = config.retention if rates is None else rates[layer]
    if getattr(config,'total_edge_retention',False):
        if special_keys%5:raise ValueError('Expected five Special Keys per frame')
        return total_edge_patch_budget(total_keys,special_keys,special_keys//5*config.dense_special_queries,rate)
    return patch_budget(total_keys, special_keys, rate)


def profile_summary(rates, total_keys, special_keys):
    budgets = [patch_budget(total_keys, special_keys, rate) for rate in rates]
    densities = [(b + special_keys) / total_keys for b in budgets]
    mean = sum(densities) / len(densities)
    return dict(layer_patch_budgets=budgets,layer_total_key_density_caps=densities,
                mean_total_key_density_cap=mean,equivalent_sparsity_cap=1-mean,
                note="Budget caps; report actual QK-weighted measured density separately")


def begin_history_layer(state, query):
    """Keep the prior layer's compact cache intact until all windows consume it.

    Kernels index dense buffers by the actual support width. Slicing a maximum
    capacity buffer would give the wrong group stride, so allocate an exact
    contiguous buffer when the budget changes. Equal-width windows can overwrite
    their own consumed history in place, as in the original execution path.
    """
    import torch
    old = state.ready_edges
    if old.shape[-1] == state.budget:
        state.ready_next_edges = old
    else:
        shape = (*old.shape[:-1], state.budget)
        if math.prod(shape)*old.element_size() > 8*2**30:
            raise MemoryError("Next-layer history exceeds explicit 8 GiB cache limit")
        state.ready_next_edges = torch.empty(shape,device=query.device,dtype=old.dtype)


def finish_history_layer(state):
    state.ready_edges = state.ready_next_edges
    state.ready_next_edges = None
