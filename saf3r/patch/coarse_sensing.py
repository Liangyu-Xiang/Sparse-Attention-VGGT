"""Head-preserving coarse relevance, independent of group-population priors.

Value grouping may be shared; Q/K head axes must not be mixed before QK.
Population is returned separately for capacity/allocation, not reliability.
The legacy mean-Q/mean-K path exists ONLY for controlled ablation.
"""
from __future__ import annotations
import math
import torch


def pool_keys(keys, token_groups, num_groups):
    """[head, patch key, d] -> pooled [head, group, d], token populations."""
    if keys.ndim != 3 or len(token_groups) != keys.shape[1]:
        raise ValueError('Expected head-wise keys and one shared token-group map')
    population = torch.zeros(num_groups, device=keys.device, dtype=torch.float32)
    population.index_add_(0, token_groups, torch.ones(len(token_groups), device=keys.device))
    if bool((population == 0).any()):
        raise ValueError('Coarse groups must be nonempty')
    pooled = torch.zeros((len(keys), num_groups, keys.shape[-1]), device=keys.device, dtype=torch.float32)
    pooled.index_add_(1, token_groups, keys.float())
    return pooled/population[None, :, None], population


def relevance_logits(queries, pooled_keys, scale=None):
    """Matched-head Q_h Kbar_h^T, no cross-head terms or size prior."""
    if queries.ndim != 3 or pooled_keys.ndim != 3 or len(queries) != len(pooled_keys):
        raise ValueError('Q and pooled K must have the same head axis')
    scale = queries.shape[-1]**-.5 if scale is None else scale
    with torch.autocast(device_type=queries.device.type, enabled=False):
        return torch.bmm(queries.float(), pooled_keys.float().transpose(1, 2))*scale


def reliability_statistics(probabilities):
    """C,S,CS independently per head (or scalar for a single shared matrix).

    Query axis is -2, never averaged together with heads. CS is an uncalibrated
    statistic, NOT automatically a justified fraction of the final edge budget.
    """
    if probabilities.ndim not in (2, 3):
        raise ValueError('Expected [query,group] or [head,query,group]')
    groups = probabilities.shape[-1]
    if groups == 1:
        zero = probabilities.new_zeros(probabilities.shape[:-2])
        return dict(concentration=zero, specificity=zero, cs=zero)
    p = probabilities.float()
    h_query = -(p*p.clamp_min(1e-30).log()).sum(-1).mean(-1)
    marginal = p.mean(-2)
    h_marginal = -(marginal*marginal.clamp_min(1e-30).log()).sum(-1)
    normalizer = math.log(groups)
    c = (1-h_query/normalizer).clamp(0, 1)
    s = ((h_marginal-h_query)/normalizer).clamp(0, 1)
    return dict(concentration=c, specificity=s, cs=c*s)


def coarse_attention(queries, keys, token_groups, num_groups, *,
                     mode='per_head', population_bias=False, scale=None):
    """Correct relevance by default; optional legacy/biased ablation controls.

    With per_head mode output has [H,Q,C]; legacy_mean_qk has [1,Q,C].
    When population_bias=True the returned probabilities are mass-proxy scores,
    NOT the default relevance signal used to define reliability.
    """
    pooled, population = pool_keys(keys, token_groups, num_groups)
    if mode == 'per_head':
        logits = relevance_logits(queries, pooled, scale)
    elif mode == 'legacy_mean_qk':
        logits = relevance_logits(queries.float().mean(0, keepdim=True), pooled.mean(0, keepdim=True), scale)
    else:
        raise ValueError('Unknown sensing mode')
    if population_bias:
        logits = logits+population.log()[None, None]
    p = logits.softmax(-1)
    return p, reliability_statistics(p), population
