"""Opt-in execution calibration; original shared statistics/mask budget remain.

rep_population compensates only the Rep state, whose loss assumes all members
are represented. History and Exact retain unit weights. cluster_prior is a
separate ablation that distributes each covered cluster's predicted mass over
its selected Keys without nearest-member ownership; dropped clusters stay lost.
cluster_population distributes the covered population equally over emitted
members; its fused CUDA path reuses the allocator's History member counts.
"""
import torch


def execution_weights(mode, selected, ids, reps, labels, population, prior, pi, eligible_count=None):
    if mode=='unit':
        return (ids>=0).to(prior.dtype)
    if mode=='cluster_population' and ids.is_cuda and eligible_count is not None:
        from saf3r.triton.shared_value_repair import cluster_weights
        return cluster_weights(selected,ids,labels,population,eligible_count)
    valid=ids>=0
    safe=ids.long().clamp_min(0)
    clusters=labels[safe]
    if mode=='rep_population':
        is_rep=selected.gather(1,clusters)==1
        return torch.where(valid,torch.where(is_rep,population[clusters],1.),0.)
    if mode=='cluster_prior':
        own=prior.gather(1,safe).masked_fill(~valid,0.)
        kept=torch.zeros_like(pi).scatter_add_(1,clusters,own)
        weights=pi.gather(1,clusters)/kept.gather(1,clusters).clamp_min(1e-30)
        return weights.masked_fill(~valid,0.)
    if mode=='cluster_population':
        count=torch.zeros_like(pi).scatter_add_(1,clusters,valid.to(pi.dtype))
        return (population[clusters]/count.gather(1,clusters).clamp_min(1.)).masked_fill(~valid,0.)
    raise ValueError(mode)
