"""Whole-Parent budget prefixes; no Child/Fine Attention ranking."""
import torch


def parent_prefix(scores,available,missing,topk=0):
    if topk:
        if scores.is_cuda:
            from saf3r.triton.parent_partial_topk import partial_parent_prefix
            return partial_parent_prefix(scores,available,missing,topk)
        # CPU oracle retains exact stable ordering; CUDA uses partial Top-K
        # and GPU-only completion, tested against this reference.
    order=scores.masked_fill(available==0,-torch.inf).argsort(dim=1,descending=True,stable=True)
    counts=available.gather(1,order)
    before=counts.cumsum(1)-counts
    selected=(counts>0)&(before<missing[:,None])
    return order,selected,dict(partial_topk_initial=0,partial_fallback_groups=0)
