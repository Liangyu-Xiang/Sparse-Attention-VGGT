"""Optional full-key attention for a small, explicit set of special queries.

This is an algorithm variant, not an equivalent execution optimization. The
ordinary sparse observer remains the history stream; the selected special
queries receive a separate dense output. Both computations must be timed and
reported, including their overlapping QK work. No NxN attention matrix is built.
"""
import torch.nn.functional as F


def query_ids(special_ids, per_frame):
    if per_frame not in (0,1,5):
        raise ValueError('Dense special-query count must be 0, 1 (camera), or 5')
    if special_ids.numel()%5:
        raise ValueError('Expected camera plus four register tokens per frame')
    return special_ids.reshape(-1,5)[:,:per_frame].reshape(-1)


def correct_output(q,k,v,output,special_ids,per_frame,scale):
    ids=query_ids(special_ids,per_frame)
    if ids.numel():
        dense=F.scaled_dot_product_attention(q[:,ids].unsqueeze(0),k.unsqueeze(0),
                                            v.unsqueeze(0),dropout_p=0.,is_causal=False,scale=scale)
        output[:,ids]=dense[0]
    return output,ids
