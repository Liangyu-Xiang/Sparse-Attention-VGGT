"""Tensor-core block-sparse attention; return evidence from the SAME QK pass."""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, KP, VP, KS, VS, IX, CNT, HEADS, OUT, EVIDENCE,
             N: tl.constexpr, NP: tl.constexpr, NS: tl.constexpr, D: tl.constexpr,
             QB: tl.constexpr, MAX_BLOCKS: tl.constexpr, SCALE: tl.constexpr,
             BQ: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    block, selected_head = tl.program_id(0), tl.program_id(1)
    head = tl.load(HEADS+selected_head)
    count = tl.load(CNT+selected_head)
    qr = block*BQ+tl.arange(0, BQ)
    dd = tl.arange(0, BD)
    qq = tl.load(Q+head*N*D+qr[:, None]*D+dd[None, :],
                 (qr[:, None] < N) & (dd[None, :] < D), 0)
    maximum = tl.full((BQ,), -float('inf'), tl.float32)
    denom = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, BD), tl.float32)
    for slot in range(count):
        key_block = tl.load(IX+(selected_head*QB+block)*MAX_BLOCKS+slot)
        kr = key_block*BK+tl.arange(0, BK)
        kk = tl.load(KP+head*NP*D+kr[None, :]*D+dd[:, None],
                     (kr[None, :] < NP) & (dd[:, None] < D), 0)
        logits = tl.dot(qq, kk)*SCALE
        logits = tl.where(kr[None, :] < NP, logits, -float('inf'))
        local_max = tl.max(logits, 1)
        local_exp = tl.exp(logits-local_max[:, None])
        local_lse = local_max+tl.log(tl.sum(local_exp, 1))
        evidence = tl.sum(tl.where(qr < N, local_lse, 0.), 0)/tl.sum((qr < N).to(tl.float32), 0)
        tl.store(EVIDENCE+(selected_head*QB+block)*MAX_BLOCKS+slot, evidence)
        new_max = tl.maximum(maximum, local_max)
        correction = tl.exp(maximum-new_max)
        # Reuse the local exponential computed for same-pass evidence;
        # a second tile-wide exp is unnecessary.
        weights = local_exp*tl.exp(local_max-new_max)[:, None]
        vv = tl.load(VP+head*NP*D+kr[:, None]*D+dd[None, :],
                     (kr[:, None] < NP) & (dd[None, :] < D), 0)
        acc = acc*correction[:, None]+tl.dot(weights.to(vv.dtype), vv)
        denom = denom*correction+tl.sum(weights, 1)
        maximum = new_max
    # Special keys are mandatory, separate from the patch-block budget/history.
    for start in range(tl.cdiv(NS, BK)):
        kr = start*BK+tl.arange(0, BK)
        kk = tl.load(KS+head*NS*D+kr[None, :]*D+dd[:, None],
                     (kr[None, :] < NS) & (dd[:, None] < D), 0)
        logits = tl.dot(qq, kk)*SCALE
        logits = tl.where(kr[None, :] < NS, logits, -float('inf'))
        new_max = tl.maximum(maximum, tl.max(logits, 1))
        correction = tl.exp(maximum-new_max)
        weights = tl.exp(logits-new_max[:, None])
        vv = tl.load(VS+head*NS*D+kr[:, None]*D+dd[None, :],
                     (kr[:, None] < NS) & (dd[None, :] < D), 0)
        acc = acc*correction[:, None]+tl.dot(weights.to(vv.dtype), vv)
        denom = denom*correction+tl.sum(weights, 1)
        maximum = new_max
    tl.store(OUT+head*N*D+qr[:, None]*D+dd[None, :], acc/denom[:, None],
             (qr[:, None] < N) & (dd[None, :] < D))


def block_attention(q, patch_k, patch_v, special_k, special_v, indices, counts,
                    head_ids, *, query_block=64, key_block=64, scale=None, out=None):
    """Shared-within-query-block, head-specific exact sparse attention.

    All inputs contiguous [H,N,D], BF16/FP16. indices [sparseH,ceil(N/BQ),maxK]
    contains UNIQUE patch block IDs; counts vary across heads. No dense NxN mask.
    Evidence ranks mean log attention MASS per selected block, up to a constant
    shared by blocks; it reuses the same exact logits, not another QK check.
    """
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('Tensor-core sparse execution requires BF16 or FP16')
    h, n, d = q.shape
    if out is None:
        out = torch.empty_like(q)
    evidence = torch.full(indices.shape, -float('inf'), device=q.device, dtype=torch.float32)
    _forward[(triton.cdiv(n, query_block), len(head_ids))](
        q, patch_k, patch_v, special_k, special_v, indices, counts, head_ids, out, evidence,
        n, patch_k.shape[1], special_k.shape[1], d, indices.shape[1], indices.shape[2],
        d**-.5 if scale is None else scale, query_block, key_block,
        triton.next_power_of_2(d), num_warps=4, num_stages=2)
    return out, evidence
