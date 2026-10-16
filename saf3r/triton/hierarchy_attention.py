"""Indexed attention with exact logits returned from the SAME QK evaluation."""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(Q, K, V, I, S, O, L, N: tl.constexpr, D: tl.constexpr,
             BUDGET: tl.constexpr, NS: tl.constexpr, ROWS: tl.constexpr,
             QS: tl.constexpr, QH: tl.constexpr, KH: tl.constexpr,
             SCALE: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    ds = tl.arange(0, BD)
    qq = tl.load(Q + head*QH + row*QS + ds, ds < D, 0).to(tl.float32)
    m = tl.full((), -float('inf'), tl.float32)
    den = tl.full((), 0., tl.float32)
    acc = tl.zeros((BD,), tl.float32)
    for start in range(tl.cdiv(BUDGET + NS, BK)):
        col = start*BK + tl.arange(0, BK)
        special = col >= BUDGET
        patch_idx = tl.load(I + row*BUDGET + col, col < BUDGET, 0)
        special_idx = tl.load(S + col-BUDGET, special & (col < BUDGET+NS), 0)
        ids = tl.where(special, special_idx, patch_idx)
        kk = tl.load(K + head*KH + ids[:, None]*D + ds[None, :],
                     (col[:, None] < BUDGET+NS) & (ds[None, :] < D), 0).to(tl.float32)
        logits = tl.sum(kk*qq[None, :], 1)*SCALE
        logits = tl.where(col < BUDGET+NS, logits, -float('inf'))
        # Patch logits alone participate in the next layer's patch history.
        tl.store(L + (head*ROWS+row)*BUDGET + col, logits, col < BUDGET)
        new_m = tl.maximum(m, tl.max(logits, 0))
        correction = tl.exp(m-new_m)
        weights = tl.exp(logits-new_m)
        vv = tl.load(V + head*KH + ids[:, None]*D + ds[None, :],
                     (col[:, None] < BUDGET+NS) & (ds[None, :] < D), 0).to(tl.float32)
        acc = acc*correction + tl.sum(weights[:, None]*vv, 0)
        den = den*correction + tl.sum(weights, 0)
        m = new_m
    tl.store(O + (head*ROWS+row)*D + ds, acc/den, ds < D)


def indexed_attention_with_logits(q, k, v, indices, specials, scale):
    """[H,Q,D], [H,N,D], shared [Q,K]; never computes dense QK.

    K/V are contiguous full-sequence tensors; Q can be a strided query slice.
    Special keys are disjoint from patch indices and included in softmax.
    """
    h, rows, d = q.shape
    out = torch.empty_like(q, memory_format=torch.contiguous_format)
    logits = torch.empty((h, rows, indices.shape[1]), device=q.device, dtype=torch.float32)
    _forward[(rows, h)](q, k, v, indices, specials, out, logits,
        k.shape[1], d, indices.shape[1], len(specials), rows,
        q.stride(1), q.stride(0), k.stride(0), scale,
        triton.next_power_of_2(d), 128, num_warps=4)
    return out, logits
