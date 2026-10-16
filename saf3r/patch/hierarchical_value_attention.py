"""Layer-shared hierarchical Value coverage and validated Attention history.

No PAM, fine-cluster kNN, dense fine QK, or cross-layer attention weights.
All selection uses head-mean features; actual multi-head attention is unchanged.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from .value_clustering_gpu import centroid_nearest_gpu
from .value_evolving_attention import token_layout


def sizes(labels, k):
    result = torch.zeros(k, device=labels.device, dtype=torch.long)
    return result.index_add_(0, labels, torch.ones_like(labels))


def allocate_budget(probabilities, budget, capacities):
    """Capacity-constrained Hamilton rounding; redistribute saturated mass.

    Probabilities have shape [query, group], capacities [group]. Each row sums
    exactly to budget. Saturated groups redistribute excess to other groups.
    """
    cap = capacities.expand_as(probabilities)
    remaining = torch.full((len(probabilities), 1), budget, device=cap.device, dtype=torch.long)
    result = torch.zeros_like(cap, dtype=torch.long)
    weights = probabilities.float()
    # Iterations only arise when a group saturates; remaining groups receive
    # proportional mass. This is not independent clamp-and-forget rounding.
    while bool((remaining > 0).any()):
        available = cap-result
        active = available > 0
        p = weights*active
        denom = p.sum(-1, keepdim=True)
        p = torch.where(denom > 0, p/denom.clamp_min(1e-30), active/active.sum(-1, keepdim=True).clamp_min(1))
        target = p*remaining
        base = torch.minimum(target.floor().long(), available)
        result += base
        remaining -= base.sum(-1, keepdim=True)
        available = cap-result
        # Only round when no group was truncated. Otherwise redistribute the
        # excess before rounding, preserving proportional capped allocation.
        saturated = (target.floor().long() > base).any(-1, keepdim=True)
        rank = (target-target.floor()).masked_fill(available == 0, -float('inf')).argsort(dim=-1, descending=True, stable=True)
        order = torch.empty_like(rank).scatter_(1, rank, torch.arange(rank.shape[1], device=rank.device).expand_as(rank))
        extra = (order < remaining) & (available > 0) & ~saturated
        result += extra
        remaining -= extra.sum(-1, keepdim=True)
        if bool(((available.sum(-1, keepdim=True) == 0) & (remaining > 0)).any()):
            raise ValueError("Budget exceeds available group capacity")
    return result


def lightweight_clusters(x, k, iterations=3, block=2048):
    """Deterministic FP32 anchored Lloyd clustering, no N-by-N distances.

    Projection-stratified real tokens initialize centers. Anchors keep all K
    groups nonempty. Distances are computed in bounded token blocks.
    """
    if k == len(x):
        return torch.arange(k, device=x.device)
    # A data-derived direction is less sensitive to feature-coordinate scale
    # than selecting the first feature. No random or pose information used.
    direction = x[-1]-x[0]
    if float(direction.square().sum()) < 1e-12:
        direction = x.var(0).sqrt()
    order = (x@direction).argsort(stable=True)
    anchors = order[torch.arange(k, device=x.device)*len(x)//k]
    centers = x[anchors].clone()
    labels = torch.empty(len(x), device=x.device, dtype=torch.long)
    for _ in range(iterations):
        for start in range(0, len(x), block):
            labels[start:start+block] = torch.cdist(x[start:start+block], centers).argmin(-1)
        labels[anchors] = torch.arange(k, device=x.device)
        count = sizes(labels, k)
        centers.zero_().index_add_(0, labels, x)
        centers /= count[:, None]
    return labels


def sibling_update(x, labels, reps, fine_to_coarse, iterations=2, block=1024, diagnostics=None):
    """Synchronous assignment, source-coarse frozen for ALL local iterations."""
    if diagnostics is not None:
        diagnostics.mark('hierarchy_mapping')
    source = fine_to_coarse[labels]
    nc = int(fine_to_coarse.max())+1
    count = sizes(fine_to_coarse, nc)
    width = int(count.max())
    order = fine_to_coarse.argsort(stable=True)
    offsets = count.cumsum(0)-count
    table = torch.full((nc, width), -1, device=x.device, dtype=torch.long)
    table[fine_to_coarse[order], torch.arange(len(reps), device=x.device)-offsets[fine_to_coarse[order]]] = order
    anchors = reps.clone()
    for _ in range(iterations):
        if diagnostics is not None:
            diagnostics.mark('fine_reassignment')
        new_labels = torch.empty_like(labels)
        for start in range(0, len(x), block):
            candidates = table[source[start:start+block]]
            dist = (x[start:start+block, None]-x[reps[candidates.clamp_min(0)]]).square().sum(-1)
            choice = dist.masked_fill(candidates < 0, float('inf')).argmin(-1)
            new_labels[start:start+block] = candidates.gather(1, choice[:, None]).squeeze(1)
        new_labels[anchors] = torch.arange(len(reps), device=x.device)
        labels = new_labels
        if diagnostics is not None:
            diagnostics.mark('representative_update')
        reps = centroid_nearest_gpu(x, labels, reps)
    return labels, reps


def initial_hierarchy(x, fine_ratio, coarse_size, cluster_iterations, local_iterations):
    """One lightweight global initialization, then local refinement."""
    nf = max(1, round(len(x)*fine_ratio))
    nc = min(nf, math.ceil(len(x)/coarse_size))
    coarse = lightweight_clusters(x, nc, cluster_iterations)
    count = sizes(coarse, nc)
    # Each coarse group gets at least one fine cluster; allocate the rest by
    # occupancy, respecting token capacity. The global fine count is exact.
    quota = 1+allocate_budget((count-1).float()[None], nf-nc, count-1)[0]
    order = coarse.argsort(stable=True)
    token_offsets = count.cumsum(0)-count
    fine_offsets = quota.cumsum(0)-quota
    labels = fine_offsets[coarse[order]] + ((torch.arange(len(x), device=x.device)-token_offsets[coarse[order]])*quota[coarse[order]]//count[coarse[order]])
    z = torch.empty_like(labels).scatter_(0, order, labels)
    reps = torch.full((nf,), len(x), device=x.device, dtype=torch.long)
    reps.scatter_reduce_(0, z, torch.arange(len(x), device=x.device), reduce='amin')
    mapping = torch.repeat_interleave(torch.arange(nc, device=x.device), quota)
    z, reps = sibling_update(x, z, reps, mapping, local_iterations)
    return z, reps


def uniform_hierarchy(x, fine_ratio):
    """Contiguous equal-occupancy groups in frame-major patch-token order."""
    nf = max(1, round(len(x)*fine_ratio))
    labels = torch.arange(len(x), device=x.device)*nf//len(x)
    first = torch.full((nf,), len(x), device=x.device, dtype=torch.long)
    first.scatter_reduce_(0, labels, torch.arange(len(x), device=x.device), reduce='amin')
    return labels, centroid_nearest_gpu(x, labels, first)


def reliability(p):
    """Layer-level C*S after averaging Q/K heads, not a per-head budget."""
    if p.shape[-1] == 1:
        return p.new_zeros(()), p.new_zeros(()), p.new_zeros(())
    entropy = -(p*p.clamp_min(1e-30).log()).sum(-1).mean()
    marginal = p.mean(0)
    norm = math.log(p.shape[-1])
    concentration = (1-entropy/norm).clamp(0, 1)
    specificity = ((-(marginal*marginal.clamp_min(1e-30).log()).sum()-entropy)/norm).clamp(0, 1)
    return concentration*specificity, concentration, specificity


def _group_ranks(mask, group_ids, nc):
    """Rank selected entries within groups; entries must be group-sorted."""
    prefix = mask.long().cumsum(-1)
    starts = torch.zeros(nc, device=mask.device, dtype=torch.long)
    group_count = sizes(group_ids, nc)
    starts[:] = group_count.cumsum(0)-group_count
    before = torch.cat((torch.zeros_like(prefix[:, :1]), prefix), -1)[:, starts]
    return prefix-before[:, group_ids]


def select_edges(p, budget, attention_budget, hierarchy, history=None, diagnostics=None):
    """Shared per-query patch indices and A/V origin, no QK evaluation.

    Quotas use original-token capacity. In the underspecified case where all
    representatives in a group are exhausted, deterministic unused real members
    of that SAME group fill its remaining quota, without any extra fine QK.
    """
    labels, reps, mapping, count = hierarchy
    nf, nc, rows = len(reps), p.shape[-1], len(p)
    if diagnostics is not None:
        diagnostics.mark('budget_allocation')
    if budget > nf:
        raise ValueError("Coverage requires edge budget <= number of fine clusters")
    group_capacity = sizes(mapping[labels], nc)
    quotas = allocate_budget(p, attention_budget, group_capacity)
    chosen = torch.full((rows, budget), -1, device=p.device, dtype=torch.long)
    covered = torch.zeros((rows, nf), device=p.device, dtype=torch.bool)
    cursor = torch.zeros(rows, device=p.device, dtype=torch.long)
    if diagnostics is not None:
        diagnostics.mark('history_lookup_filter_dedup')
    # History is only key indices, not old scores or old final masks.
    if history is not None and history.shape[1]:
        hf = labels[history.long()]
        hg = mapping[hf]
        hc = torch.zeros((rows, nc), device=p.device, dtype=torch.long).scatter_add_(1, hg, torch.ones_like(hg))
        # Overfull groups deduplicate by fine id. A lexicographic sort provides
        # deterministic one-per-fine representatives before coverage ranking.
        token_ids = history.long()
        key = hg*(nf+1)*(len(labels)+1)+hf*(len(labels)+1)+token_ids
        order = key.argsort(dim=-1, stable=True)
        hf, hg, token_ids = [t.gather(1, order) for t in (hf, hg, token_ids)]
        unique = torch.cat((torch.ones_like(hf[:, :1], dtype=torch.bool), hf[:, 1:] != hf[:, :-1]), -1)
        eligible = (hc.gather(1, hg) <= quotas.gather(1, hg)) | unique
        # Group then descending cluster population; no old/current fine logits.
        score = hg*(len(labels)+1)+(len(labels)-count[hf])
        order = score.argsort(dim=-1, stable=True)
        hf, hg, token_ids, eligible = [t.gather(1, order) for t in (hf, hg, token_ids, eligible)]
        prefix = eligible.long().cumsum(-1)
        start = torch.full((rows, nc), history.shape[1], device=p.device, dtype=torch.long)
        positions = torch.arange(history.shape[1], device=p.device).expand_as(hg)
        start.scatter_reduce_(1, hg, positions, reduce='amin')
        prefix0 = torch.cat((torch.zeros_like(prefix[:, :1]), prefix), -1)
        rank = prefix-prefix0.gather(1, start.gather(1, hg))
        keep = eligible & (rank <= quotas.gather(1, hg))
        location = keep.long().cumsum(-1)-1
        r, c = keep.nonzero(as_tuple=True)
        chosen[r, location[r, c]] = token_ids[r, c]
        covered[r, hf[r, c]] = True
        used = torch.zeros_like(quotas).scatter_add_(1, hg, keep.long())
        cursor = keep.sum(-1)
    else:
        used = torch.zeros_like(quotas)
    reused_count = cursor
    if diagnostics is not None:
        diagnostics.mark('exploration')
    # Explore highest-population uncovered fine clusters inside each group.
    # This order is also used to rank by population before the final V fill.
    priority = count.argsort(descending=True, stable=True)
    group_order = priority[mapping[priority].argsort(stable=True)]
    groups = mapping[group_order]
    eligible = ~covered[:, group_order]
    rank = _group_ranks(eligible, groups, nc)
    keep = eligible & (rank <= (quotas-used)[:, groups])
    location = keep.long().cumsum(-1)-1+cursor[:, None]
    r, c = keep.nonzero(as_tuple=True)
    chosen[r, location[r, c]] = reps[group_order[c]]
    covered[r, group_order[c]] = True
    cursor = cursor+keep.sum(-1)
    used += torch.zeros_like(quotas).scatter_add_(1, groups.expand(rows, -1), keep.long())
    # Representatives alone cannot always realize token-capacity quotas.
    # Exhaust representatives first, then fill with unused original members.
    missing = quotas-used
    if bool((missing > 0).any()):
        token_groups = mapping[labels]
        token_order = token_groups.argsort(stable=True)
        offsets = group_capacity.cumsum(0)-group_capacity
        # At most quota chosen tokens can be excluded per group, so scanning
        # min(2*quota, population) members guarantees enough remaining tokens.
        candidate_count = torch.minimum(2*quotas, group_capacity)
        ends = candidate_count.cumsum(-1)
        width = int(ends[:, -1].max())
        slot = torch.arange(width, device=p.device).expand(rows, -1).contiguous()
        candidate_group = torch.searchsorted(ends.contiguous(), slot, right=True).clamp_max(nc-1)
        starts = ends-candidate_count
        within = slot-starts.gather(1, candidate_group)
        position = offsets[candidate_group]+within
        valid = slot < ends[:, -1:]
        candidate = token_order[position.clamp_max(len(labels)-1)]
        already = chosen[:, :attention_budget].sort(-1).values.contiguous()
        found = torch.searchsorted(already, candidate.contiguous()).clamp_max(attention_budget-1)
        valid &= already.gather(1, found) != candidate
        prefix = valid.long().cumsum(-1)
        padded = torch.cat((torch.zeros_like(prefix[:, :1]), prefix), -1)
        group_start = starts.gather(1, candidate_group).clamp_max(width)
        rank = prefix-padded.gather(1, group_start)
        keep_fallback = valid & (rank <= missing.gather(1, candidate_group))
        location = keep_fallback.long().cumsum(-1)-1+cursor[:, None]
        r, c = keep_fallback.nonzero(as_tuple=True)
        chosen[r, location[r, c]] = candidate[r, c]
        covered[r, labels[candidate[r, c]]] = True
    # The Attention prefix must be exact before selecting coverage.
    if diagnostics is not None:
        diagnostics.mark('value_fill')
    eligible = ~covered[:, priority]
    keep = eligible & (eligible.long().cumsum(-1) <= budget-attention_budget)
    location = keep.long().cumsum(-1)-1+attention_budget
    r, c = keep.nonzero(as_tuple=True)
    chosen[r, location[r, c]] = reps[priority[c]]
    if diagnostics is not None:
        diagnostics.mark('mask_assemble')
    origin = torch.arange(budget, device=p.device).expand(rows, -1) < attention_budget
    if diagnostics is not None:
        diagnostics.selected(chosen, reused_count, quotas)
    return chosen, origin


class HierarchicalValueState:
    def __init__(self, retention=.10, fine_ratio=.15, coarse_size=128,
                 sparse_start=1, sparse_end=24, iterations=2, cluster_iterations=3,
                 query_chunk=128, initialization='uniform', diagnostics=None):
        if not 0 < retention <= fine_ratio <= 1 or coarse_size < 1:
            raise ValueError("Require 0 < retention <= fine_ratio <= 1")
        if not 1 <= sparse_start <= sparse_end <= 24:
            raise ValueError("Invalid sparse layer range")
        if min(iterations, cluster_iterations, query_chunk) < 1:
            raise ValueError("Iterations and chunk size must be positive")
        if initialization not in {'uniform', 'lightweight'}:
            raise ValueError('Unknown hierarchy initialization')
        self.initialization = initialization
        self.diagnostics = diagnostics
        self.retention, self.fine_ratio, self.coarse_size = retention, fine_ratio, coarse_size
        self.sparse_start, self.sparse_end = sparse_start, sparse_end
        self.iterations, self.cluster_iterations, self.query_chunk = iterations, cluster_iterations, query_chunk
        self.reset(1, 1, 1)

    def reset(self, frames, height, width):
        self.frames, self.patch_height, self.patch_width = frames, height, width
        self.hierarchy = self.history = self.previous_layer = None
        self.records = []
        self.events = []

    def update(self, values, layer):
        diagnostics = self.diagnostics
        if diagnostics is not None:
            diagnostics.previous_hierarchy = self.hierarchy
        if self.hierarchy is None:
            if diagnostics is not None:
                diagnostics.mark('value_initialization')
            if self.initialization == 'uniform':
                labels, reps = uniform_hierarchy(values, self.fine_ratio)
            else:
                labels, reps = initial_hierarchy(values, self.fine_ratio, self.coarse_size,
                                                self.cluster_iterations, self.iterations)
        else:
            if layer != self.previous_layer+1:
                raise ValueError("Hierarchy must inherit the immediately previous layer")
            labels, reps, mapping, _ = self.hierarchy
            labels, reps = sibling_update(values, labels, reps, mapping, self.iterations,
                                         diagnostics=diagnostics)
        nc = min(len(reps), math.ceil(len(values)/self.coarse_size))
        if diagnostics is not None:
            diagnostics.mark('coarse_grouping')
        mapping = lightweight_clusters(values[reps], nc, self.cluster_iterations)
        count = sizes(labels, len(reps))
        self.hierarchy = labels, reps, mapping, count
        self.previous_layer = layer
        return self.hierarchy


def sparse_reference(q, k, v, indices, specials, scale):
    """Small CPU test oracle, never used for CUDA model inference."""
    keys = torch.cat((indices, specials.expand(len(indices), -1)), -1)
    kk, vv = k[:, keys], v[:, keys]
    logits = (q[:, :, None].float()*kk.float()).sum(-1)*scale
    output = (logits.softmax(-1)[..., None]*vv.float()).sum(-2).to(v.dtype)
    return output, logits[..., :indices.shape[1]]


def validate_history(indices, origin, exact_logits, attention_budget, diagnostics=None):
    """A candidates can fail, V candidates can be promoted; save indices only."""
    winners = exact_logits.mean(0).topk(attention_budget, -1).indices
    history = indices.gather(1, winners).to(torch.int32)
    promoted = (~origin.gather(1, winners)).sum()
    if diagnostics is not None:
        diagnostics.validated(winners)
    return history, promoted


class HierarchicalValueAttention(nn.Module):
    def __init__(self, original, state, layer_idx, special_per_frame=5):
        super().__init__()
        self.original, self.state, self.layer_idx = original, state, layer_idx
        self.special_per_frame = special_per_frame

    def forward(self, x, pos=None):
        if self.training or x.shape[0] != 1:
            raise ValueError("Inference with batch size one required")
        o, state = self.original, self.state
        diagnostics = state.diagnostics
        if diagnostics is not None:
            diagnostics.begin_layer(self.layer_idx, state.history)
            diagnostics.mark('qkv')
        b, n, channels = x.shape
        if n != state.frames*(state.patch_height*state.patch_width+self.special_per_frame):
            raise ValueError("Sequence geometry mismatch")
        event = None
        if x.is_cuda:
            event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            event[0].record()
        qkv = o.qkv(x).reshape(b, n, 3, o.num_heads, o.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        patches, specials = token_layout(state.frames, state.patch_height*state.patch_width, self.special_per_frame, x.device)
        uniform_start = state.hierarchy is None and state.initialization == 'uniform'
        if diagnostics is not None:
            diagnostics.mark('mean_value')
        with torch.autocast(device_type=x.device.type, enabled=False):
            mean_v = v[0, :, patches].float().mean(0)
            hierarchy = state.update(mean_v, self.layer_idx+1)
        if diagnostics is not None:
            diagnostics.mark('normalize_rope')
        q, k = o.q_norm(q), o.k_norm(k)
        if o.rope is not None:
            q, k = o.rope(q, pos), o.rope(k, pos)
        q, k, v = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
        # Total 10% includes mandatory special keys, not 10% plus specials.
        total_budget = max(1, round(state.retention*n))
        budget = total_budget-len(specials)
        if budget < 1:
            raise ValueError("Budget cannot accommodate mandatory special keys")
        labels, reps, mapping, count = hierarchy
        nc = int(mapping.max())+1
        token_groups = mapping[labels]
        if uniform_start:
            # Explicit Value-only bootstrap; coarse groups still seed layer 1.
            alpha = concentration = specificity = mean_v.new_zeros(())
            attention_budget = 0
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                if diagnostics is not None:
                    diagnostics.mark('pooled_key')
                pooled = torch.zeros((nc, k.shape[-1]), device=x.device, dtype=torch.float32)
                pooled.index_add_(0, token_groups, k[:, patches].float().mean(0))
                population = sizes(token_groups, nc)
                pooled /= population[:, None]
                mean_q = q.float().mean(0)
                if diagnostics is not None:
                    diagnostics.mark('coarse_qk')
                coarse_logits = mean_q@pooled.T*o.scale+population.float().log()
                if diagnostics is not None:
                    diagnostics.mark('coarse_softmax')
                coarse_p = coarse_logits.softmax(-1)
                del coarse_logits
                if diagnostics is not None:
                    diagnostics.mark('coarse_statistics')
                alpha, concentration, specificity = reliability(coarse_p)
                attention_budget = int((alpha*budget).round())
        if diagnostics is not None:
            diagnostics.context(hierarchy, patches, specials, None if uniform_start else coarse_p,
                                budget, attention_budget, uniform_start)
            diagnostics.mark('mask_assemble')
        history = torch.empty((n, attention_budget), device=x.device, dtype=torch.int32)
        promoted = torch.zeros((), device=x.device, dtype=torch.long)
        if attention_budget == 0:
            # Exact mathematical degeneration of the proposed algorithm:
            # every query selects the same largest-population V clusters.
            # No history is retained, hence no fine logits need exporting.
            if uniform_start:
                # Equal coverage of the entire sequence, not the first K fine
                # clusters when equal-size clusters tie in coverage priority.
                priority = (torch.arange(budget, device=x.device)*len(reps)//budget)
            else:
                priority = count.argsort(descending=True, stable=True)[:budget]
            selected = torch.cat((patches[reps[priority]], specials))
            if diagnostics is not None:
                diagnostics.uniform_selected(reps[priority], n)
                diagnostics.mark('sparse_qk_softmax_av_fused')
            attended = F.scaled_dot_product_attention(q[None], k[:, selected][None], v[:, selected][None],
                                                       dropout_p=0., scale=o.scale)[0]
        else:
            attended = torch.empty_like(q)
        for start in range(0, n if attention_budget else 0, state.query_chunk):
            end = min(n, start+state.query_chunk)
            previous = None if state.history is None else state.history[start:end]
            indices, origin = select_edges(coarse_p[start:end], budget, attention_budget, hierarchy, previous,
                                          diagnostics=diagnostics)
            global_indices = patches[indices].to(torch.int32)
            if diagnostics is not None:
                diagnostics.mark('sparse_qk_softmax_av_fused')
            if x.is_cuda:
                from saf3r.triton.hierarchy_attention import indexed_attention_with_logits
                out, logits = indexed_attention_with_logits(q[:, start:end], k, v, global_indices, specials, o.scale)
            else:
                out, logits = sparse_reference(q[:, start:end], k, v, global_indices.long(), specials, o.scale)
            attended[:, start:end] = out
            if diagnostics is not None:
                diagnostics.mark('validation')
            # Validation reuses the logits returned by that one sparse QK.
            # Only indices survive; no per-head state or old weights retained.
            validated, promotion = validate_history(indices, origin, logits, attention_budget,
                                                    diagnostics=diagnostics)
            history[start:end] = validated
            promoted += promotion
        state.history = history
        record = {"layer": self.layer_idx+1, "layer_zero_based": self.layer_idx,
                  "uniform_value_only_start": uniform_start,
                  "patch_tokens": len(patches), "fine_clusters": len(reps),
                  "coarse_groups": nc, "total_keys": total_budget, "patch_keys": budget,
                  "special_keys": len(specials), "connection_retention": total_budget/n,
                  "attention_budget": attention_budget, "value_budget": budget-attention_budget,
                  "alpha": alpha.detach(), "concentration": concentration.detach(),
                  "specificity": specificity.detach(), "promoted_value_edges": promoted.detach()}
        record['execution'] = 'shared_sdpa_zero_attention_budget' if attention_budget == 0 else 'indexed_exact_logits'
        state.records.append(record)
        if diagnostics is not None:
            diagnostics.mark('projection')
        result = o.proj_drop(o.proj(attended.transpose(0, 1).reshape(b, n, channels)))
        if event is not None:
            event[1].record()
            state.events.append(event)
        if diagnostics is not None:
            diagnostics.finish_layer(record, q, k, v, attended, state.history, o.scale)
        return result


def install_hierarchical_value_attention(model, **kwargs):
    if hasattr(model, 'hierarchical_value_state'):
        raise ValueError("Hierarchy attention already installed")
    state = HierarchicalValueState(**kwargs)
    for layer in range(state.sparse_start-1, state.sparse_end):
        block = model.aggregator.global_blocks[layer]
        block.attn = HierarchicalValueAttention(block.attn, state, layer).eval()
    def reset(module, inputs):
        images = inputs[0]
        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError("Expected batch-one image sequence")
        state.reset(images.shape[1], images.shape[-2]//14, images.shape[-1]//14)
    state.reset_hook_handle = model.aggregator.register_forward_pre_hook(reset)
    model.hierarchical_value_state = state
    return state
