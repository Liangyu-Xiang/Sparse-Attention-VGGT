"""Fixed-budget, current-Value coarse sensing; one policy/mask per layer.

No head routing, query-shared blocks, dense probes, or retention fallback.
Coarse groups are ephemeral. Only fine labels/reps and validated indices recur.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .hierarchical_value_attention import sizes, lightweight_clusters, sibling_update, allocate_budget
from .value_clustering_gpu import centroid_nearest_gpu, stable_cluster_mean
from .value_evolving_attention import token_layout


def sample_queries(n, fraction, device):
    if not 0 < fraction <= 1:
        raise ValueError('sampling fraction must lie in (0,1]')
    m = max(1, math.ceil(n*fraction))
    return ((torch.arange(m, device=device, dtype=torch.float64)+.5)*n/m).long()


def sharpen_reliability(score, gamma=1.):
    """Monotone odds sharpening, distinct from coarse-logit temperature."""
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('reliability gamma must be finite and positive')
    if not 0 <= score <= 1:
        raise ValueError('reliability score must lie in [0,1]')
    if gamma == 1. or score in (0., 1.):
        return score
    odds = gamma*(math.log(score)-math.log1p(-score))
    if odds >= 0:
        return 1/(1+math.exp(-odds))
    exp_odds = math.exp(odds)
    return exp_odds/(1+exp_odds)


def fixed_branch_alpha(layer, schedule):
    """Zero-based global layers; no dense-prefix or exact-TopK substitution."""
    if schedule == 'adaptive':
        return None
    if schedule not in ('value_attention_value', 'value_dense_value') or not 0 <= layer < 24:
        raise ValueError('invalid fixed branch schedule/layer')
    return float(schedule == 'value_attention_value' and 10 <= layer <= 16)


def branch_statistics(p, dominance=.9, reliability_gamma=1., reliability_metric='harmonic_cd'):
    if reliability_metric not in ('harmonic_cd', 'concentration'):
        raise ValueError('unknown reliability metric')
    if dominance is not None and not .5 < dominance <= 1:
        raise ValueError('dominance threshold must lie in (.5,1]')
    if p.shape[-1] == 1:
        c = d = p.new_zeros(())
    else:
        c = (1+(p*p.clamp_min(1e-30).log()).sum(-1).mean()/math.log(p.shape[-1])).clamp(0, 1)
        d = (1-F.cosine_similarity(p, p.mean(0)[None], dim=-1).mean()).clamp(0, 1)
    # Keep D as an observational diagnostic; it does NOT enter C-only decisions.
    alpha = c if reliability_metric == 'concentration' else 2*c*d/(c+d+1e-12)
    raw = float(alpha)
    mapped = sharpen_reliability(raw, reliability_gamma)
    executed = mapped
    if dominance is not None:
        executed = 0. if mapped <= 1-dominance else 1. if mapped >= dominance else mapped
    return dict(C=float(c), D=float(d), R=1-float(d), alpha_raw=raw,
                alpha_mapped=mapped, reliability_gamma=reliability_gamma,
                reliability_metric=reliability_metric, alpha=executed)


def coarse_probabilities(queries, pooled, scale):
    """Canonical GEMM row tiles, independent of mask/AV memory chunk size.

    FP32 GEMM shapes can change reduction order. Tiny coarse-logit differences
    can break many quota ties and then alter subsequent fine/history states.
    Fixed arithmetic tiles do NOT share routing decisions or query masks.
    """
    return torch.cat([(queries[i:i+128]@pooled.T*scale).softmax(-1)
        for i in range(0, len(queries), 128)], 0)


def spatial_hierarchy(frames, height, width, fine_ratio, coarse_size, device):
    """Approximately square 2D cells; fixed spatial-center representatives."""
    fh = max(1, min(height, round(height*math.sqrt(fine_ratio))))
    fw = max(1, min(width, round(height*width*fine_ratio/fh)))
    target = max(1, round(height*width/coarse_size))
    ch = max(1, min(fh, round(math.sqrt(target*height/width))))
    cw = max(1, min(fw, round(target/ch)))
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    fy, fx = yy*fh//height, xx*fw//width
    labels = (fy*fw+fx)[None]+torch.arange(frames, device=device)[:, None, None]*fh*fw
    ry = ((torch.arange(fh, device=device)+.5)*height/fh).long()
    rx = ((torch.arange(fw, device=device)+.5)*width/fw).long()
    reps = (ry[:, None]*width+rx)[None]+torch.arange(frames, device=device)[:, None, None]*height*width
    gy = torch.arange(fh, device=device)*ch//fh
    gx = torch.arange(fw, device=device)*cw//fw
    mapping = (gy[:, None]*cw+gx)[None]+torch.arange(frames, device=device)[:, None, None]*ch*cw
    return labels.flatten(), reps.flatten(), mapping.flatten()


def current_groups(values, reps, count, coarse_size, iterations=2):
    nc = min(len(reps), max(1, math.ceil(len(values)/coarse_size)))
    return lightweight_clusters(values[reps], nc, iterations, deterministic=True)


def choose_edges(p, budget, ka, hierarchy, history=None):
    """Capacity-capped Hamilton; reuse distinct validated TOKEN edges.

    In group c capacity = fine_count + history_tokens - covered_history_fines.
    Thus multiple validated tokens in one fine cluster can survive; exploration
    still uses only representatives of uncovered fine clusters. No arbitrary
    unvalidated member tokens are needed to fill quotas.
    """
    labels, reps, groups, count = hierarchy
    rows, nf, nc = len(p), len(reps), p.shape[1]
    if not 0 <= ka <= budget <= nf:
        raise ValueError('budget must fit distinct fine representatives')
    capacity = sizes(groups, nc).expand(rows, -1).clone()
    if history is not None and history.shape[1] and ka:
        h = history.long()
        token_order = h.argsort(dim=-1, stable=True)
        hs = h.gather(1, token_order)
        unique = torch.cat((torch.ones_like(hs[:, :1], dtype=torch.bool), hs[:, 1:] != hs[:, :-1]), 1)
        eligible = torch.zeros_like(unique).scatter_(1, token_order, unique)
        fine = labels[h]
        hg = groups[fine]
        fine_order = fine.argsort(dim=-1, stable=True)
        fs = fine.gather(1, fine_order)
        distinct_fines = torch.cat((torch.ones_like(fs[:, :1], dtype=torch.bool), fs[:, 1:] != fs[:, :-1]), 1)
        unique_fine_groups = groups[fs]
        hist_counts = torch.zeros_like(capacity).scatter_add_(1, hg, eligible.long())
        fine_counts = torch.zeros_like(capacity).scatter_add_(1, unique_fine_groups, distinct_fines.long())
        capacity += hist_counts-fine_counts
    quotas = allocate_budget(p, ka, capacity)
    chosen = torch.full((rows, budget), -1, device=p.device, dtype=torch.long)
    covered = torch.zeros((rows, nf), device=p.device, dtype=torch.bool)
    used = torch.zeros_like(quotas)
    cursor = torch.zeros(rows, device=p.device, dtype=torch.long)
    # Process history in saved descending validation rank. Vectorized across
    # queries; no old scores or old coarse labels are consulted.
    if history is not None and history.shape[1] and ka:
        # Sort by group, maintaining validation order within each group.
        order = hg.argsort(dim=-1, stable=True)
        gs, es = hg.gather(1, order), eligible.gather(1, order)
        prefix = es.long().cumsum(-1)
        first = torch.full((rows, nc), h.shape[1], device=p.device, dtype=torch.long)
        positions = torch.arange(h.shape[1], device=p.device).expand(rows, -1)
        first.scatter_reduce_(1, gs, positions, reduce='amin')
        padded = F.pad(prefix, (1, 0))
        rank = prefix-padded.gather(1, first.gather(1, gs))
        keep = es & (rank <= quotas.gather(1, gs))
        loc = keep.long().cumsum(-1)-1
        rr, cc = keep.nonzero(as_tuple=True)
        tokens = h.gather(1, order)
        chosen[rr, loc[rr, cc]] = tokens[rr, cc]
        covered[rr, labels[tokens[rr, cc]]] = True
        used.scatter_add_(1, gs, keep.long())
        cursor = keep.sum(-1)
    reused = cursor.sum()
    priority = count.argsort(descending=True, stable=True)
    ordered = priority[groups[priority].argsort(stable=True)]
    group_ids = groups[ordered]
    eligible = ~covered[:, ordered]
    prefix = eligible.long().cumsum(-1)
    offsets = sizes(group_ids, nc).cumsum(0)-sizes(group_ids, nc)
    before = F.pad(prefix, (1, 0))[:, offsets]
    rank = prefix-before[:, group_ids]
    keep = eligible & (rank <= (quotas-used)[:, group_ids])
    loc = keep.long().cumsum(-1)-1+cursor[:, None]
    rr, cc = keep.nonzero(as_tuple=True)
    chosen[rr, loc[rr, cc]] = reps[ordered[cc]]
    covered[rr, ordered[cc]] = True
    if budget > ka:
        eligible = ~covered[:, priority]
        keep = eligible & (eligible.long().cumsum(-1) <= budget-ka)
        loc = keep.long().cumsum(-1)-1+ka
        rr, cc = keep.nonzero(as_tuple=True)
        chosen[rr, loc[rr, cc]] = reps[priority[cc]]
    return chosen, reused


class LayerSharedAVState:
    def __init__(self, retention=.1, fine_ratio=.15, coarse_size=128,
                 sampling=.01, dominance=.9, logit_scale=1., query_chunk=128,
                 cluster_iterations=2, sensing_only=False, capture_logits=False,
                 reliability_gamma=1., reliability_metric='harmonic_cd',
                 branch_schedule='adaptive'):
        if not 0 < retention <= fine_ratio <= 1 or coarse_size < 1 or logit_scale <= 0:
            raise ValueError('invalid budget, hierarchy or scale')
        if query_chunk < 128 or query_chunk % 128:
            raise ValueError('query_chunk must be a positive multiple of canonical arithmetic tile128')
        self.retention, self.fine_ratio, self.coarse_size = retention, fine_ratio, coarse_size
        self.sampling, self.dominance, self.logit_scale = sampling, dominance, logit_scale
        self.query_chunk, self.cluster_iterations = query_chunk, cluster_iterations
        self.sensing_only, self.capture_logits = sensing_only, capture_logits
        sharpen_reliability(.5, reliability_gamma)  # validate before inference
        self.reliability_gamma = reliability_gamma
        if reliability_metric not in ('harmonic_cd', 'concentration'):
            raise ValueError('unknown reliability metric')
        self.reliability_metric = reliability_metric
        fixed_branch_alpha(0, branch_schedule)
        self.branch_schedule = branch_schedule
        self.reset(1, 1, 1)

    def reset(self, frames, height, width):
        self.frames, self.height, self.width = frames, height, width
        self.fine = self.history = None
        self.previous_layer = -1
        self.records, self.sampled_logits = [], []

    def update(self, values, layer):
        if layer != self.previous_layer+1:
            raise ValueError('fine state must come from immediately previous layer')
        if layer == 0:
            labels, reps, groups = spatial_hierarchy(self.frames, self.height, self.width,
                self.fine_ratio, self.coarse_size, values.device)
        else:
            labels, reps = self.fine
            reps = centroid_nearest_gpu(values, labels, reps, deterministic=True)
            temporary = current_groups(values, reps, sizes(labels, len(reps)), self.coarse_size, self.cluster_iterations)
            labels, reps = sibling_update(values, labels, reps, temporary, iterations=1, deterministic=True)
            groups = current_groups(values, reps, sizes(labels, len(reps)), self.coarse_size, self.cluster_iterations)
        self.fine = labels, reps  # deliberately NOT coarse mapping
        self.previous_layer = layer
        return labels, reps, groups, sizes(labels, len(reps))


class LayerSharedAVAttention(nn.Module):
    def __init__(self, original, state, layer):
        super().__init__()
        self.original, self.state, self.layer = original, state, layer

    def forward(self, x, pos=None):
        if self.training or x.shape[0] != 1:
            raise ValueError('batch-one inference required')
        o, s = self.original, self.state
        b, n, c = x.shape
        q, k, v = o.qkv(x).reshape(b, n, 3, o.num_heads, o.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        patches, specials = token_layout(s.frames, s.height*s.width, 5, x.device)
        marks = []
        def mark(name):
            e = torch.cuda.Event(enable_timing=True) if x.is_cuda else None
            if e is not None: e.record()
            marks.append((name, e))
        mark('hierarchy')
        value_source_layer = s.previous_layer
        with torch.autocast(x.device.type, enabled=False):
            hierarchy = s.update(v[0, :, patches].float().mean(0), self.layer)
        labels, reps, groups, count = hierarchy
        q, k = o.q_norm(q), o.k_norm(k)
        if o.rope is not None: q, k = o.rope(q, pos), o.rope(k, pos)
        q, k, v = q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
        mark('sampled_sensing')
        nc = int(groups.max())+1
        with torch.autocast(x.device.type, enabled=False):
            mean_q = q.float().mean(0)
            pooled = stable_cluster_mean(k[:, patches].float().mean(0), groups[labels], nc)
            sample = sample_queries(n, s.sampling, x.device)
            logits = (mean_q[sample]@pooled.T)*o.scale
            if s.capture_logits: s.sampled_logits.append(logits.detach().cpu())
            stats = branch_statistics((logits*s.logit_scale).softmax(-1), s.dominance,
                                      s.reliability_gamma, s.reliability_metric)
        prescribed = fixed_branch_alpha(self.layer, s.branch_schedule)
        stats['branch_schedule'] = s.branch_schedule
        stats['diagnostic_alpha'] = stats['alpha']
        if prescribed is not None:
            # C/D and their mapping remain observational, never control this arm.
            stats['alpha'] = prescribed
        # Mandatory special keys count INSIDE the final 10% edge budget.
        total = math.floor(s.retention*n)
        budget = total-len(specials)
        if not 0 < budget <= len(reps):
            raise ValueError('budget incompatible with mandatory specials/fine support')
        ka = round(stats['alpha']*budget)
        dense_middle = s.branch_schedule == 'value_dense_value' and 10 <= self.layer <= 16
        mark('routing_and_sparse_attention')
        reused = promoted = 0
        if s.sensing_only or dense_middle:
            attended = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=o.scale)[0]
            # Fine Value state has already advanced above, even on Dense layers.
            # Do not create or inherit Attention-branch routing history here.
            s.history = None
        elif ka == 0:
            priority = count.argsort(descending=True, stable=True)[:budget]
            if self.layer == 0:
                # Spatial cold start: distribute equal-tie support over frames.
                priority = torch.arange(budget, device=x.device)*len(reps)//budget
            keys = torch.cat((patches[reps[priority]], specials))
            attended = F.scaled_dot_product_attention(q[None], k[:, keys][None], v[:, keys][None], scale=o.scale)[0]
            s.history = None
        else:
            from saf3r.triton.hierarchy_attention import indexed_attention_with_logits
            attended = torch.empty_like(q)
            history = torch.empty((n, ka), device=x.device, dtype=torch.int32)
            for start in range(0, n, s.query_chunk):
                end = min(n, start+s.query_chunk)
                with torch.autocast(x.device.type, enabled=False):
                    p = coarse_probabilities(mean_q[start:end], pooled, o.scale*s.logit_scale)
                old = None if s.history is None else s.history[start:end]
                indices, nr = choose_edges(p, budget, ka, hierarchy, old)
                out, exact, lse = indexed_attention_with_logits(q[:, start:end], k, v,
                    patches[indices].to(torch.int32), specials, o.scale, return_lse=True)
                attended[:, start:end] = out
                # Normalized per-head weights, NOT mean logits. LSE includes
                # mandatory specials; QK is evaluated only once for these edges.
                scores = (exact-lse[..., None]).exp().mean(0)
                winners = scores.topk(ka, -1).indices
                history[start:end] = indices.gather(1, winners).to(torch.int32)
                reused += int(nr)
                promoted += int((winners >= ka).sum())
            s.history = history
        mark('end')
        dense_output = s.sensing_only or dense_middle
        s.records.append(dict(layer=self.layer, **stats,
            execution_mode='dense' if dense_output else 'value' if ka == 0 else 'sparse_av',
            dense_middle=dense_middle, value_updated=True,
            value_source_layer=value_source_layer, value_state_layer=s.previous_layer,
            attention_keys=0 if dense_output else ka, value_keys=0 if dense_output else budget-ka,
            dense_keys=n if dense_output else 0,
            total_keys=n if dense_output else total, special_keys=len(specials),
            actual_density=1. if dense_output else total/n,
            fine_clusters=len(reps), coarse_groups=nc, sampled_queries=len(sample),
            full_query_coarse=bool(ka and not dense_output), sensing_only=s.sensing_only,
            reused_edges=reused, promoted_value_edges=promoted,
            logit_scale=s.logit_scale, sampling=s.sampling, dominance=s.dominance,
            history_unit='token_edges_v3_canonical_qk',
            _events=marks))
        return o.proj_drop(o.proj(attended.transpose(0, 1).reshape(b, n, c)))


def finish_records(state):
    result = []
    for record in state.records:
        row = {k: v for k, v in record.items() if k != '_events'}
        events = record['_events']
        row['stage_seconds'] = {a: ea.elapsed_time(eb)/1000 for (a, ea), (_, eb) in zip(events, events[1:]) if ea is not None}
        result.append(row)
    return result


def install_layer_shared_av(model, **kwargs):
    state = LayerSharedAVState(**kwargs)
    for layer, block in enumerate(model.aggregator.global_blocks):
        block.attn = LayerSharedAVAttention(block.attn, state, layer).eval()
    def reset(module, inputs):
        images = inputs[0]
        state.reset(images.shape[1], images.shape[-2]//14, images.shape[-1]//14)
    state.reset_hook_handle = model.aggregator.register_forward_pre_hook(reset)
    model.layer_shared_av_state = state
    return state
