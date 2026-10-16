"""Head-aware audited budgets, Key-space sensing, shared Value hierarchy.

Query/key blocks make masks executable by tensor cores. This is an explicit
block-quantized redesign, not the old per-query exact-mask implementation.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from .coarse_sensing import reliability_statistics
from .hierarchical_value_attention import HierarchicalValueState
from .value_evolving_attention import token_layout


def block_mean(x, block):
    n = x.shape[-2]
    count = math.ceil(n/block)
    pad = count*block-n
    result = F.pad(x, (0, 0, 0, pad)).reshape(*x.shape[:-2], count, block, x.shape[-1]).sum(-2)
    populations = torch.full((count,), block, device=x.device, dtype=torch.long)
    populations[-1] -= pad
    return result/populations.to(result.dtype)[..., None], populations


def coverage_order(hierarchy, num_blocks, key_block, uniform=False):
    labels, reps, mapping, count = hierarchy
    if uniform:
        # Bit-reversal traversal gives a nested, evenly spread block coverage
        # for every budget, rather than taking a contiguous prefix.
        ids = torch.arange(num_blocks, device=reps.device)
        bits = math.ceil(math.log2(max(num_blocks, 2)))
        rev = torch.zeros_like(ids)
        for b in range(bits):
            rev |= ((ids >> b) & 1) << (bits-b-1)
        return rev.argsort(stable=True)
    weight = torch.zeros(num_blocks, device=reps.device, dtype=torch.float32)
    weight.index_add_(0, reps//key_block, count.float())
    return weight.argsort(descending=True, stable=True)


def select_block_mask(scores, value_order, count, attention_count, history=None):
    """[rows,key blocks], integer block budgets; no NxN token mask.

    At most half of A slots reuse history; those blocks must currently fall
    within the top-2A relevance region. The rest explores current relevance.
    All other slots cover previously uncovered Value-oriented blocks.
    """
    rows, blocks = scores.shape
    attention_count = min(count, max(0, attention_count))
    if attention_count == 0:
        ids = value_order[:count].sort().values[None].expand(rows,-1)
        return ids,torch.zeros((rows,count),device=scores.device,dtype=torch.uint8)
    if attention_count == count and (history is None or not history.shape[-1]):
        ids = scores.topk(count,dim=-1).indices.sort(dim=-1).values
        return ids,torch.full((rows,count),2,device=scores.device,dtype=torch.uint8)
    a = torch.zeros_like(scores, dtype=torch.bool)
    reused = torch.zeros_like(a)
    if attention_count:
        if history is not None and history.shape[-1] and attention_count >= 2:
            eligible_rank = min(blocks, 2*attention_count)
            cutoff = scores.topk(eligible_rank, -1).values[:, -1:]
            history = history.long()
            valid = history >= 0
            old_scores = scores.gather(1, history.clamp_min(0)).masked_fill(~valid, -float('inf'))
            take = min(attention_count//2, history.shape[-1])
            slots = old_scores.topk(take, -1).indices
            ids = history.gather(1, slots)
            keep = valid.gather(1, slots) & (old_scores.gather(1, slots) >= cutoff)
            # Fixed-shape scatter avoids nonzero's device-to-host shape sync.
            # Torch 2.3 CUDA scatter-reduce does not support Bool.
            reused_bytes = torch.zeros_like(reused,dtype=torch.uint8)
            reused_bytes.scatter_reduce_(1,ids.clamp_min(0),keep.to(torch.uint8),
                reduce='amax',include_self=True)
            reused = reused_bytes.bool()
            a |= reused
        remaining = attention_count-a.sum(-1)
        current = scores.masked_fill(a, -float('inf')).topk(attention_count, -1).indices
        keep = torch.arange(attention_count, device=scores.device)[None] < remaining[:, None]
        a.scatter_(1,current,keep | a.gather(1,current))
    v = ~a[:, value_order]
    fill = v & (v.long().cumsum(-1) <= count-attention_count)
    selected = a.clone()
    selected[:,value_order] |= fill
    # Exactly count selected entries per row: recover sorted IDs without a
    # dynamic nonzero result or any token-wise Python processing.
    ids = torch.where(selected,torch.arange(blocks,device=scores.device),blocks).topk(
        count,dim=-1,largest=False,sorted=True).indices
    origin = torch.where(reused.gather(1, ids), 1,
                         torch.where(a.gather(1, ids), 2, 0)).to(torch.uint8)
    return ids, origin


def audit_contributions(q, k_patch, v_patch, k_special, v_special, probe_ids, block, scale):
    """Dense ONLY for a small deterministic query probe, not the full layer.

    Pool exact dense weighted V and mass by key block. Summing selected block
    contributions and renormalizing evaluates a mask without repeating fine QK.
    """
    qp = q[:, probe_ids].float()
    keys = torch.cat((k_patch, k_special), 1).float()
    logits = qp@keys.transpose(1, 2)*scale
    p = logits.softmax(-1)
    npatch = len(k_patch[0])
    nb = math.ceil(npatch/block)
    padded = nb*block
    pp = F.pad(p[..., :npatch], (0, padded-npatch))
    vp = F.pad(v_patch.float(), (0, 0, 0, padded-npatch))
    h, nq, _ = pp.shape
    bp = pp.reshape(h, nq, nb, block).permute(0, 2, 1, 3).contiguous()
    bv = vp.reshape(h, nb, block, vp.shape[-1])
    weighted = torch.matmul(bp, bv).permute(0, 2, 1, 3).contiguous()
    mass = pp.reshape(h, nq, nb, block).sum(-1)
    special_mass = p[..., npatch:].sum(-1)
    special_weighted = p[..., npatch:]@v_special.float()
    dense = weighted.sum(2)+special_weighted
    return mass, weighted, special_mass, special_weighted, dense


def audit_error(contrib, head, ids):
    mass, weighted, sm, sw, reference = contrib
    selected_mass = mass[head].gather(1, ids).sum(-1)+sm[head]
    selected_weighted = weighted[head].gather(1, ids[..., None].expand(-1, -1, weighted.shape[-1])).sum(1)+sw[head]
    out = selected_weighted/selected_mass[:, None].clamp_min(1e-20)
    error = (out-reference[head]).square().sum(-1)
    energy = reference[head].square().sum(-1).clamp_min(1e-12)
    return error.sum()/energy.sum(), torch.quantile(error/energy, .9)


def ranked_mask(ranks, order, count, attention_counts, history_mask=None):
    """Vectorized equivalent of select_block_mask for audit heads/queries."""
    a = attention_counts[:, None, None]
    reused = torch.zeros_like(ranks, dtype=torch.bool)
    if history_mask is not None:
        eligible = history_mask & (ranks < 2*a)
        # Rank eligible old blocks by current relevance, never old weights.
        sorted_ids = ranks.argsort(-1)
        old_sorted = eligible.gather(-1, sorted_ids)
        keep = old_sorted & (old_sorted.long().cumsum(-1) <= a//2)
        reused.scatter_(-1, sorted_ids, keep)
    remaining = a-reused.sum(-1, keepdim=True)
    sorted_ids = ranks.argsort(-1)
    available = (~reused).gather(-1, sorted_ids)
    fill = available & (available.long().cumsum(-1) <= remaining)
    selected = reused.clone().scatter_(-1, sorted_ids, fill | reused.gather(-1, sorted_ids))
    eligible_v = (~selected)[..., order]
    fill_v = eligible_v & (eligible_v.long().cumsum(-1) <= count-a)
    selected[..., order] |= fill_v
    return selected


def audit_mask_error(contrib, selected):
    mass, weighted, sm, sw, reference = contrib
    h, nq, nb = mass.shape
    selected_float = selected.float()
    numerator = torch.bmm(selected_float.reshape(h*nq, 1, nb),
        weighted.reshape(h*nq, nb, -1)).reshape(h, nq, -1)+sw
    denominator = (mass*selected_float).sum(-1)+sm
    out = numerator/denominator[..., None].clamp_min(1e-20)
    error = (out-reference).square().sum(-1)
    energy = reference.square().sum(-1).clamp_min(1e-12)
    return error.sum(-1)/energy.sum(-1), torch.quantile(error/energy, .9, dim=-1)


def audited_decisions(state, routing, value_order, stats, contrib, audit_blocks,
                      matching, enrichment, n, nspecial, nb, bootstrap):
    """Select all heads together; at most one host synchronization per budget."""
    scores = routing[:, audit_blocks]
    h, nq, _ = scores.shape
    rank_order = scores.argsort(dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(rank_order).scatter_(-1, rank_order,
        torch.arange(nb, device=scores.device).expand_as(rank_order))
    history_mask = torch.zeros_like(scores, dtype=torch.bool)
    if not bootstrap and state.history is not None:
        for head in range(h):
            old = state.history[matching[head]]
            if enrichment[head] > 1. and old is not None and old.shape[-1]:
                ids = old[audit_blocks].long()
                history_mask[head].scatter_(-1, ids, True)
    done = torch.zeros(h, device=scores.device, dtype=torch.bool)
    # rho, blocks, a_blocks, alpha, use_history, nmse, p90, fallback, passed
    chosen = torch.zeros((h, 9), device=scores.device)
    alpha_options = [torch.zeros(h, device=scores.device)]
    if not bootstrap:
        alpha_options += [stats['concentration'], torch.ones(h, device=scores.device)]
    for rho in state.candidates:
        blocks = min(nb, max(1, math.ceil((round(rho*n)-nspecial)/state.key_block)))
        if rho == 1. or blocks == nb:
            fallback = torch.tensor([1.,nb,0.,0.,0.,0.,0.,1.,1.], device=scores.device)
            chosen = torch.where(done[:, None], chosen, fallback)
            break
        errors, tails, options = [], [], []
        for alpha in alpha_options:
            a = torch.round(alpha*blocks).long()
            for use_old in ([False, True] if not bootstrap and state.history is not None else [False]):
                selected = ranked_mask(ranks, value_order, blocks, a,
                    history_mask if use_old else None)
                error, p90 = audit_mask_error(contrib, selected)
                errors.append(error); tails.append(p90); options.append((alpha,a,use_old))
        errors, tails = torch.stack(errors), torch.stack(tails)
        passed = (errors <= state.audit_nmse) & (tails <= state.audit_p90)
        acceptable = passed if state.audit_guard else torch.ones_like(passed)
        winners = errors.masked_fill(~acceptable, float('inf')).argmin(0)
        hh = torch.arange(h, device=scores.device)
        error, tail = errors[winners, hh], tails[winners, hh]
        alpha = torch.stack([o[0] for o in options])[winners, hh]
        a = torch.stack([o[1] for o in options])[winners, hh]
        use_old = torch.tensor([o[2] for o in options], device=scores.device)[winners]
        row = torch.stack((torch.full_like(error,rho),torch.full_like(error,blocks),
            a,alpha,use_old,error,tail,torch.zeros_like(error),passed[winners,hh]), -1)
        found = acceptable.any(0)
        chosen = torch.where((~done & found)[:,None],row,chosen)
        done |= found
        if bool(done.all()):
            break
    return [dict(rho=r[0],blocks=int(r[1]),a_blocks=int(r[2]),alpha=r[3],
                 use_history=bool(r[4]),audit_nmse=r[5],audit_p90=r[6],
                 dense_fallback=bool(r[7]),audit_passed=bool(r[8]))
            for r in chosen.detach().cpu().tolist()]


def match_history(scores, histories, query_blocks):
    """Match current heads to old validated indices using CURRENT relevance.

    No old attention weights or unvalidated masks are stored. Weak enrichment
    relative to uniform does not authorize reuse.
    """
    h, nq, nb = scores.shape
    if histories is None:
        return list(range(h)), [0.]*h
    cross = torch.zeros((h, h), device=scores.device)
    for previous, history in enumerate(histories):
        if history is None or not history.shape[-1]:
            continue
        ix = history[query_blocks].long()
        valid = ix >= 0
        probability = scores.gather(-1, ix.clamp_min(0)[None].expand(h, -1, -1))
        cross[:, previous] = ((probability*valid[None]).sum(-1)/valid.sum(-1).clamp_min(1)[None]*nb).mean(-1)
    from scipy.optimize import linear_sum_assignment
    current, previous = linear_sum_assignment(-cross.detach().cpu().numpy())
    assignment = list(range(h))
    for c, p in zip(current, previous):
        assignment[int(c)] = int(p)
    enrichment = [float(cross[c, assignment[c]]) for c in range(h)]
    return assignment, enrichment


class AdaptiveBlockState:
    def __init__(self, retention=.10, budget_candidates=(.10,.20,.35,.50,.75,1.),
                 audit_nmse=.02, audit_p90=.10, audit_queries=32,
                 query_block=64, key_block=64, history_min_blocks=4,
                 fine_ratio=.15, coarse_size=128, capture=False, audit_guard=True,
                 validation_queries=0, profile_stages=False):
        if not 0 < retention <= 1 or audit_nmse <= 0 or audit_p90 <= 0:
            raise ValueError('Invalid retention/audit thresholds')
        if query_block not in (32,64) or key_block not in (32,64,128):
            raise ValueError('Tensor-core block dimensions unsupported')
        if audit_queries < 4 or history_min_blocks < 0:
            raise ValueError('Invalid audit/history size')
        if validation_queries < 0:
            raise ValueError('Invalid independent validation size')
        if not all(0 < r <= 1 for r in budget_candidates):
            raise ValueError('Invalid candidate budget')
        self.retention, self.candidates = retention, sorted(set(r for r in (retention,*budget_candidates,1.) if r>=retention))
        self.audit_guard = audit_guard
        self.audit_nmse, self.audit_p90, self.audit_queries = audit_nmse, audit_p90, audit_queries
        self.query_block, self.key_block, self.history_min = query_block, key_block, history_min_blocks
        self.value = HierarchicalValueState(fine_ratio=fine_ratio, coarse_size=coarse_size,deterministic=True)
        self.capture = capture
        self.validation_queries, self.profile_stages = validation_queries, profile_stages
        self.reset(1,1,1)

    def reset(self, frames, height, width):
        self.frames, self.height, self.width = frames, height, width
        self.value.reset(frames, height, width)
        self.history = None
        self.records, self.snapshots = [], []


class AdaptiveBlockValueAttention(nn.Module):
    def __init__(self, original, state, layer_idx):
        super().__init__()
        self.original, self.state, self.layer_idx = original, state, layer_idx

    def forward(self, x, pos=None):
        if self.training or len(x) != 1:
            raise ValueError('Inference, batch one only')
        o, s = self.original, self.state
        stage_events = []
        def mark(name):
            if s.profile_stages and x.is_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record(); stage_events.append((name,event))
        mark('start')
        _, n, channels = x.shape
        h, d = o.num_heads, o.head_dim
        q, k, v = o.qkv(x).reshape(1,n,3,h,d).permute(2,0,3,1,4).unbind(0)
        patches, specials = token_layout(s.frames,s.height*s.width,5,x.device)
        mark('qkv_layout')
        with torch.autocast(device_type=x.device.type, enabled=False):
            hierarchy = s.value.update(v[0,:,patches].float().mean(0),self.layer_idx+1)
        mark('value_hierarchy')
        q, k = o.q_norm(q), o.k_norm(k)
        if o.rope is not None:
            q, k = o.rope(q,pos), o.rope(k,pos)
        # Match formal dense Flash precision after normalization/RoPE.
        q, k, v = q[0].to(v.dtype).contiguous(), k[0].to(v.dtype).contiguous(), v[0].contiguous()
        kp, vp = k[:,patches].contiguous(), v[:,patches].contiguous()
        ks, vs = k[:,specials].contiguous(), v[:,specials].contiguous()
        qb, kb = s.query_block, s.key_block
        nq, nb = math.ceil(n/qb), math.ceil(len(patches)/kb)
        bootstrap = self.layer_idx == 0
        mark('normalization_layout')
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled, population = block_mean(kp.float(),kb)
            # Two distinct actual queries per block, average AFTER QK/softmax.
            block_ids = torch.arange(nq,device=x.device)
            p1 = (block_ids*qb+qb//4).clamp_max(n-1)
            p2 = (block_ids*qb+3*qb//4).clamp_max(n-1)
            # Auditing uses DIFFERENT query offsets from routing.
            audit_blocks = torch.linspace(0,nq-1,min(nq,s.audit_queries),device=x.device).long().unique()
            probes = (audit_blocks*qb+qb//2).clamp_max(n-1)
            if bootstrap:
                routing = torch.zeros((1,1,nb),device=x.device).expand(h,nq,nb)
                sensing = torch.full((h,len(probes),nb),1/nb,device=x.device)
                stats = {key:torch.zeros(h,device=x.device) for key in ('concentration','specificity','cs')}
            else:
                routing = ((q[:,p1].float()@pooled.transpose(1,2)*o.scale).softmax(-1)+
                           (q[:,p2].float()@pooled.transpose(1,2)*o.scale).softmax(-1))*.5
                sensing = (q[:,probes].float()@pooled.transpose(1,2)*o.scale).softmax(-1)
                stats = reliability_statistics(sensing)
            mark('coarse_sensing')
            contrib = audit_contributions(q,kp,vp,ks,vs,probes,kb,o.scale)
            mark('dense_budget_probes')
            value_order = coverage_order(hierarchy,nb,kb,uniform=bootstrap)
            matching, enrichment = match_history(sensing,s.history,audit_blocks)
            decisions = audited_decisions(s,routing,value_order,stats,contrib,audit_blocks,
                matching,enrichment,n,len(specials),nb,bootstrap)
            mark('history_matching_budget_selection')
        sparse_heads = [head for head,c in enumerate(decisions) if not c['dense_fallback']]
        dense_heads = [head for head,c in enumerate(decisions) if c['dense_fallback']]
        out = torch.empty_like(q)
        new_history = [None]*h
        reused_counts, explored_counts, promoted_counts = [0]*h,[0]*h,[0]*h
        actual_density = [1.]*h
        snapshot_masks = [None]*h
        if sparse_heads:
            max_count = max(decisions[head]['blocks'] for head in sparse_heads)
            masks = torch.full((len(sparse_heads),nq,max_count),-1,device=x.device,dtype=torch.int32)
            origins = []
            mask_stats = []
            for slot,head in enumerate(sparse_heads):
                c = decisions[head]
                old = s.history[matching[head]] if c['use_history'] else None
                ids,origin = select_block_mask(routing[head],value_order,c['blocks'],c['a_blocks'],old)
                masks[slot,:,:c['blocks']] = ids.int()
                origins.append(origin)
                mask_stats.append(torch.stack(((origin==1).sum(),(origin==2).sum(),
                    (population[ids].sum(-1).float()+len(specials)).mean()/n)))
                if s.capture:
                    snapshot_masks[head] = ids.int().cpu()
            for head,(reused,explored,density) in zip(sparse_heads,torch.stack(mask_stats).cpu().tolist()):
                reused_counts[head],explored_counts[head],actual_density[head] = int(reused),int(explored),density
            mark('mask_assembly')
            if x.is_cuda:
                from saf3r.triton.adaptive_block_attention import block_attention
                counts = torch.tensor([decisions[head]['blocks'] for head in sparse_heads],device=x.device,dtype=torch.int32)
                out,evidence = block_attention(q,kp,vp,ks,vs,masks,counts,
                    torch.tensor(sparse_heads,device=x.device,dtype=torch.int32),
                    query_block=qb,key_block=kb,scale=o.scale,out=out)
            else:
                evidence = torch.full_like(masks,-float('inf'),dtype=torch.float32)
                for slot,head in enumerate(sparse_heads):
                    for block in range(nq):
                        qr = slice(block*qb,min((block+1)*qb,n))
                        ids = masks[slot,block,:decisions[head]['blocks']].long()
                        ix = (ids[:,None]*kb+torch.arange(kb,device=x.device)).flatten()
                        ix = ix[ix<len(patches)]
                        logits = q[head,qr].float()@kp[head,ix].float().T*o.scale
                        for j,block_id in enumerate(ids):
                            span = (ix//kb == block_id)
                            evidence[slot,block,j] = logits[:,span].logsumexp(-1).mean()
                        keys = torch.cat((kp[head,ix],ks[head]))
                        values = torch.cat((vp[head,ix],vs[head]))
                        out[head,qr] = F.scaled_dot_product_attention(q[head,qr],keys,values,scale=o.scale)
            mark('sparse_attention')
            promotion_stats = []
            for slot,head in enumerate(sparse_heads):
                c = decisions[head]
                # Discovery memory is independent of current A allocation, so
                # a transient Value-only layer does not erase all evidence.
                keep = 0 if bootstrap else min(c['blocks'],max(s.history_min,c['a_blocks']))
                winners = evidence[slot,:,:c['blocks']].topk(keep,-1).indices
                new_history[head] = masks[slot,:,:c['blocks']].gather(1,winners).clone()
                promotion_stats.append((origins[slot].gather(1,winners)==0).sum())
            for head,promoted in zip(sparse_heads,torch.stack(promotion_stats).cpu().tolist()):
                promoted_counts[head] = int(promoted)
            mark('same_pass_history_update')
        if dense_heads:
            hh = torch.tensor(dense_heads,device=x.device)
            # Flash SDPA requires an explicit batch dimension. A 3-D long
            # sequence can silently dispatch to a quadratic-memory backend.
            out[hh] = F.scaled_dot_product_attention(q[hh][None],k[hh][None],v[hh][None],
                dropout_p=0.,scale=o.scale)[0]
            # Dense fallback does not retain a full attention matrix or pretend
            # that unvalidated coarse candidates are exact history.
            for head in dense_heads:
                new_history[head] = torch.empty((nq,0),device=x.device,dtype=torch.int32)
        mark('dense_fallback')
        s.history = new_history
        s.records.append(dict(layer_zero_based=self.layer_idx,heads=h,query_block=qb,key_block=kb,
            patch_tokens=len(patches),special_tokens=len(specials),audit_queries=len(probes),
            decisions=decisions,actual_density=actual_density,dense_fallback_heads=dense_heads,
            concentration=stats['concentration'].detach().cpu().tolist(),
            specificity=stats['specificity'].detach().cpu().tolist(),
            head_matching=matching,history_enrichment=enrichment,reused_blocks=reused_counts,
            explored_blocks=explored_counts,promoted_value_blocks=promoted_counts,
            history_blocks_per_head=[0 if a is None else a.shape[-1] for a in new_history]))
        if s.validation_queries:
            # Independent output check: different query offsets from BOTH
            # routing (1/4,3/4) and budget selection (1/2). Never changes masks.
            blocks = torch.linspace(0,nq-1,min(nq,s.validation_queries),device=x.device).long().unique()
            validation_ids = (blocks*qb+3*qb//8).clamp_max(n-1)
            with torch.autocast(device_type=x.device.type,enabled=False):
                reference = F.scaled_dot_product_attention(q[:,validation_ids][None],k[None],v[None],
                    dropout_p=0.,scale=o.scale)[0].float()
                error = (out[:,validation_ids].float()-reference).square().sum(-1)
                energy = reference.square().sum(-1).clamp_min(1e-12)
                s.records[-1]['independent_validation'] = dict(query_ids=validation_ids.cpu().tolist(),
                    output_nmse=(error.sum(-1)/energy.sum(-1)).cpu().tolist(),
                    p90_query_nmse=torch.quantile(error/energy,.9,dim=-1).cpu().tolist(),
                    note='Held-out queries, not budget calibration probes; sparse-prefix current input. Does not alter masks.')
        if stage_events:
            stage_events[-1][1].synchronize()
            s.records[-1]['stage_ms'] = {stage_events[i][0]:stage_events[i-1][1].elapsed_time(stage_events[i][1])
                for i in range(1,len(stage_events))}
        if s.capture:
            s.snapshots.append(dict(layer=self.layer_idx,masks=snapshot_masks,
                hierarchy=[a.detach().cpu() for a in hierarchy],probes=probes.cpu(),
                history=[a.cpu() for a in new_history]))
        return o.proj_drop(o.proj(out.transpose(0,1).reshape(1,n,channels)))


def install_adaptive_block_value(model, **kwargs):
    if hasattr(model,'adaptive_block_value_state'):
        raise ValueError('Adaptive block attention already installed')
    state = AdaptiveBlockState(**kwargs)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn = AdaptiveBlockValueAttention(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim != 5 or len(images)!=1:
            raise ValueError('Expected batch-one images')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14)
    state.reset_hook_handle = model.aggregator.register_forward_pre_hook(reset)
    model.adaptive_block_value_state = state
    return state
