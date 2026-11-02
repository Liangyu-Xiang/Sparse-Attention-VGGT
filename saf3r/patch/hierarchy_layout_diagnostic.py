"""Nested parent layouts and exact moments of fine Reps, for diagnostics only.

This module is deliberately separate from the single-parent Attention router.
Its distortions concern fine Rep vectors, not all Values or Attention outputs.
"""
import torch
import torch.nn.functional as F

from .hierarchical_reuse import balanced_parents


def build_layout(vectors, weights, branching):
    """Build one or more balanced parent levels with real descendant Reps.

    ``vectors`` are unit fine-Rep Values. A node stores the UNNORMALIZED
    weighted sum and population. Only its grouping feature is normalized.
    ``representative`` indexes a real fine Rep; no virtual Key is emitted.
    Per-level descendant owners keep construction linear in leaf count except
    for balanced_parents' sorting/refinement. No leaf-to-all-parent search.
    """
    factors = tuple(branching)
    if not factors or any(b not in (2, 4) for b in factors):
        raise ValueError('Each parent level must have 2 or 4 children')
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError('Expected nonempty [fine_clusters, Value_dimension]')
    if weights.shape != (len(vectors),):
        raise ValueError('One weight is required per fine cluster')
    if not bool(torch.isfinite(vectors).all() and torch.isfinite(weights).all()):
        raise ValueError('Nonfinite input')
    if not bool((weights > 0).all()):
        raise ValueError('Weights must be positive')
    norm = vectors.float().norm(dim=-1)
    if not bool(((norm - 1).abs() < 1e-4).all()):
        raise ValueError('Fine Rep vectors must have unit norm')
    vectors, weights = vectors.float(), weights.float()
    device = vectors.device
    leaf_ids = torch.arange(len(vectors), device=device)
    owner = leaf_ids
    mass = weights
    moment = weights[:, None] * vectors
    representative = leaf_ids
    feature = vectors
    levels = []
    for depth, arity in enumerate(factors, 1):
        children = balanced_parents(feature, torch.arange(len(feature), device=device), arity)
        valid = children >= 0
        child_to_parent = torch.empty(len(feature), device=device, dtype=torch.long)
        parent_id = torch.arange(len(children), device=device)[:, None].expand_as(children)
        child_to_parent[children[valid].long()] = parent_id[valid]
        owner = child_to_parent[owner]
        safe = children.clamp_min(0).long()
        parent_mass = (mass[safe] * valid).sum(1)
        parent_moment = (moment[safe] * valid[:, :, None]).sum(1)
        center = parent_moment / parent_mass[:, None]
        # Unit member closest to the node mean. Ties choose lowest leaf ID.
        distances = (vectors - center[owner]).square().sum(-1)
        best = torch.full_like(parent_mass, torch.inf)
        best.scatter_reduce_(0, owner, distances, reduce='amin')
        choices = torch.where(distances == best[owner], leaf_ids, len(vectors))
        representative = torch.full((len(children),), len(vectors), device=device, dtype=torch.long)
        representative.scatter_reduce_(0, owner, choices, reduce='amin')
        # Opposing Values may cancel. Keep the zero moment for loss evaluation;
        # use the real selected Rep solely as a deterministic grouping fallback.
        direction = F.normalize(parent_moment, dim=-1)
        feature = torch.where((parent_moment.norm(dim=-1) > 1e-12)[:, None],
                              direction, vectors[representative])
        levels.append(dict(depth=depth, arity=arity, children=children,
                           leaf_owner=owner, mass=parent_mass, moment=parent_moment,
                           representative=representative))
        mass, moment = parent_mass, parent_moment
    return levels


def summarize_layout(vectors, weights, levels):
    """Exact weighted cosine distortion over fine Rep vectors at each level."""
    rows = []
    for level in levels:
        rep = level['representative']
        loss = (level['mass'] - (vectors[rep] * level['moment']).sum(-1)).clamp_min(0)
        direct = (weights * (1 - (vectors * vectors[rep[level['leaf_owner']]]).sum(-1))).sum()
        rows.append(dict(depth=level['depth'], arity=level['arity'],
                         nodes=len(rep), max_fine_descendants=int(torch.bincount(level['leaf_owner']).max()),
                         weighted_rep_cosine_distortion=float(loss.sum() / weights.sum()),
                         moment_vs_direct_abs_error=float((loss.sum() - direct).abs() / weights.sum()),
                         total_mass=float(level['mass'].sum())))
    return rows
