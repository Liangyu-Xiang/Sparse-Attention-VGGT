"""GPU-resident local clustering with exact and cheap representative policies.

Exact-policy FP64 distance arithmetic matches the CPU reference. Its ragged
member pairs use bounded chunks, not largest-cluster padding. Centroid-nearest
and sampled policies approximate the original unsquared Euclidean objective.
"""
from __future__ import annotations

import torch


def source_neighborhoods_gpu(values, medoids, q, block_size=512):
    x = values.double()[medoids]
    k = len(medoids)
    if not 0 <= q < k:
        raise ValueError("q must lie in 0..K-1")
    result = torch.empty((k, q+1), dtype=torch.long, device=x.device)
    for start in range(0, k, block_size):
        stop = min(k, start+block_size)
        # Same FP64 Euclidean metric, but use GEMM rather than the slow direct
        # CUDA cdist reduction. No TF32/FP32 distance approximation is used.
        distance = torch.cdist(x[start:stop], x, compute_mode="use_mm_for_euclid_dist")
        distance[torch.arange(stop-start, device=x.device), torch.arange(start, stop, device=x.device)] = -torch.inf
        # Select only q+1 entries, not an O(K log K) full row sort. Repair
        # arbitrary topk boundary ties using the smallest actual cluster IDs,
        # then lexicographically sort only 2*(q+1) candidates.
        width = q+1
        best_distance, best_index = distance.topk(width, dim=1, largest=False, sorted=True)
        cutoff = best_distance[:, -1:]
        strict = torch.where(best_distance < cutoff, best_index, k)
        boundary_ids = torch.where(distance == cutoff, torch.arange(k, device=x.device)[None], k)
        boundary = boundary_ids.topk(width, dim=1, largest=False, sorted=False).values
        ids = torch.cat((strict, boundary), dim=1).sort(dim=1).values
        costs = distance.gather(1, ids.clamp_max(k-1)).masked_fill(ids == k, torch.inf)
        rank = costs.argsort(dim=1, stable=True)[:, :width]
        result[start:stop] = ids.gather(1, rank)
    return result


def synchronous_assignment_gpu(values, source_labels, current_labels, medoids, neighborhoods,
                               anchors, block_size=8192):
    labels = torch.empty_like(current_labels)
    for start in range(0, len(values), block_size):
        stop = min(len(values), start+block_size)
        x = values[start:stop]
        candidates = neighborhoods[source_labels[start:stop]]
        distances = (x[:, None]-values[medoids[candidates]]).square().sum(-1).sqrt()
        best, choice = distances.min(1)
        chosen = candidates.gather(1, choice[:, None]).squeeze(1)
        old = current_labels[start:stop]
        own = (x-values[medoids[old]]).square().sum(-1).sqrt()
        labels[start:stop] = torch.where(own <= best, old, chosen)
    labels[anchors] = torch.arange(len(medoids), device=values.device)
    return labels


def recenter_partition_gpu(values, labels, previous_medoids, pair_block=65536):
    """Evaluate every real member as medoid, with O(pair_block*d) workspace."""
    n, k = len(values), len(previous_medoids)
    counts = torch.bincount(labels, minlength=k)
    # One scalar synchronization determines a finite ragged loop, not a CPU fit.
    pairs_per_cluster = counts.square()
    pair_ends = pairs_per_cluster.cumsum(0)
    total_pairs = int(pair_ends[-1])
    offsets = counts.cumsum(0)-counts
    pair_starts = pair_ends-pairs_per_cluster
    order = labels.argsort(stable=True)
    costs = torch.zeros(n, device=values.device, dtype=torch.float64)
    for start in range(0, total_pairs, pair_block):
        pair_ids = torch.arange(start, min(total_pairs, start+pair_block), device=values.device)
        cluster = torch.searchsorted(pair_ends, pair_ids, right=True)
        within = pair_ids-pair_starts[cluster]
        size = counts[cluster]
        candidate = order[offsets[cluster]+torch.div(within, size, rounding_mode="floor")]
        member = order[offsets[cluster]+within.remainder(size)]
        distance = (values[candidate]-values[member]).square().sum(-1).sqrt()
        costs.scatter_add_(0, candidate, distance)
    minimum = torch.full((k,), torch.inf, device=values.device, dtype=torch.float64)
    minimum.scatter_reduce_(0, labels, costs, reduce="amin", include_self=True)
    ids = torch.arange(n, device=values.device)
    optimal = torch.where(costs == minimum[labels], ids, n)
    medoids = torch.full((k,), n, device=values.device, dtype=torch.long)
    medoids.scatter_reduce_(0, labels, optimal, reduce="amin", include_self=True)
    own = torch.arange(k, device=values.device)
    tolerance = 1e-12*minimum.abs().clamp_min(1.)
    keep = (labels[previous_medoids] == own) & (costs[previous_medoids] <= minimum+tolerance)
    return torch.where(keep, previous_medoids, medoids), pairs_per_cluster.sum()


def _minimum_cost_members(costs, labels, previous_medoids):
    n, k = len(costs), len(previous_medoids)
    minimum = torch.full((k,), torch.inf, device=costs.device, dtype=costs.dtype)
    minimum.scatter_reduce_(0, labels, costs, reduce="amin", include_self=True)
    ids = torch.arange(n, device=costs.device)
    selected = torch.where(costs == minimum[labels], ids, n)
    representatives = torch.full((k,), n, device=costs.device, dtype=torch.long)
    representatives.scatter_reduce_(0, labels, selected, reduce="amin", include_self=True)
    own = torch.arange(k, device=costs.device)
    keep = (labels[previous_medoids] == own) & (
        costs[previous_medoids] <= minimum+1e-12*minimum.abs().clamp_min(1.))
    return torch.where(keep, previous_medoids, representatives)


def stable_cluster_mean(values, labels, k):
    """Fixed token-order segmented means, no floating-point atomic updates."""
    order = labels.argsort(stable=True)
    counts = torch.zeros(k,device=labels.device,dtype=torch.long)
    counts.index_add_(0,labels,torch.ones_like(labels))
    return torch.segment_reduce(values[order],reduce='mean',lengths=counts).nan_to_num()


def centroid_nearest_gpu(values, labels, previous_medoids, deterministic=False):
    """O(Nd): choose an actual member nearest to its cluster's centroid.

    Exact for the squared-Euclidean member-representative objective (up to
    floating point tolerance), approximate for the original sum of L2 norms.
    """
    k = len(previous_medoids)
    # Fixed-size scatter avoids bincount's dynamic output-size discovery.
    counts = torch.zeros(k, device=values.device, dtype=values.dtype)
    counts.index_add_(0, labels, torch.ones(len(values), device=values.device, dtype=values.dtype))
    if deterministic:
        centers = stable_cluster_mean(values,labels,k)
    else:
        sums = torch.zeros((k, values.shape[1]), device=values.device, dtype=values.dtype)
        sums.index_add_(0, labels, values)
        centers = sums/counts.clamp_min(1)[:, None]
    costs = (values-centers[labels]).square().sum(-1)
    return _minimum_cost_members(costs, labels, previous_medoids)


def sampled_medoid_gpu(values, labels, previous_medoids, candidates=4, block_size=8192):
    """Evaluate at most c deterministic member candidates against all members.

    Include the previous representative when valid; fill remaining slots with
    uniformly spaced positions in token-index-ordered membership. Small
    clusters can repeat candidates. O(cNd) distance work, no member-pair matrix.
    """
    if candidates < 2:
        raise ValueError("sampled medoids need at least two candidate slots")
    n, k = len(values), len(previous_medoids)
    counts = torch.bincount(labels, minlength=k)
    offsets = counts.cumsum(0)-counts
    order = labels.argsort(stable=True)
    positions = torch.div(counts[:, None]*torch.arange(candidates-1, device=values.device)[None],
                          candidates-1, rounding_mode="floor")
    sampled = order[offsets[:, None]+positions]
    own = torch.arange(k, device=values.device)
    valid_old = labels[previous_medoids] == own
    first = torch.where(valid_old, previous_medoids, order[offsets])
    ids = torch.cat((first[:, None], sampled), dim=1)
    costs = torch.zeros((k, candidates), device=values.device, dtype=values.dtype)
    for start in range(0, n, block_size):
        stop = min(n, start+block_size)
        group = labels[start:stop]
        distance = (values[start:stop, None]-values[ids[group]]).square().sum(-1).sqrt()
        costs.scatter_add_(0, group[:, None].expand(-1, candidates), distance)
    minimum = costs.amin(1)
    selected = torch.where(costs == minimum[:, None], ids, n).amin(1)
    keep = valid_old & (costs[:, 0] <= minimum+1e-12*minimum.abs().clamp_min(1.))
    return torch.where(keep, previous_medoids, selected)


@torch.inference_mode()
def local_refine_gpu(values, source_labels, initial_medoids, neighborhoods, iterations=2,
                     representative_update="exact", sampled_candidates=4):
    if representative_update not in {"exact", "previous", "centroid", "sampled"}:
        raise ValueError("Unknown representative update")
    x = values.double()
    labels, medoids = source_labels.clone(), initial_medoids.clone()
    pairs = []
    snapshots = []
    for _ in range(iterations):
        labels = synchronous_assignment_gpu(x, source_labels, labels, medoids, neighborhoods,
                                            anchors=initial_medoids)
        if representative_update == "exact":
            medoids, pair_count = recenter_partition_gpu(x, labels, medoids)
        elif representative_update == "centroid":
            medoids = centroid_nearest_gpu(x, labels, medoids)
            pair_count = len(x)
        elif representative_update == "sampled":
            medoids = sampled_medoid_gpu(x, labels, medoids, sampled_candidates)
            pair_count = len(x)*sampled_candidates
        else:
            # Anchors and zero-distance tie retention keep these members valid.
            pair_count = 0
        pairs.append(pair_count)
        snapshots.append((labels.clone(), medoids.clone()))
    distances = (x-x[medoids[labels]]).square().sum(-1).sqrt()
    return {"assignments": labels, "medoids": medoids, "distances": distances,
            "objective": distances.sum(),
            "member_pair_counts": pairs if representative_update == "exact" else [],
            "representative_distance_counts": pairs,
            "representative_update": representative_update, "snapshots": snapshots}
