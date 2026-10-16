"""Memory-bounded exact-distance eager PAM for long VGGT inputs.

Uses a resident FP32 N x N distance matrix, but no other N x N temporaries.
Every non-medoid is examined for every removal at final convergence. Eager
candidate order and seeded starts differ from the small-input best-swap PAM.
"""
from __future__ import annotations

import time

import numpy as np
import torch


@torch.inference_mode()
def blocked_distances(values, block_size=256):
    x = values.float()
    squared = x.square().sum(1)
    n = len(x)
    distance = torch.empty((n, n), device=x.device, dtype=torch.float32)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        block = x[start:stop] @ x[start:].T
        block.mul_(-2).add_(squared[None, start:]).add_(squared[start:stop, None]).clamp_min_(0).sqrt_()
        diagonal = block[:, :stop-start]
        diagonal.copy_((diagonal + diagonal.T)*.5)
        distance[start:stop, start:] = block
        distance[start:, start:stop] = block.T
    distance.fill_diagonal_(0)
    return distance


@torch.inference_mode()
def host_distances(values, device="cuda", block_size=256):
    """Full FP32 matrix in pinned host RAM, GPU-sized row tiles only."""
    x = values.float().to(device)
    squared = x.square().sum(1)
    distance = torch.empty((len(x), len(x)), dtype=torch.float32, pin_memory=True)
    for start in range(0, len(x), block_size):
        stop = min(len(x), start+block_size)
        block = x[start:stop] @ x[start:].T
        block.mul_(-2).add_(squared[None, start:]).add_(squared[start:stop, None]).clamp_min_(0).sqrt_()
        diagonal = block[:, :stop-start]
        diagonal.copy_((diagonal + diagonal.T)*.5)
        distance[start:stop, start:].copy_(block)
        distance[start:, start:stop].copy_(block.T)
    distance.fill_diagonal_(0)
    return distance


def nearest_two(distance, medoids, max_entries=16_000_000):
    # Gather contiguous rows before transposing to avoid a huge strided gather.
    if len(distance)*len(medoids) <= max_entries:
        return distance[medoids].T.contiguous().topk(2, dim=1, largest=False)
    nearest = torch.empty((len(distance), 2), dtype=distance.dtype, device=distance.device)
    positions = torch.empty((len(distance), 2), dtype=torch.long, device=distance.device)
    for start in range(0, len(distance), 1024):
        stop = min(len(distance), start+1024)
        block, indices = distance[medoids, start:stop].T.contiguous().topk(2, dim=1, largest=False)
        nearest[start:stop], positions[start:stop] = block, indices
    return nearest, positions


def swap_costs(distance_rows, d1, d2, owner, k):
    base = (distance_rows - d1[None]).clamp_max_(0)
    correction = torch.minimum(distance_rows, d2[None]) - d1[None] - base
    grouped = torch.zeros((len(distance_rows), k), device=distance_rows.device, dtype=distance_rows.dtype)
    grouped.scatter_add_(1, owner[None].expand(len(distance_rows), -1), correction)
    return grouped + base.sum(1, keepdim=True)


@torch.inference_mode()
def eager_pam(distance, k, seed, max_passes=100, candidate_batch=32, verbose=False, compute_device=None):
    n = len(distance)
    if not 2 <= k < n:
        raise ValueError("Eager PAM requires 2 <= K < N")
    device = torch.device(compute_device) if compute_device is not None else distance.device
    hybrid = device != distance.device
    generator = torch.Generator(device=device).manual_seed(seed)
    # Min total distance is the initial real medoid, followed by distance-
    # squared weighted sampling. No restriction to a sampled token subset.
    first = int(distance.sum(1).argmin())
    medoids = torch.empty(k, device=device, dtype=torch.long)
    medoids[0] = first
    selected = torch.zeros(n, device=device, dtype=torch.bool)
    selected[first] = True
    closest = distance[first].to(device).clone()
    for j in range(1, k):
        weights = closest.square().masked_fill_(selected, 0)
        if float(weights.sum()) == 0:
            candidate = int((~selected).nonzero()[0])
        else:
            candidate = int(torch.multinomial(weights, 1, generator=generator))
        medoids[j] = candidate
        selected[candidate] = True
        torch.minimum(closest, distance[candidate].to(device, non_blocking=True), out=closest)
    if hybrid:
        nearest, positions = distance[medoids.cpu()].to(device).T.contiguous().topk(2, dim=1, largest=False)
    else:
        nearest, positions = nearest_two(distance, medoids)
    objective = float(nearest[:, 0].double().sum())
    initial_objective = objective
    tolerance = max(1e-6, objective*1e-7)
    history = [objective]
    pass_history = []
    converged = False
    final_delta = None
    for epoch in range(max_passes):
        start_time = time.perf_counter()
        cursor, accepted, best_delta = 0, 0, float("inf")
        while cursor < n:
            stop = min(n, cursor + candidate_batch)
            costs = swap_costs(distance[cursor:stop].to(device, non_blocking=True), nearest[:, 0], nearest[:, 1], positions[:, 0], k)
            costs.masked_fill_(selected[cursor:stop, None], torch.inf)
            deltas, removals = costs.min(1)
            best_delta = min(best_delta, float(deltas.min()))
            improvements = (deltas < -tolerance).nonzero().flatten()
            if len(improvements) == 0:
                cursor = stop
                continue
            offset = int(improvements[0])
            candidate, removed = cursor + offset, int(removals[offset])
            old_medoid = int(medoids[removed])
            proposed = medoids.clone()
            proposed[removed] = candidate
            if hybrid or device.type == "cuda":
                # Only points losing a first/second medoid need all K distances.
                # Every other point just inserts the new candidate into its top2.
                affected = (positions == removed).any(1).nonzero().flatten()
                row = distance[candidate].to(device, non_blocking=True)
                first_mask = row < nearest[:, 0]
                second_mask = (~first_mask) & (row < nearest[:, 1])
                new_nearest = torch.stack((torch.where(first_mask, row, nearest[:, 0]),
                                          torch.where(first_mask, nearest[:, 0], torch.where(second_mask, row, nearest[:, 1]))), dim=1)
                new_positions = torch.stack((torch.where(first_mask, removed, positions[:, 0]),
                                            torch.where(first_mask, positions[:, 0], torch.where(second_mask, removed, positions[:, 1]))), dim=1)
                if len(affected):
                    if hybrid:
                        sub = distance.numpy()[np.ix_(proposed.cpu().numpy(), affected.cpu().numpy())]
                        sub = torch.from_numpy(sub).to(device)
                    else:
                        sub = distance[proposed[:, None], affected[None, :]]
                    exact, exact_positions = sub.T.contiguous().topk(2, dim=1, largest=False)
                    new_nearest[affected] = exact
                    new_positions[affected] = exact_positions
            else:
                new_nearest, new_positions = nearest_two(distance, proposed)
            new_objective = float(new_nearest[:, 0].double().sum())
            if objective - new_objective > tolerance:
                selected[old_medoid] = False
                selected[candidate] = True
                medoids, nearest, positions = proposed, new_nearest, new_positions
                objective = new_objective
                history.append(objective)
                accepted += 1
            cursor = candidate + 1
        pass_history.append({"pass": epoch+1, "swaps": accepted, "objective": objective,
                             "seconds": time.perf_counter()-start_time, "minimum_evaluated_delta": best_delta})
        if verbose:
            print(f"  eager PAM seed={seed} pass={epoch+1} swaps={accepted} objective={objective:.6f} seconds={pass_history[-1]['seconds']:.1f}", flush=True)
        if accepted == 0:
            final_delta = best_delta
            converged = best_delta >= -tolerance
            break
    return {"medoids": medoids.cpu().numpy(), "assignments": positions[:, 0].cpu().numpy(),
            "distances": nearest[:, 0].cpu().numpy(), "objective": objective,
            "initial_objective": initial_objective, "objective_history": history,
            "swaps": len(history)-1, "passes": pass_history, "converged": converged,
            "stop_reason": "no_improving_single_swap" if converged else "max_passes_or_numerical_tolerance",
            "best_final_swap_delta": final_delta, "tolerance": tolerance,
            "initialisation": "seeded_distance_weighted", "seed": seed,
            "algorithm": "FP32 exact-distance exhaustive eager PAM (blocked candidates)"}


def rust_pam(distance, k, seed, max_passes=100, cpu_threads=4):
    """Compiled CPU fallback; certify the chosen fit with one FastPAM1 scan."""
    import kmedoids
    matrix = distance.numpy() if isinstance(distance, torch.Tensor) else distance
    start = time.perf_counter()
    fit = kmedoids.fasterpam(matrix, k, random_state=seed, max_iter=max_passes, n_cpu=cpu_threads)
    print(f"  FasterPAM seed={seed}: passes={fit.n_iter} swaps={fit.n_swap} seconds={time.perf_counter()-start:.1f}", flush=True)
    # FastPAM1 examines all candidate/removal pairs for its best single swap.
    # If rounding leaves an improvement, finish those best swaps explicitly.
    certificate = kmedoids.fastpam1(matrix, fit.medoids, max_iter=max_passes)
    check = kmedoids.fastpam1(matrix, certificate.medoids, max_iter=1)
    medoids = np.asarray(certificate.medoids, dtype=np.int64)
    labels = np.asarray(certificate.labels, dtype=np.int64)
    distances = matrix[np.arange(len(matrix)), medoids[labels]]
    objective = float(distances.sum(dtype=np.float64))
    tolerance = max(1e-6, objective*1e-7)
    return {"medoids": medoids, "assignments": labels, "distances": distances,
            "objective": objective, "initial_objective": None, "objective_history": [objective],
            "swaps": int(fit.n_swap + certificate.n_swap), "passes": int(fit.n_iter),
            "converged": int(check.n_swap) == 0, "stop_reason": "independent_fastpam1_exhaustive_no_swap_check",
            "best_final_swap_delta": float(check.loss-certificate.loss), "tolerance": tolerance,
            "initialisation": "seeded_random", "seed": seed, "certificate_swaps": int(check.n_swap),
            "algorithm": "kmedoids 0.5.5 FasterPAM + FastPAM1 exhaustive single-swap certificate"}
