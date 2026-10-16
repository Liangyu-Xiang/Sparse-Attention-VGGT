"""FP64 Euclidean distortion and synchronous, anchored local k-medoids.

Graph neighborhoods remain anchored to the source partition throughout a fit.
This module keeps frozen labels, frozen representatives and recentered labels
separate so experimental baselines cannot silently conflate these choices.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def distortion(values, labels, medoids):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    medoids = np.asarray(medoids, dtype=np.int64)
    distances = np.linalg.norm(values-values[medoids[labels]], axis=1)
    return float(distances.sum(dtype=np.float64)), distances


def recenter_partition(values, labels, old_medoids, block_size=256):
    """Exact medoid of every fixed cluster, not mean/nearest-to-mean."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    medoids = np.asarray(old_medoids, dtype=np.int64).copy()
    ordered = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(labels, minlength=len(medoids)))]
    for cluster in range(len(medoids)):
        members = ordered[offsets[cluster]:offsets[cluster+1]]
        if not len(members):
            raise ValueError(f"Empty cluster {cluster}")
        points = values[members]
        previous = medoids[cluster]
        best = int(previous) if np.any(members == previous) else int(members[0])
        best_cost = float(np.linalg.norm(points-values[best], axis=1).sum())
        for start in range(0, len(members), block_size):
            costs = cdist(points[start:start+block_size], points).sum(axis=1)
            local = int(costs.argmin())
            cost = float(costs[local])
            # Preserve the previous real token on a numerical tie.
            if cost < best_cost-1e-12*max(1., abs(best_cost)):
                best, best_cost = int(members[start+local]), cost
        medoids[cluster] = best
    return medoids


def source_neighborhoods(source_values, source_medoids, q, block_size=512):
    source_medoids = np.asarray(source_medoids, dtype=np.int64)
    k = len(source_medoids)
    if not 0 <= q < k:
        raise ValueError("q must lie in 0..K-1")
    values = np.asarray(source_values, dtype=np.float64)[source_medoids]
    neighbors = np.empty((k, q+1), dtype=np.int64)
    for start in range(0, k, block_size):
        stop = min(k, start+block_size)
        distance = cdist(values[start:stop], values)
        distance[np.arange(stop-start), np.arange(start, stop)] = -np.inf
        partial = np.argpartition(distance, q, axis=1)[:, :q+1]
        for local, row in enumerate(distance):
            cutoff = row[partial[local]].max()
            candidates = np.flatnonzero(row <= cutoff)
            # Exact tie-breaking by source label, identical to stable full sort.
            selected = np.lexsort((candidates, row[candidates]))[:q+1]
            neighbors[start+local] = candidates[selected]
    return neighbors


def synchronous_assignment(values, anchor_labels, current_labels, medoids, neighborhoods, block_size=1024, anchors=None):
    values = np.asarray(values, dtype=np.float64)
    labels = np.empty_like(current_labels)
    distances = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), block_size):
        stop = min(len(values), start+block_size)
        candidates = neighborhoods[anchor_labels[start:stop]]
        difference = values[start:stop, None, :]-values[medoids[candidates]]
        d = np.linalg.norm(difference, axis=2)
        choice = np.argmin(d, axis=1)
        best = d[np.arange(stop-start), choice]
        chosen_labels = candidates[np.arange(stop-start), choice]
        old = current_labels[start:stop]
        own_distance = np.linalg.norm(values[start:stop]-values[medoids[old]], axis=1)
        # This also preserves each medoid's own label on zero-distance ties,
        # keeping all K clusters nonempty without arbitrary reseeding.
        retain = own_distance <= best
        labels[start:stop] = np.where(retain, old, chosen_labels)
        distances[start:stop] = np.where(retain, own_distance, best)
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=np.int64)
        labels[anchors] = np.arange(len(medoids))
        distances[anchors] = np.linalg.norm(values[anchors]-values[medoids], axis=1)
    return labels, float(distances.sum(dtype=np.float64))


def local_refine(values, anchor_labels, initial_medoids, neighborhoods, max_iterations=100, anchors=None,
                 save_iterations=(), stop_when_converged=True):
    values = np.asarray(values, dtype=np.float64)
    anchor_labels = np.asarray(anchor_labels, dtype=np.int64)
    labels = anchor_labels.copy()
    medoids = np.asarray(initial_medoids, dtype=np.int64).copy()
    initial, _ = distortion(values, labels, medoids)
    history = [{"iteration": 0, "objective": initial, "changed_labels": 0, "changed_medoids": 0}]
    converged = False
    snapshots = {}
    for iteration in range(1, max_iterations+1):
        assigned, assignment_objective = synchronous_assignment(values, anchor_labels, labels, medoids, neighborhoods, anchors=anchors)
        updated = recenter_partition(values, assigned, medoids)
        objective, _ = distortion(values, assigned, updated)
        tolerance = 1e-10*max(1., initial)
        if assignment_objective > history[-1]["objective"]+tolerance or objective > assignment_objective+tolerance:
            raise AssertionError("Synchronous assignment/medoid update increased distortion")
        changed_labels = int(np.count_nonzero(assigned != labels))
        changed_medoids = int(np.count_nonzero(updated != medoids))
        history.append({"iteration": iteration, "objective_after_assignment": assignment_objective,
                        "objective": objective, "changed_labels": changed_labels, "changed_medoids": changed_medoids})
        labels, medoids = assigned, updated
        if iteration in save_iterations:
            snapshots[iteration] = {"assignments": labels.copy(), "medoids": medoids.copy(), "objective": objective}
        if changed_labels == 0 and changed_medoids == 0:
            converged = True
            if stop_when_converged:
                break
    if not np.all(np.any(neighborhoods[anchor_labels] == labels[:, None], axis=1)):
        raise AssertionError("A token escaped its fixed source neighborhood")
    return {"assignments": labels, "medoids": medoids, "objective": history[-1]["objective"],
            "history": history, "converged": converged, "iterations": len(history)-1,
            "migration_fraction": float(np.mean(labels != anchor_labels)), "snapshots": snapshots}
