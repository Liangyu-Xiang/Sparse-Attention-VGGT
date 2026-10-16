import numpy as np
import pytest

from scripts.report_value_neighborhood_recall import neighborhood_recall


def test_token_provenance_recall_matches_explicit_cluster_unions():
    # A and B are close, C far away. Target labels are unrelated to source IDs.
    values = np.array([[0.], [.1], [1.], [1.1], [10.], [10.1]])
    source = np.array([0, 0, 1, 1, 2, 2])
    source_medoids = np.array([0, 2, 4])
    target = np.array([1, 0, 0, 1, 2, 2])
    target_medoids = np.array([2, 0, 4])
    rows = neighborhood_recall(values, source, source_medoids, target, target_medoids, [0, 1, 2])
    assert [r["recall"] for r in rows] == [4/6, 1., 1.]
    for q, row in enumerate(rows):
        hits, candidates, chance = [], [], []
        for i in range(6):
            own = source[i]
            neighbors = sorted((c for c in range(3) if c != own),
                               key=lambda c: abs(values[source_medoids[c], 0]-values[source_medoids[own], 0]))
            union = np.flatnonzero(np.isin(source, [own]+neighbors[:q]))
            hits.append(target_medoids[target[i]] in union)
            candidates.append(len(union)/6)
            chance.append(np.mean(np.isin(target_medoids[target], union)))
        assert row["recall"] == np.mean(hits)
        assert row["candidate_token_fraction"] == pytest.approx(np.mean(candidates))
        assert row["independent_assignment_recall"] == pytest.approx(np.mean(chance))
    # Permuting target cluster IDs must change nothing.
    perm = np.array([2, 0, 1])
    new_medoids = np.empty_like(target_medoids)
    new_medoids[perm] = target_medoids
    assert rows == neighborhood_recall(values, source, source_medoids, perm[target], new_medoids, [0, 1, 2])


def test_self_cluster_first_even_when_medoid_vectors_coincide():
    rows = neighborhood_recall(np.zeros((4, 2)), [0, 0, 1, 1], [0, 2],
                               [0, 0, 1, 1], [0, 2], [0, 1])
    assert rows[0]["recall"] == 1
    assert rows[0]["candidate_token_fraction"] == .5
    assert rows[1]["candidate_token_fraction"] == 1
    with pytest.raises(ValueError):
        neighborhood_recall(np.zeros((4, 2)), [0, 0, 1, 1], [0, 2],
                            [0, 0, 1, 1], [0, 2], [2])


def test_blocked_fp64_verification_matches_full_distance_check(tmp_path):
    from scipy.spatial.distance import cdist
    from scripts.report_value_neighborhood_recall import verify_distances
    values = np.random.default_rng(13).normal(size=(31, 7)).astype(np.float32)
    medoids = np.array([0, 5, 10, 20])
    full = cdist(values.astype(float), values[medoids].astype(float))
    labels = full.argmin(1)
    np.savez(tmp_path/"fit.npz", distances=full.min(1))
    reference = verify_distances(values, (labels, medoids), tmp_path, "fit", block_size=1024)
    assert reference == verify_distances(values, (labels, medoids), tmp_path, "fit", block_size=3)
