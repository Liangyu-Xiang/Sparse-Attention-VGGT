import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scripts.analyze_value_cluster_evolution import (
    all_contingencies,
    cluster_change,
    partition_scores,
)


def test_all_head_pair_scores_match_sklearn():
    rng = np.random.default_rng(94)
    source = rng.integers(0, 5, (3, 71))
    target = rng.integers(0, 5, (4, 71))
    counts = all_contingencies(source, target, 5)
    ari, nmi = partition_scores(counts)
    np.testing.assert_array_equal(counts.sum(axis=(-2, -1)), np.full((3, 4), 71))
    for i in range(3):
        for j in range(4):
            assert np.isclose(ari[i, j], adjusted_rand_score(source[i], target[j]))
            assert np.isclose(nmi[i, j], normalized_mutual_info_score(source[i], target[j]))
            for a in range(5):
                for b in range(5):
                    assert counts[i, j, a, b] == ((source[i] == a) & (target[j] == b)).sum()


def test_relabeling_and_degenerate_partitions():
    labels = np.array([[0, 0, 1, 1, 2, 2]])
    remapping = np.array([2, 0, 1])
    counts = all_contingencies(labels, remapping[labels], 3)[0, 0]
    np.testing.assert_allclose(partition_scores(counts), [1, 1])
    change = cluster_change(counts)
    assert change["aligned_same_fraction"] == 1
    assert change["source_to_target_cluster_mapping"] == remapping.tolist()
    assert change["split_source_clusters"] == 0
    assert change["merged_target_clusters"] == 0
    for a, b in ((np.zeros(6, int), np.zeros(6, int)),
                 (np.zeros(6, int), np.arange(6)),
                 (np.arange(6), np.arange(6)[::-1])):
        counts = all_contingencies(a[None], b[None], 6)[0, 0]
        ari, nmi = partition_scores(counts)
        assert np.isclose(ari, adjusted_rand_score(a, b))
        assert np.isclose(nmi, normalized_mutual_info_score(a, b))


def test_split_and_merge_are_membership_flows_not_equal_cluster_ids():
    # Source C1 splits evenly; target C2 merges equal portions from both sources.
    counts = np.array([[5, 5], [0, 5]])
    change = cluster_change(counts)
    assert change["split_source_clusters"] == 1
    assert change["merged_target_clusters"] == 1
    assert np.isclose(change["aligned_same_fraction"], 10 / 15)
