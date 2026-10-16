import numpy as np
from scipy.spatial.distance import cdist

from scripts.value_local_distortion import distortion, recenter_partition, source_neighborhoods, local_refine


def test_fixed_partition_uses_true_euclidean_medoid():
    values = np.array([[0.], [1.], [20.], [30.], [31.], [100.]])
    labels = np.repeat([0, 1], 3)
    medoids = recenter_partition(values, labels, [0, 5])
    np.testing.assert_array_equal(medoids, [1, 4])
    objective, distances = distortion(values, labels, medoids)
    assert objective == 90
    assert np.isclose(objective, distances.sum())


def test_exact_medoid_is_not_member_nearest_to_mean():
    values = np.array([[0.], [1.], [2.], [4.], [100.]])
    exact = recenter_partition(values, np.zeros(5, dtype=int), [4])
    np.testing.assert_array_equal(exact, [2])
    assert int(np.abs(values-values.mean()).argmin()) == 3


def test_q_zero_preserves_labels_and_equals_recentered_partition():
    values = np.array([[0.], [1.], [9.], [10.], [11.], [30.]])
    labels = np.repeat([0, 1], 3)
    medoids = np.array([0, 5])
    neighborhoods = source_neighborhoods(values, medoids, 0)
    fit = local_refine(values, labels, medoids, neighborhoods)
    assert fit["converged"]
    np.testing.assert_array_equal(fit["assignments"], labels)
    expected = recenter_partition(values, labels, medoids)
    np.testing.assert_array_equal(fit["medoids"], expected)
    assert fit["objective"] == distortion(values, labels, expected)[0]


def test_blocked_neighbors_match_stable_full_sort_including_ties():
    values = np.arange(24, dtype=float).reshape(12, 2)
    values[4] = values[3]
    medoids = np.arange(12)
    distance = cdist(values, values)
    np.fill_diagonal(distance, -np.inf)
    for q in (0, 4, 8, 11):
        expected = np.argsort(distance, axis=1, kind="stable")[:, :q+1]
        np.testing.assert_array_equal(source_neighborhoods(values, medoids, q, block_size=3), expected)


def test_synchronous_local_updates_reduce_distortion_and_are_fixed_point():
    old_values = np.array([[0.], [1.], [2.], [10.], [11.], [12.]])
    new_values = np.array([[0.], [1.], [9.], [10.], [11.], [12.]])
    labels = np.repeat([0, 1], 3)
    medoids = np.array([1, 4])
    neighborhoods = source_neighborhoods(old_values, medoids, 1)
    fit = local_refine(new_values, labels, medoids, neighborhoods)
    assert fit["converged"]
    assert fit["migration_fraction"] > 0
    assert fit["objective"] < fit["history"][0]["objective"]
    assert all(b["objective"] <= a["objective"] for a, b in zip(fit["history"], fit["history"][1:]))
    d = cdist(new_values, new_values[fit["medoids"]])
    np.testing.assert_allclose(d.min(1).sum(), fit["objective"])
    np.testing.assert_array_equal(recenter_partition(new_values, fit["assignments"], fit["medoids"]), fit["medoids"])


def test_duplicate_vectors_preserve_all_clusters_on_assignment_ties():
    values = np.zeros((8, 2))
    labels = np.repeat([0, 1], 4)
    fit = local_refine(values, labels, np.array([0, 4]), np.array([[0, 1], [1, 0]]))
    assert fit["converged"]
    np.testing.assert_array_equal(fit["assignments"], labels)


def test_two_iteration_snapshots_lock_original_medoid_anchors():
    values = np.array([[0.], [.1], [10.], [11.], [11.1], [11.2]])
    labels = np.repeat([0, 1], 3)
    anchors = np.array([2, 4])
    fit = local_refine(values, labels, anchors, np.array([[0, 1], [1, 0]]),
                       max_iterations=2, anchors=anchors, save_iterations=(1, 2), stop_when_converged=False)
    assert fit["iterations"] == 2
    for iteration in (1, 2):
        np.testing.assert_array_equal(fit["snapshots"][iteration]["assignments"][anchors], [0, 1])
    assert fit["snapshots"][2]["objective"] <= fit["snapshots"][1]["objective"]
    unconstrained = local_refine(values, labels, anchors, np.array([[0, 1], [1, 0]]), max_iterations=2)
    assert unconstrained["assignments"][2] == 1


def test_second_iteration_does_not_propagate_to_new_clusters_neighbors():
    values = np.array([[-10.], [0.], [4.8], [0.], [5.], [5.1], [6.], [6.1], [6.2]])
    labels = np.repeat([0, 1, 2], 3)
    medoids = np.array([0, 3, 6])
    neighbors = np.array([[0, 1], [1, 2], [2, 1]])
    fit = local_refine(values, labels, medoids, neighbors, max_iterations=2,
                       anchors=medoids, save_iterations=(1, 2), stop_when_converged=False)
    assert fit["snapshots"][1]["assignments"][2] == 1
    assert fit["snapshots"][2]["assignments"][2] == 1
    # After A -> B, C becomes closer, but C was never in this token's A graph.
    final = values[fit["medoids"]]
    assert abs(values[2, 0]-final[2, 0]) < abs(values[2, 0]-final[1, 0])


def test_comparison_saves_both_iterations_and_uses_mean_distortion(tmp_path):
    from scripts.compare_value_distortion import compare_case
    values = np.arange(18, dtype=np.float64)[:, None]
    labels = np.repeat(np.arange(9), 2)
    medoids = np.arange(0, 18, 2)
    result = compare_case(values, values, (labels, medoids), (labels, medoids), tmp_path)
    assert result["global_distortion"] == .5
    assert result["frozen_distortion"] == .5
    for q in (4, 8):
        for iteration in (1, 2):
            with np.load(tmp_path/f"local_q{q}_iter{iteration}.npz") as fit:
                np.testing.assert_array_equal(fit["assignments"][medoids], np.arange(9))


def test_head_summary_gap_is_ratio_of_means():
    from scripts.compare_value_distortion import summarize
    rows = []
    for reference, frozen in ((1., 2.), (3., 3.)):
        row = {"source_layer": 0, "mode": "matched_head_mean", "global_distortion": reference, "frozen_distortion": frozen}
        for q in (4, 8):
            for iteration in (1, 2):
                prefix = f"local_q{q}_iter{iteration}"
                row[prefix+"_distortion"] = frozen
                row[prefix+"_gap_pct"] = (frozen/reference-1)*100
        rows.append(row)
    result = summarize(rows)[0]
    assert result["frozen_gap_pct"] == 25.
    assert result["local_q4_iter2_gap_pct"] == 25.
    assert result["local_q4_iter2_mean_individual_gap_pct"] == 50.
