import numpy as np

from scripts.analyze_value_medoid_neighborhoods import neighborhood_masks, point_masks, scores, selections
from scripts.analyze_value_medoid_positions import PositionComparison


def test_neighborhood_matrix_matches_direct_same_frame_distance_counts():
    height, width = 3, 4
    a_sets = [np.array([0, 5, 12, 17]), np.array([3, 10, 15, 20])]
    b_sets = [np.array([1, 7, 13, 18]), np.array([4, 11, 16, 23])]
    a, b = point_masks(a_sets, 12), point_masks(b_sets, 12)
    comparison = PositionComparison(height, width)
    for radius, key in ((0, "exact_overlap"), (1, "within_1_patch"), (2, "within_2_patches")):
        actual = scores(a, b, neighborhood_masks(a, height, width, radius), neighborhood_masks(b, height, width, radius))
        expected = np.array([[comparison.compare(x, y)[key] for y in b_sets] for x in a_sets])
        np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_neighborhoods_do_not_cross_frame_or_head_boundaries():
    mask = np.zeros((2, 24), dtype=np.float32)
    mask[0, 11] = 1
    expanded = neighborhood_masks(mask, 3, 4, 2)
    assert expanded[0, :12].sum() > 0
    assert expanded[0, 12:].sum() == 0
    assert expanded[1].sum() == 0


def test_head_matching_is_invariant_to_head_label_permutations():
    matrix = np.array([[0.2, 0.8, 0.5], [0.3, 0.9, 0.4], [0.7, 0.4, 0.1]])
    permuted = matrix[[2, 0, 1]][:, [1, 2, 0]]
    for strategy in ("row_best", "one_to_one", "all_pairs"):
        ii, jj = selections(matrix)[strategy]
        pi, pj = selections(permuted)[strategy]
        np.testing.assert_allclose(matrix[ii, jj].mean(), permuted[pi, pj].mean())
    ii, jj = selections(matrix)["row_best"]
    assert len(set(jj)) < len(jj)  # independent source heads may share a target
