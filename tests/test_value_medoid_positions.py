from math import comb

import numpy as np

from scripts.analyze_value_medoid_positions import PositionComparison, no_hit_probability


def test_random_no_hit_probability_equals_hypergeometric_count():
    for excluded in range(13):
        expected = comb(12 - excluded, 3) / comb(12, 3) if 12 - excluded >= 3 else 0
        assert np.isclose(no_hit_probability(12, excluded, 3), expected)


def test_random_single_medoid_distance_is_mean_distance_to_all_positions():
    comparison = PositionComparison(2, 3)
    mean_distance, near1, _ = comparison.random_nearest(1)
    np.testing.assert_allclose(mean_distance, comparison.distance.mean(1), atol=1e-12)
    np.testing.assert_allclose(near1, (comparison.distance <= 1).mean(1), atol=1e-12)


def test_spatial_matching_never_crosses_image_domains():
    comparison = PositionComparison(2, 3)
    a, b = np.array([0, 11]), np.array([5, 6])
    result = comparison.compare(a, b)
    assert result["exact_overlap"] == 0
    assert np.isclose(result["nearest_distance_patches"], np.sqrt(5))
    identical = comparison.compare(a, a)
    assert identical["exact_overlap"] == 1
    assert identical["nearest_distance_patches"] == 0
