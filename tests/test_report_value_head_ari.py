import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from scripts.report_value_head_ari import all_head_ari, match_next_to_previous, adjacent_layer_pairs


def test_sparse_ari_matches_sklearn_for_large_and_small_cluster_counts():
    rng = np.random.default_rng(139)
    for k,n in ((5,71),(1036,10360)):
        a = rng.integers(0,k,(3,n))
        b = rng.integers(0,k,(4,n))
        actual = all_head_ari(a,b)
        expected = [[adjusted_rand_score(x,y) for y in b] for x in a]
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


def test_sparse_ari_is_invariant_to_cluster_relabeling_and_handles_degenerate_cases():
    a = np.array([[0,0,1,1,2,2],[0,0,0,0,0,0],[0,1,2,3,4,5]])
    b = a[:,::-1]
    actual = all_head_ari(a,b)
    np.testing.assert_allclose(actual,[[adjusted_rand_score(x,y) for y in b]for x in a],atol=1e-12)
    np.testing.assert_allclose(all_head_ari(a,101+7*b),actual,atol=1e-12)
    np.testing.assert_array_equal(all_head_ari(np.zeros((2,1),int),np.ones((3,1),int)),np.ones((2,3)))


def test_sparse_ari_rejects_inconsistent_patch_domains_and_invalid_labels():
    for a,b in ((np.zeros((2,4),int),np.zeros((2,5),int)),
                (np.full((2,4),-1),np.zeros((2,4),int)),
                (np.zeros((2,4)),np.zeros((2,4),int))):
        with pytest.raises(ValueError):
            all_head_ari(a,b)


def test_current_heads_independently_choose_previous_heads_not_rows_or_diagonal():
    matrix = np.array([[.1,.9,.2],[.8,.7,.6]])
    previous,scores = match_next_to_previous(matrix)
    np.testing.assert_array_equal(previous,[1,0,1])
    np.testing.assert_allclose(scores,[.8,.9,.6])
    # Both current heads 0 and 2 reuse previous head 1; no bijection enforced.
    assert len(previous)==matrix.shape[1]


def test_invalid_matching_matrices_are_rejected():
    for matrix in ([],np.zeros((0,2)),np.zeros((2,0)),np.full((2,2),np.nan)):
        with pytest.raises(ValueError):
            match_next_to_previous(matrix)


def test_first_ten_global_layers_include_exactly_nine_adjacent_pairs():
    pairs = adjacent_layer_pairs(10)
    assert pairs==[(i,i+1)for i in range(9)]
    assert pairs[-1]==(8,9)  # Natural layer 9 -> 10, not code layer 10.
    assert len(pairs)*16*16==2304
    for invalid in (0,1,25):
        with pytest.raises(ValueError):
            adjacent_layer_pairs(invalid)
