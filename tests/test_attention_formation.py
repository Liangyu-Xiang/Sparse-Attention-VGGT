import numpy as np
import pytest
from scripts.analyze_attention_formation import top_indices,recall,statistics,coarse_recall


def test_nested_topk_and_recall():
    a=np.array([[4.,4.,2.,1.],[1.,2.,3.,4.]])
    np.testing.assert_array_equal(top_indices(a,2),[[0,1],[3,2]])
    target=np.array([[0,2],[3,1]])
    assert recall(top_indices(a,1),target,4)==.5
    assert recall(top_indices(a,3),target,4)==1.


def test_concentration_uniform_and_delta():
    assert statistics(np.ones((4,4))/4)['C']==pytest.approx(0.)
    assert statistics(np.eye(4))['C']==pytest.approx(1.)


def test_coarse_key_coverage_not_group_fraction():
    groups=np.array([0,0,0,1])
    scores=np.array([[3.,1.],[1.,3.]])
    target=np.array([[0,3],[3,1]])
    result=coarse_recall(scores,groups,target,.5)
    assert result['recall']==.5
    assert result['density']==.5  # Query sizes are .75 and .25, not both .5.
    assert result['chance_corrected']==pytest.approx(-1/3)
