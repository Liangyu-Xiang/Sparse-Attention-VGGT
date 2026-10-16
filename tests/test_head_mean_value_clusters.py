import numpy as np
import pytest

from scripts.analyze_head_mean_value_clusters import average_heads


def test_average_uses_head_axis_and_float32_accumulation():
    raw = np.arange(16*7*4,dtype=np.float32).reshape(16,7,4).astype(np.float16)
    mean = average_heads(raw)
    assert mean.shape == (7,4)
    assert mean.dtype == np.float32
    np.testing.assert_array_equal(mean,raw.astype(np.float32).sum(axis=0)/16)
    np.testing.assert_array_equal(mean,average_heads(raw[::-1]))


def test_head_averaging_can_cancel_opposing_head_features():
    raw = np.array([[[2.,1.],[3.,4.]],[[-2.,-1.],[-3.,-4.]]])
    np.testing.assert_array_equal(average_heads(raw),np.zeros((2,2)))


def test_bad_input_is_rejected():
    for raw in (np.ones((7,4)), np.ones((0,7,4)), np.full((2,7,4),np.nan)):
        with pytest.raises(ValueError):
            average_heads(raw)
