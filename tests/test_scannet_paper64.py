import numpy as np
import pytest
from scripts.evaluate_scannet_paper64 import distance_metrics,PROTOCOL

def test_distances_are_unsquared_and_cd_is_mean():
    d=distance_metrics(np.array([.02,.1]),np.array([.04,.2]))
    assert d['acc']==pytest.approx(.06)
    assert d['comp']==pytest.approx(.12)
    assert d['overall']==pytest.approx(.09)
    assert d['fastvggt_sum_cd']==pytest.approx(.18)
    assert d['fscore']==pytest.approx(.5)

def test_clipping_does_not_turn_far_points_into_f1_inliers():
    d=distance_metrics(np.array([10.]),np.array([20.]))
    assert d['overall']==.5
    assert d['fscore']==0

def test_empty_cloud_is_failure_not_perfect_metric():
    with pytest.raises(ValueError):distance_metrics(np.array([]),np.array([0.]))

def test_protocol_has_no_gt_camera_fusion_and_discloses_k():
    assert 'no GT-camera fusion' in PROTOCOL['reconstruction']
    assert 'not unpublished per-scene K reproduction' in PROTOCOL['parameter_provenance']
