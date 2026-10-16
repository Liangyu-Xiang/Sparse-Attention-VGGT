import numpy as np

from scripts.visualize_best_value_head_pair import resize_cluster_ids, distinct_palette, align_target_colours


def test_resize_does_not_truncate_cluster_ids_above_255():
    labels=np.array([[0,255],[256,1035]])
    pixels=resize_cluster_ids(labels,(6,6))
    np.testing.assert_array_equal(pixels,np.repeat(np.repeat(labels,3,0),3,1))
    assert pixels.max()==1035


def test_palette_has_distinct_deterministic_colours_for_all_1036_clusters():
    a=distinct_palette(1036)
    assert a.shape==(1036,3) and len(np.unique(a,axis=0))==1036
    np.testing.assert_array_equal(a,distinct_palette(1036))


def test_colour_alignment_does_not_assume_matching_cluster_ids():
    source=np.array([0,0,1,1,2,2]);target=np.array([2,2,0,0,1,1])
    lookup,counts=align_target_colours(source,target,3)
    np.testing.assert_array_equal(lookup[target],source)
    assert counts.sum()==6
