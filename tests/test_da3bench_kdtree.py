import numpy as np
from scipy.spatial import KDTree

from evaluation.benchmarks.da3.bench.utils import nn_correspondance


def test_parallel_ckdtree_matches_legacy_kdtree(monkeypatch):
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(257, 3))
    queries = rng.normal(size=(113, 3))
    expected, _ = KDTree(reference).query(queries)

    monkeypatch.setenv("SAF3R_KDTREE_WORKERS", "2")
    actual = nn_correspondance(reference, queries)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
