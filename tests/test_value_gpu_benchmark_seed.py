import json

import numpy as np
import pytest
import torch

from scripts.benchmark_layer_value_gpu import prepare_seed
from saf3r.patch.value_evolving_attention import LayerValueState


def seed_fixture(tmp_path, n=120, k=24):
    source = tmp_path/"source"
    source.mkdir()
    (source/"clustering.json").write_text(json.dumps({"n": n, "k": k, "converged": True}))
    np.savez(source/"clusters.npz", assignments=np.arange(n)%k, medoids=np.arange(k))
    return source, torch.arange(n*2, dtype=torch.float32).reshape(n, 2)


def test_matching_seed_is_reused_without_global_refit(tmp_path, monkeypatch):
    source, values = seed_fixture(tmp_path)
    def forbidden(*args):
        raise AssertionError("Unexpected PAM initialization")
    monkeypatch.setattr(LayerValueState, "_seed", forbidden)
    cache, stats = prepare_seed(values, source, tmp_path/"fit", .2)
    assert len(cache["medoids"]) == stats["k"] == 24
    assert not (tmp_path/"fit").exists()


def test_smaller_retention_is_refitted_not_truncated_and_then_cached(tmp_path, monkeypatch):
    source, values = seed_fixture(tmp_path)
    calls = []
    def fake(self, actual):
        torch.testing.assert_close(actual, values)
        k = round(self.retention*len(actual))
        calls.append(k)
        # Choose different representatives from the beginning of the old set.
        labels = np.arange(len(actual))%k
        medoids = np.arange(k, 2*k)
        return labels, medoids, {"converged": True, "objective": 1.}
    monkeypatch.setattr(LayerValueState, "_seed", fake)
    cache, stats = prepare_seed(values, source, tmp_path/"fit", .1)
    assert calls == [12] and len(cache["medoids"]) == 12
    np.testing.assert_array_equal(cache["medoids"], np.arange(12, 24))
    assert stats["initialization_seconds_excluded"] >= 0
    again, _ = prepare_seed(values, source, tmp_path/"fit", .1)
    assert calls == [12]
    torch.testing.assert_close(cache["medoids"], again["medoids"])
    with pytest.raises(ValueError, match="different K"):
        prepare_seed(values, source, tmp_path/"fit", .15)


def test_invalid_seed_indices_are_rejected(tmp_path):
    source, values = seed_fixture(tmp_path)
    np.savez(source/"clusters.npz", assignments=np.zeros(120, dtype=int), medoids=np.arange(24))
    with pytest.raises(AssertionError):
        prepare_seed(values, source, tmp_path/"fit", .2)
