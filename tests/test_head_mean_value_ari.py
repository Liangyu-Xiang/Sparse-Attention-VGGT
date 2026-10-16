import json
import sys

import numpy as np
import pytest

from scripts import analyze_head_mean_value_ari as analysis


def test_resume_shard_validates_cached_fit_without_gpu_or_shared_report(tmp_path,monkeypatch):
    values = np.zeros((16,20,64),np.float32)
    values[:,10:] = 1
    np.savez(tmp_path/"values.npz",layer_0=values)
    (tmp_path/"metadata.json").write_text(json.dumps({"values_per_head":20,"frame_indices":list(range(10))}))
    directory = tmp_path/"head_mean_10pct_ari/layer_00"
    directory.mkdir(parents=True)
    labels = np.repeat([0,1],10)
    np.savez(directory/"clusters.npz",medoids=np.array([0,10]),assignments=labels,
             distances=np.zeros(20,np.float32),cluster_sizes=np.array([10,10]))
    (directory/"clustering.json").write_text(json.dumps({"converged":True,"objective":0.,"swaps":0,"starts":[]}))
    def unexpected(*args,**kwargs):
        raise AssertionError("Cached shard must not recompute PAM or allocate a GPU distance matrix")
    monkeypatch.setattr(analysis,"pam",unexpected)
    monkeypatch.setattr(analysis,"pairwise_distances",unexpected)
    monkeypatch.setattr(sys,"argv",["analysis","--input-dir",str(tmp_path),"--device","cpu",
                                   "--num-layers","10","--resume","--cluster-only","--fit-layers","0"])
    analysis.main()
    np.testing.assert_array_equal(np.load(directory/"clusters.npz")["assignments"],labels)
    assert not (directory.parent/"summary.json").exists()
    assert not (directory.parent/"mean_values.npz").exists()


def test_partial_layer_shard_cannot_publish_incomplete_report(monkeypatch):
    monkeypatch.setattr(sys,"argv",["analysis","--num-layers","10","--fit-layers","3"])
    with pytest.raises(SystemExit) as caught:
        analysis.main()
    assert caught.value.code==2
