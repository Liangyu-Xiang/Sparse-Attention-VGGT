"""Check PAM swap evaluation independently against explicit enumeration."""

import json
import numpy as np
import torch

from scripts.visualize_vggt_value_medoids import best_swap, pairwise_distances, pam, patch_values_from_tokens, valid_cached_cluster


def test_best_swap_matches_every_explicit_insertion_removal():
    for seed in range(5):
        torch.manual_seed(seed)
        distance = pairwise_distances(torch.randn(17, 5))
        medoids = torch.tensor([0, 5, 11])
        objective = distance[:, medoids].amin(1).sum()
        expected = []
        for removed in range(3):
            for candidate in range(17):
                if candidate in medoids:
                    continue
                proposed = medoids.clone()
                proposed[removed] = candidate
                expected.append(float(distance[:, proposed].amin(1).sum() - objective))
        delta, removed, candidate = best_swap(distance, medoids)
        assert candidate not in medoids
        assert 0 <= removed < len(medoids)
        assert abs(delta - min(expected)) < 2e-5


def test_pam_converges_to_single_swap_local_optimum_with_real_medoids():
    torch.manual_seed(7)
    distance = pairwise_distances(torch.randn(23, 4))
    result = pam(distance, k=4, method="build", seed=0, max_swaps=200)
    assert result["converged"]
    assert len(set(result["medoids"])) == 4
    assert all(0 <= medoid < 23 for medoid in result["medoids"])
    assert all(a > b for a, b in zip(result["objective_history"], result["objective_history"][1:]))
    delta, _, _ = best_swap(distance, torch.tensor(result["medoids"]))
    assert delta >= -result["tolerance"]


def test_identical_values_keep_unique_indices_and_zero_objective():
    distance = pairwise_distances(torch.ones(12, 3))
    for method in ("build", "sample"):
        result = pam(distance, k=3, method=method, seed=0, max_swaps=200)
        assert result["objective"] == 0
        assert len(set(result["medoids"])) == 3
        assert result["converged"]


def test_progress_logging_does_not_change_pam_solution():
    torch.manual_seed(10)
    distance = pairwise_distances(torch.randn(23,4))
    for method in ("build","sample"):
        events = []
        original = pam(distance,4,method,seed=0,max_swaps=200)
        observed = pam(distance,4,method,seed=0,max_swaps=200,progress=events.append)
        np.testing.assert_array_equal(original["medoids"],observed["medoids"])
        assert original["objective_history"]==observed["objective_history"]
        assert events[0]["stage"]=="initializing" and events[-1]["stage"]=="fit_finished"


def test_patch_value_layout_handles_two_and_ten_frames_without_special_tokens():
    for frames in (2,10):
        patches, specials, heads, dim = 6,5,3,4
        raw = torch.arange(frames*(patches+specials)*heads*dim).reshape(frames*(patches+specials),heads,dim)
        result = patch_values_from_tokens(raw,frames,patches,specials)
        assert result.shape==(heads,frames*patches,dim)
        for head in range(heads):
            for frame in range(frames):
                for patch in range(patches):
                    torch.testing.assert_close(result[head,frame*patches+patch],raw[frame*(patches+specials)+specials+patch,head])


def test_resume_rejects_incomplete_unconverged_or_inconsistent_cache(tmp_path):
    arrays,stats_path = tmp_path/"fit.npz",tmp_path/"fit.json"
    stats = {"converged":True,"n":20,"k":2,"requested_retention":.1,"objective":0.}
    stats_path.write_text(json.dumps(stats))
    assert not valid_cached_cluster(arrays,stats_path,20,2,.1)
    labels = np.repeat([0,1],10)
    np.savez(arrays,medoids=np.array([0,10]),assignments=labels,distances=np.zeros(20),cluster_sizes=np.array([10,10]))
    assert valid_cached_cluster(arrays,stats_path,20,2,.1)
    assert not valid_cached_cluster(arrays,stats_path,20,3,.1)
    assert not valid_cached_cluster(arrays,stats_path,20,2,.01)
    stats_path.write_text(json.dumps({**stats,"converged":False}))
    assert not valid_cached_cluster(arrays,stats_path,20,2,.1)
    stats_path.write_text(json.dumps(stats))
    np.savez(arrays,medoids=np.array([0,10]),assignments=labels,distances=np.zeros(20),cluster_sizes=np.array([9,11]))
    assert not valid_cached_cluster(arrays,stats_path,20,2,.1)
