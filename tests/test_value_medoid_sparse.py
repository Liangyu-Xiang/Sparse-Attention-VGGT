import torch
import torch.nn.functional as F
import numpy as np

from scripts.run_value_medoid_sparse import token_indices, mean_patch_values, selected_attention


def test_frame_major_patch_special_layout():
    patches, specials = token_indices(2, 3, 2, "cpu")
    assert patches.tolist() == [2, 3, 4, 7, 8, 9]
    assert specials.tolist() == [0, 1, 5, 6]


def test_raw_head_average_excludes_special_values():
    v = torch.arange(2*10*3, dtype=torch.float32).reshape(1, 2, 10, 3)
    patches, _ = token_indices(2, 3, 2, "cpu")
    result = mean_patch_values(v, patches)
    torch.testing.assert_close(result, (v[0, 0, patches] + v[0, 1, patches])/2)


def test_sparse_output_matches_explicit_mask_and_keeps_specials_dense():
    torch.manual_seed(4)
    q, k, v = [torch.randn(1, 2, 10, 4) for _ in range(3)]
    patches, specials = token_indices(2, 3, 2, "cpu")
    medoids = torch.tensor([1, 4])
    output = selected_attention(q, k, v, patches, specials, medoids)
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[specials] = True
    mask[:, specials] = True
    mask[:, patches[medoids]] = True
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(output, expected)
    full = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(output[:, :, specials], full[:, :, specials])


def test_retaining_all_patch_keys_recovers_dense_attention():
    torch.manual_seed(9)
    q, k, v = [torch.randn(1, 3, 14, 8) for _ in range(3)]
    patches, specials = token_indices(2, 5, 2, "cpu")
    actual = selected_attention(q, k, v, patches, specials, torch.arange(len(patches)))
    torch.testing.assert_close(actual, F.scaled_dot_product_attention(q, k, v))


def test_autocast_handles_rope_promoted_fp32_qk_and_bf16_v():
    torch.manual_seed(12)
    q,k = [torch.randn(1,2,14,8) for _ in range(2)]
    v = torch.randn(1,2,14,8).bfloat16()
    patches,specials = token_indices(2,5,2,"cpu")
    with torch.autocast("cpu",dtype=torch.bfloat16):
        actual = selected_attention(q,k,v,patches,specials,torch.arange(len(patches)))
        expected = F.scaled_dot_product_attention(q,k,v)
    assert actual.dtype==torch.bfloat16
    torch.testing.assert_close(actual,expected)


def test_online_wrapper_runs_real_pam_in_fp32_under_autocast(tmp_path, monkeypatch):
    import scripts.run_value_medoid_sparse as experiment
    from saf3r.models.official_vggt.layers.attention import Attention
    torch.manual_seed(11)
    original = Attention(dim=8, num_heads=2, qk_norm=True).eval()
    wrapper = experiment.MeanValueMedoidAttention(original, 2, 4, 0, tmp_path).eval()
    raw_distance = experiment.pairwise_distances
    observed = []

    def checked_distance(values):
        result = raw_distance(values)
        observed.append((values.shape, values.dtype, result.dtype))
        return result

    monkeypatch.setattr(experiment, "pairwise_distances", checked_distance)
    x = torch.randn(1, 18, 8)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = wrapper(x)
        m = np.load(tmp_path / "layer_00/clusters.npz")["medoids"]
        assert len(np.unique(m)) == 2
        qkv = original.qkv(x).reshape(1, 18, 3, 2, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = original.q_norm(q), original.k_norm(k)
        patches, specials = token_indices(2, 4, 5, "cpu")
        attended = selected_attention(q, k, v, patches, specials, torch.from_numpy(m))
        expected = original.proj_drop(original.proj(attended.transpose(1, 2).reshape(1, 18, 8)))
    torch.testing.assert_close(output, expected)
    assert observed == [(torch.Size([8, 4]), torch.float32, torch.float32)]
