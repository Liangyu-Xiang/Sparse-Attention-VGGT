| Scene/method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0238_00/dense | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Traceback (most recent call last):
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/scripts/evaluate_scannet50_paper.py", line 52, in run
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
                                                                                ^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/models/vggt.py", line 65, in forward
    aggregated_tokens_list, patch_start_idx = self.aggregator(images)
                                              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/models/aggregator.py", line 257, in forward
    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/models/aggregator.py", line 296, in _process_frame_attention
    tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/layers/block.py", line 105, in forward
    x = x + attn_residual_func(x, pos=pos)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/layers/block.py", line 83, in attn_residual_func
    return self.ls1(self.attn(self.norm1(x), pos=pos))
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/layers/attention.py", line 57, in forward
    q, k = self.q_norm(q), self.k_norm(k)
                           ^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/normalization.py", line 229, in forward
    return F.layer_norm(
           ^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/functional.py", line 2901, in layer_norm
    return torch.layer_norm(
           ^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.99 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.76 GiB is free. Including non-PyTorch memory, this process has 29.59 GiB memory in use. Of the allocated memory 26.65 GiB is allocated by PyTorch, and 2.32 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
