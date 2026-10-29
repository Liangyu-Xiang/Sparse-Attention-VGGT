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
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/models/aggregator.py", line 261, in forward
    tokens, global_idx, global_intermediates = self._process_global_attention(
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/models/aggregator.py", line 317, in _process_global_attention
    tokens = self.global_blocks[global_idx](tokens, pos=pos)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/layers/attention.py", line 60, in forward
    q = self.rope(q, pos)
        ^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/models/official_vggt/layers/rope.py", line 188, in forward
    return torch.cat((vertical_features, horizontal_features), dim=-1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.33 GiB. GPU 0 has a total capacity of 31.36 GiB of which 1.08 GiB is free. Including non-PyTorch memory, this process has 30.27 GiB memory in use. Of the allocated memory 25.42 GiB is allocated by PyTorch, and 4.23 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
