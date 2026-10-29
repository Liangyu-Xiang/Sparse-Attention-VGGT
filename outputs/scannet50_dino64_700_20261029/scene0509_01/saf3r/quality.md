| Scene/method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0509_01/saf3r | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

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
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/patch/patch_vggt.py", line 345, in forward
    est_topk_idx = compute_qk_topk_indices_batched(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/saf3r/patch/utils.py", line 140, in compute_qk_topk_indices_batched
    topk_idx_out = torch.empty((B, Nq, out_width), device=device, dtype=torch.int32)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 121.03 GiB. GPU 0 has a total capacity of 31.36 GiB of which 7.35 GiB is free. Including non-PyTorch memory, this process has 24.00 GiB memory in use. Of the allocated memory 18.13 GiB is allocated by PyTorch, and 5.26 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
