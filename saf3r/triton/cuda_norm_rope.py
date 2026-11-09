import ctypes
from pathlib import Path

import torch

_launch = ctypes.CDLL(str(Path(__file__).with_name('norm_rope.so'))).launch_norm_rope
_launch.argtypes = ([ctypes.c_void_p] * 13 + [ctypes.c_int] * 2 +
                    [ctypes.c_float] * 2 + [ctypes.c_int] * 6 + [ctypes.c_void_p])
_launch.restype = ctypes.c_int


def apply_cuda(q,k,v,pos,rope,qn,kn,*,position_count=None):
    h,n,d=q.shape
    assert d==64 and all(t.dtype==torch.bfloat16 for t in (q,k,v,qn.weight,qn.bias,kn.weight,kn.bias))
    pos=pos.contiguous()
    if position_count is None:
        position_count=int(pos.max())+1
    cos,sin=rope._compute_frequency_components(32,position_count,q.device,torch.float32)
    out=tuple(torch.empty((h,n,d),device=q.device,dtype=q.dtype) for _ in range(3))
    tensors=(q,k,v,qn.weight,qn.bias,kn.weight,kn.bias,pos,cos,sin,*out)
    result=_launch(*(t.data_ptr() for t in tensors),n,h,qn.eps,kn.eps,
                   q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
                   torch.cuda.current_stream(q.device).cuda_stream)
    if result:
        raise RuntimeError(f'CUDA norm/RoPE launch failed with status {result}')
    return out
