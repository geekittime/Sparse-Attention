"""
TVM FFI Bindings for DeltaNet Recurrent CUDA Kernel.

The actual kernel function is exported from kernel.cu via:
    TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, DeltaNetRecurrentForward)

This binding.py registers a Python-side wrapper using tvm.ffi.register_func
so the FlashInfer-Bench framework can discover and call the kernel.

DPS (Destination Passing Style): output and new_state are pre-allocated
and passed as the last two arguments.
"""

import ctypes
from tvm.ffi import register_func


@register_func("flashinfer.kernel")
def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    DeltaNet recurrent step - TVM FFI binding.

    Parameters
    ----------
    q : TensorView [B, T, H, K] bf16
    k : TensorView [B, T, H, K] bf16
    v : TensorView [B, T, HV, V] bf16
    state : TensorView [B, HV, V, K] f32
    A_log : TensorView [HV] f32
    a : TensorView [B, T, HV] bf16
    dt_bias : TensorView [HV] f32
    b : TensorView [B, T, HV] bf16
    scale : float
    output : TensorView [B, T, HV, V] bf16 (DPS pre-allocated)
    new_state : TensorView [B, HV, V, K] f32 (DPS pre-allocated)
    """
    # The C++ side TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, ...) handles the
    # actual dispatch. This Python registration serves as the framework's
    # entry point for kernel discovery.
    pass
