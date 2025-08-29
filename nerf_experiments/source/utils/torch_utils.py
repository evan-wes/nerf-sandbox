"""General utilities for PyTorch and NumPy interactions"""

from __future__ import annotations
import numpy as np
import torch

# Map any NumPy dtype (or NumPy dtype *type*) to a torch.dtype
_NUMPY2TORCH = {
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int8'):    torch.int8,
    np.dtype('int16'):   torch.int16,
    np.dtype('int32'):   torch.int32,
    np.dtype('int64'):   torch.int64,
    np.dtype('uint8'):   torch.uint8,
    np.dtype('bool'):    torch.bool,
}

def _normalize_torch_dtype(dt: np.dtype | torch.dtype | type | None) -> torch.dtype | None:
    if dt is None:
        return None
    if isinstance(dt, torch.dtype):
        return dt
    # Handles np.dtype objects and np.* dtype *types* (e.g., np.float32)
    try:
        return _NUMPY2TORCH[np.dtype(dt)]
    except Exception:
        raise TypeError(f"Unsupported dtype: {dt!r}")

def to_torch(
    x: np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    dtype: np.dtype | torch.dtype | type = torch.float32,
) -> torch.Tensor:
    """
    Convert input to a torch.Tensor, optionally moving to `device` and (for floating tensors)
    casting to `dtype`. Uses zero-copy when possible.

    Notes:
    - Uses torch.as_tensor(...) so NumPy inputs share memory when writable.
    - Only casts dtype if the *input tensor* is floating, matching the original behavior.
    """
    t = torch.as_tensor(x)  # works for Tensor, NumPy, lists, scalars

    target_dtype = _normalize_torch_dtype(dtype)
    target_device = torch.device(device) if device is not None else None

    to_kwargs = {}
    if target_device is not None and t.device != target_device:
        to_kwargs["device"] = target_device

    if target_dtype is not None and t.dtype != target_dtype and t.is_floating_point():
        to_kwargs["dtype"] = target_dtype

    if to_kwargs:
        t = t.to(**to_kwargs)

    return t
