"""General utilities for PyTorch and NumPy interactions"""

import numpy as np
import torch

def to_torch(
    x: np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    dtype: np.dtype | torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Returns a torch.Tensor with the specified datatype located on the specified device.
    Accepts numpy arrays or torch tensors.

    Parameters
    ----------
    x : np.ndarray | torch.Tensor
        Input array or tensor to convert
    device : str | torch.device, optional
        The device to move the tensor to, if provided and different from the tensor's
        current device. Defaults to None.
    dtype : np.dtype | torch.dtype, optional
        The data type to convert, if different from the input data type. Default is
        torch.float32

    Returns
    -------
    torch.Tensor :
        The tensor with the specified datatype and device location
    """
    if isinstance(x, torch.Tensor):
        t = x
        # Upgrade dtype if needed (e.g., float64 -> float32)
        if dtype is not None and t.dtype != dtype and t.is_floating_point():
            t = t.to(dtype)
        if device is not None and t.device != torch.device(device):
            t = t.to(device)
        return t
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None and t.dtype != dtype and t.is_floating_point():
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    else:
        # Assume Python list/tuple or scalar
        t = torch.tensor(x, dtype=dtype, device=device)
        return t