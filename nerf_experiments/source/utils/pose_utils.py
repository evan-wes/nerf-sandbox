"""Contains utilities for generating poses"""

import math
import torch

def look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor = None) -> torch.Tensor:
    """
    Build a camera-to-world (c2w) 4x4 from eye, target, up.
    Convention: +Z forward (from camera), +X right, +Y up in world.
    Returns: [4,4] torch.float32 on the same device as inputs.
    """
    if up is None:
        up = torch.tensor([0., 1., 0.], device=eye.device, dtype=eye.dtype)
    up = up / (up.norm() + 1e-9)
    f = (target - eye)
    f = f / (f.norm() + 1e-9)
    r = torch.cross(f, up, dim=-1)
    r = r / (r.norm() + 1e-9)
    u = torch.cross(r, f, dim=-1)
    # camera looks along +Z in camera frame, but we want columns to be world axes in camera basis
    R = torch.stack([r, u, f], dim=1)  # [3,3], columns = right, up, forward
    T = eye[:, None]                    # [3,1]
    c2w = torch.eye(4, device=eye.device, dtype=eye.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3]  = T.squeeze(-1)
    return c2w
