"""Utilities for generating camera rays for rendering scenes"""

from __future__ import annotations
import numpy as np
import torch

from nerf_sandbox.source.utils.torch_utils import to_torch


@torch.no_grad()
def get_camera_rays(
    image_h: int,
    image_w: int,
    intrinsic_matrix: np.ndarray | torch.Tensor,      # K: (3,3)
    transform_camera_to_world: np.ndarray | torch.Tensor,  # c2w: (3,4) or (4,4)
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    convention: str = "opengl",
    pixel_center: bool = False,
    as_ndc: bool = False,
    near_plane: float = 1.0,
    pixels_xy: np.ndarray | torch.Tensor | None = None,
):
    """
    Returns:
        rays_o_world, rays_d_world_unit, rays_d_world_norm,
        rays_o_marching, rays_d_marching_unit, rays_d_marching_norm

    """
    dev = torch.device(device) if device is not None else None
    K   = torch.as_tensor(intrinsic_matrix, device=dev, dtype=dtype)
    c2w = torch.as_tensor(transform_camera_to_world, device=dev, dtype=dtype)

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    if c2w.shape[-2:] not in {(3,4), (4,4)}:
        raise ValueError(f"c2w must be (3,4) or (4,4), got {c2w.shape}")

    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]

    # ---- pixel grid ----
    if pixels_xy is None:
        ys, xs = torch.meshgrid(
            torch.arange(image_h, device=dev, dtype=dtype),
            torch.arange(image_w, device=dev, dtype=dtype),
            indexing="ij",  # row-major: y first, x second
        )
        if pixel_center:
            xs = xs + 0.5
            ys = ys + 0.5
        x_img = xs.reshape(-1)  # (H*W,)
        y_img = ys.reshape(-1)  # (H*W,)
    else:
        px = torch.as_tensor(pixels_xy, device=dev, dtype=dtype)  # (N,2) [x,y]
        if pixel_center:
            px = px + 0.5
        x_img = px[..., 0].reshape(-1)
        y_img = px[..., 1].reshape(-1)

    # ---- intrinsics ----
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (x_img - cx) / fx
    y_cam = (y_img - cy) / fy

    conv = (convention or "opengl").lower()
    if conv in ("opengl", "blender", "nerf"):
        dirs_cam = torch.stack([x_cam, -y_cam, -torch.ones_like(x_cam)], dim=-1)
    elif conv in ("opencv", "colmap"):
        dirs_cam = torch.stack([x_cam,  y_cam,  torch.ones_like(x_cam)],  dim=-1)
    elif conv in ("pytorch3d", "p3d"):
        dirs_cam = torch.stack([x_cam, -y_cam,  torch.ones_like(x_cam)],  dim=-1)
    else:
        raise ValueError(f"Unknown convention '{convention}'")

    # ---- WORLD rays ----
    rays_d_world_raw  = (dirs_cam @ R.transpose(-1, -2))                   # (N,3)
    rays_d_world_norm = rays_d_world_raw.norm(dim=-1, keepdim=True)        # (N,1)
    rays_d_world_unit = rays_d_world_raw / (rays_d_world_norm + 1e-9)
    rays_o_world      = t.expand_as(rays_d_world_raw)                       # (N,3)

    # ---- MARCHING rays ----
    if not as_ndc:
        # March in WORLD
        rays_o_marching         = rays_o_world
        rays_d_marching_unit    = rays_d_world_unit
        rays_d_marching_norm    = rays_d_world_norm     # (N,1)

    else:
        # --- March in NDC to match nerf-pytorch exactly ---
        # Use WORLD-frame rays directly (not camera-frame), and a scalar focal.
        # This mirrors: ndc_rays(H, W, focal, near, rays_o, rays_d) from nerf-pytorch.

        rays_o_w = rays_o_world              # (N,3) world
        rays_d_w = rays_d_world_raw          # (N,3) world (NOT unit)

        focal = K[0, 0]                      # scalar focal like official (fx)
        sx = 2.0 * focal / float(image_w)
        sy = 2.0 * focal / float(image_h)

        # Shift origins so they intersect the near plane in WORLD-Z
        # t = -(near + o_z) / d_z  ;  o' = o + t d
        oz = rays_o_w[..., 2]
        dz = rays_d_w[..., 2]
        t_ndc = -(near_plane + oz) / (dz + 1e-9)
        o_w = rays_o_w + t_ndc[..., None] * rays_d_w

        # NDC origin
        o0 = -sx * (o_w[..., 0] / (o_w[..., 2] + 1e-9))
        o1 = -sy * (o_w[..., 1] / (o_w[..., 2] + 1e-9))
        o2 =  1.0 + 2.0 * near_plane / (o_w[..., 2] + 1e-9)

        # NDC direction (NOT unit)
        d0 = -sx * ((rays_d_w[..., 0] / (rays_d_w[..., 2] + 1e-9))
                    - (o_w[..., 0] / (o_w[..., 2] + 1e-9)))
        d1 = -sy * ((rays_d_w[..., 1] / (rays_d_w[..., 2] + 1e-9))
                    - (o_w[..., 1] / (o_w[..., 2] + 1e-9)))
        d2 = -2.0 * near_plane / (o_w[..., 2] + 1e-9)

        rays_o_marching      = torch.stack([o0, o1, o2], dim=-1)
        rays_d_ndc_raw       = torch.stack([d0, d1, d2], dim=-1)
        rays_d_marching_norm = rays_d_ndc_raw.norm(dim=-1, keepdim=True)        # used for Δ metric scaling
        rays_d_marching_unit = torch.nn.functional.normalize(rays_d_ndc_raw, dim=-1)  # used for viewdirs


    return (
        rays_o_world,            # (N,3)
        rays_d_world_unit,       # (N,3)
        rays_d_world_norm,       # (N,1)
        rays_o_marching,         # (N,3)
        rays_d_marching_unit,    # (N,3)
        rays_d_marching_norm,    # (N,1)
    )

