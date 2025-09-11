"""Utilities for generating camera rays for rendering scenes"""

import numpy as np
import torch

from nerf_experiments.source.utils.torch_utils import to_torch

def get_camera_rays(
    image_h: int,
    image_w: int,
    intrinsic_matrix: np.ndarray | torch.Tensor,
    transform_camera_to_world: np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    dtype: np.dtype | torch.dtype = torch.float32,
    normalize_dirs: bool = True,
    as_ndc: bool = False,
    near_plane: float = 1.0,
    pixels_xy: np.ndarray | torch.Tensor | None = None,
    convention: str = "opengl"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs camera rays in the world frame for each pixel in an image, or a specific
    set of pixels. Can return rays with the following coordinate system convention.

    Conventions (camera-space axes):
      - 'opengl'/'blender'/'nerf': +X right, +Y up, camera looks along -Z
      - 'opencv'/'colmap'        : +X right, +Y down, camera looks along +Z
      - 'pytorch3d'/'p3d'        : +X right, +Y up, camera looks along +Z

    Parameters
    ----------
    image_h : int
        Height of the image in pixels
    image_w : int
        Width of the image in pixels
    intrinsic_matrix : np.ndarray | torch.Tensor
        Camera intrinsic matrix with shape (3, 3).
    transform_camera_to_world : np.ndarray | torch.Tensor
        Transformation matrix from the camera's local frame to the world frame.
        Can be shape (3, 4) or (4, 4) with the first three rows being [R | t].
    device : str | torch.device, optional
        The device to move the tensors of generated rays to. Defaults to None.
    dtype : np.dtype | torch.dtype, optional
        The data type to use for the generated rays. Default is torch.float32
    normalize_dirs : bool, optional
        A flag to convert the ray direction vectors to unit vectors. Default is
        True.
    as_ndc : bool, optional
        A flag to return the rays in Normalized Device Coordinates (NDC). Default
        is False
    near_plane : float, optional
        The near plane distance for the NDC transformation. Default is 1.0
    pixels_xy : np.ndarray | torch.Tensor, optional
        A set of image pixels to generate rays for. If not provided, rays are
        generated for the whole image. Defaults to None
    convention : str, optional
        See above. Default 'opengl' (Blender/vanilla NeRF).

    Returns
    -------
    ray_origins_world : torch.Tensor
        Tensor of shape (N, 3) with dtype and device specified by the inputs
        representing the origins of each ray in the world frame or NDC coordinates.
        N is equal to the product of the input image width and height if pixels_xy
        is not provided, otherwise len(pixels_xy).
    ray_directions_world : torch.Tensor
        Tensor of shape (N, 3) with dtype and device specified by the inputs
        representing the directions of each ray in the world frame or NDC coordinates.
        N is equal to the product of the input image width and height if pixels_xy
        is not provided, otherwise len(pixels_xy). The vectors are normalized if the
        input `normalize_dirs` flag is True.
    """

    # Standardize the input arrays
    intrinsic_matrix = to_torch(intrinsic_matrix, device=device, dtype=dtype)
    transform_camera_to_world = to_torch(transform_camera_to_world, device=device, dtype=dtype)

    # Validate shapes of intrinsic and transform matrices
    if intrinsic_matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsic matrix shape (3, 3), got {intrinsic_matrix.shape}")

    sh = transform_camera_to_world.shape[-2:]
    if sh not in {(4, 4), (3, 4)}:
        raise ValueError(
            "Expected camera-to-world shape (4, 4) or (3, 4), "
            f"got {transform_camera_to_world.shape}"
        )

    # Extract rotation (R) and translation (t) from the input transform matrix
    rotation = transform_camera_to_world[..., :3, :3]
    translation = transform_camera_to_world[..., :3, 3]

    # Create a grid of pixel indexing (xy indexing -> i = x, j = y)
    if pixels_xy is not None:
        pixels_xy = to_torch(pixels_xy, device=device, dtype=dtype)
        x_img = pixels_xy[:, 0]
        y_img = pixels_xy[:, 1]
    else:
        i, j = torch.meshgrid(
            torch.arange(image_w, device=intrinsic_matrix.device, dtype=intrinsic_matrix.dtype),
            torch.arange(image_h, device=intrinsic_matrix.device, dtype=intrinsic_matrix.dtype),
            indexing="xy"
        )  # each shape [W, H]
        x_img = i.reshape(-1)  # [N]
        y_img = j.reshape(-1)  # [N]

    # Directions in camera frame (pinhole model)
    # x_cam = (x_img - c_x)/f_x, y_cam = (y_img - c_y)/f_y, z = 1
    f_x, f_y = intrinsic_matrix[..., 0, 0], intrinsic_matrix[..., 1, 1]
    c_x, c_y = intrinsic_matrix[..., 0, 2], intrinsic_matrix[..., 1, 2]

    # Broadcast f_x, f_y, c_x, c_y if they have extra leading dims
    x_cam = (x_img - c_x) / f_x
    y_cam = (y_img - c_y) / f_y

    conv = (convention or "opengl").lower()
    if conv in ("opengl", "blender", "nerf"):
        # +X right, +Y up, forward -Z  → flip image y, z = -1
        ray_directions_camera = torch.stack([x_cam, -y_cam, -torch.ones_like(x_cam)], dim=-1)
    elif conv in ("opencv", "colmap"):
        # +X right, +Y down, forward +Z → no flip, z = +1
        ray_directions_camera = torch.stack([x_cam,  y_cam,  torch.ones_like(x_cam)], dim=-1)
    elif conv in ("pytorch3d", "p3d"):
        # +X right, +Y up, forward +Z → flip image y, z = +1
        ray_directions_camera = torch.stack([x_cam, -y_cam,  torch.ones_like(x_cam)], dim=-1)
    else:
        raise ValueError(
            f"Unknown camera convention '{convention}'. "
            "Choose from: 'opengl'|'blender'|'nerf', 'opencv'|'colmap', 'pytorch3d'|'p3d'."
        )

    # Rotate directions to world frame: dirs_world = dirs_cam @ R^T
    ray_directions_world = (
        ray_directions_camera[..., None, :] @ rotation.transpose(-1, -2)
    ).squeeze(-2)  # shape (N, 3)

    # Skip normalization if returning rays in NDC coordinates
    if normalize_dirs and not as_ndc:
        ray_directions_world = (
            ray_directions_world / (ray_directions_world.norm(dim=-1, keepdim=True) + 1e-9)
        )

    # Ray origins: camera position in world for every pixel
    ray_origins_world = translation.expand_as(ray_directions_world)  # shape (N, 3)

    # If not transforming to NDC coordinates, return
    if not as_ndc:
        return ray_origins_world.contiguous(), ray_directions_world.contiguous()

    #----- Normalized Device Coordinate transform (forward-facing convention) -----
    # Transforms the scene into a [-1,1] cube with the near plane at z=0 and far plane at z=1.
    # Compute translation to bring the ray origins to the near plane
    translation_ndc = -(near_plane + ray_origins_world[..., 2]) / ray_directions_world[..., 2]
    ray_origins_world = ray_origins_world + translation_ndc[..., None] * ray_directions_world

    # Compute ray origin components via projection
    ray_origins_ndc_x = -1.0 / (image_w / (2.0 * f_x)) * ray_origins_world[..., 0] / ray_origins_world[..., 2]
    ray_origins_ndc_y = -1.0 / (image_h / (2.0 * f_y)) * ray_origins_world[..., 1] / ray_origins_world[..., 2]
    ray_origins_ndc_z = 1.0 + 2.0 * near_plane / ray_origins_world[..., 2]

    # Compute ray direction components via projection
    ray_directions_ndc_x = -1.0 / (image_w / (2.0 * f_x)) * (
        ray_directions_world[..., 0] / ray_directions_world[..., 2]
        - ray_origins_world[..., 0] / ray_origins_world[..., 2]
    )
    ray_directions_ndc_y = -1.0 / (image_h / (2.0 * f_y)) * (
        ray_directions_world[..., 1] / ray_directions_world[..., 2]
        - ray_origins_world[..., 1] / ray_origins_world[..., 2]
    )
    ray_directions_ndc_z = -2.0 * near_plane / ray_origins_world[..., 2]

    # Stack components and return
    ray_origins_ndc = torch.stack(
        [ray_origins_ndc_x, ray_origins_ndc_y, ray_origins_ndc_z], -1
    )
    ray_directions_ndc = torch.stack(
        [ray_directions_ndc_x, ray_directions_ndc_y, ray_directions_ndc_z], -1
    )

    return ray_origins_ndc.contiguous(), ray_directions_ndc.contiguous()
