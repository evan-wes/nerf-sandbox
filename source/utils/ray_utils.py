"""Utilities for generating camera rays for rendering scenes"""

import numpy as np
import torch

from nerf_experiments.source.utils.torch_utils import to_torch

def get_camera_rays(
    image_h: int,
    image_w: int,
    intrinsic_matrix: np.ndarray | torch.Tensor,
    transform_camera_to_world: np.ndarray | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs camera rays in the world frame for each pixel in an image.

    Parameters
    ----------
    image_h : int
        Height of the image
    image_w : int
        Width of the image
    intrinsic_matrix : np.ndarray | torch.Tensor

    """
    i, j = torch.meshgrid(
        torch.arange(image_h, dtype=torch.float32),
        torch.arange(image_w, dtype=torch.float32), indexing="xy")
    dirs = torch.stack([
        (i - K[0,2]) / K[0,0],
        (j - K[1,2]) / K[1,1],
        torch.ones_like(i)], -1)           # [H,W,3]
    dirs = (dirs[..., None, :] @ c2w[:3,:3].T).squeeze(-2)
    origins = c2w[:3, 3].expand_as(dirs)
    return origins.reshape(-1,3), dirs.reshape(-1,3)

def get_camera_rays(
    image_h: int,
    image_w: int,
    intrinsic_matrix: np.ndarray | torch.Tensor,           # (3,3) intrinsics; numpy or torch
    transform_camera_to_world: np.ndarray | torch.Tensor,         # (4,4) or (3,4) camera-to-world; numpy or torch
    device: str | torch.device | None = None,
    dtype: np.dtype | torch.dtype = torch.float32,
    normalize_dirs: bool = True,
    as_ndc: bool = False,
    near_plane: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs camera rays in the world frame for each pixel in an image.

    Parameters
    ----------
    image_h : int
        Height of the image in pixels
    image_w : int
        Width of the image in pixels
    intrinsic_matrix : np.ndarray | torch.Tensor
        Camera intrinsic matrix with shape (batch_size, 3, 3).
    transform_camera_to_world : np.ndarray | torch.Tensor
        Transformation matrix from the camera's local frame to the world frame.
        Can be shape (batch_size, 3, 4) or (batch_size, 4, 4) with the first
        three rows being [R | t].
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

    Returns
    -------
    ray_origins_world : torch.Tensor
        Tensor of shape (N, 3) with dtype and device specified by the inputs
        representing the origins of each ray in the world frame or NDC coordinates.
        N is equal to the product of the input image width and height.
    ray_directions_world : torch.Tensor
        Tensor of shape (N, 3) with dtype and device specified by the inputs
        representing the directions of each ray in the world frame or NDC coordinates.
        N is equal to the product of the input image width and height. The vectors are
        normalized if the input `normalize_dirs` flag is True.
    """

    # Standardize the input arrays
    intrinsic_matrix = to_torch(intrinsic_matrix, device=device, dtype=dtype)
    transform_camera_to_world = to_torch(transform_camera_to_world, device=device, dtype=dtype)

    # Validate shapes of intrinsic and transform matrices
    assert (
        intrinsic_matrix.shape[-2:] == (3, 3),
        f"Expected intrinsic matrix shape (batch_size, 3, 3), got {intrinsic_matrix.shape}"
    )
    assert (
        transform_camera_to_world.shape[-2:] in [(4, 4), (3, 4)],
        (
            "Expected camera to world transfrom shape (batch_size, 4, 4) or "
            f"(batch_size, 3, 4), got {transform_camera_to_world.shape}"
        )
    )

    # Extract rotation (R) and translation (t) from the input transform matrix
    rotation = transform_camera_to_world[..., :3, :3]
    translation = transform_camera_to_world[..., :3, 3]

    # Create a grid of pixel indexing (xy indexing -> i = x, j = y)
    i, j = torch.meshgrid(
        torch.arange(image_w, device=intrinsic_matrix.device, dtype=dtype),
        torch.arange(image_h, device=intrinsic_matrix.device, dtype=dtype),
        indexing="xy"
    )  # each [W, H]

    # Directions in camera frame (pinhole model)
    # x = (i - c_x)/f_x, y = (j - c_y)/f_y, z = 1
    f_x, f_y = intrinsic_matrix[..., 0, 0], intrinsic_matrix[..., 1, 1]
    c_x, c_y = intrinsic_matrix[..., 0, 2], intrinsic_matrix[..., 1, 2]

    # Broadcast f_x, f_y, c_x, c_y if they have extra leading dims
    x_cam = (i - c_x) / f_x
    y_cam = (j - c_y) / f_y
    ones  = torch.ones_like(x_cam)

    ray_directions_camera = torch.stack([x_cam, y_cam, ones], dim=-1)  # [W, H, 3]

    # Rotate directions to world frame
    # dirs_world = dirs_cam @ R^T
    ray_directions_world = (ray_directions_camera[..., None, :] @ rotation.transpose(-1, -2)).squeeze(-2)  # [W, H, 3]

    # Skip normalization if returning rays in NDC coordinates
    if normalize_dirs and not as_ndc:
        ray_directions_world = ray_directions_world / (ray_directions_world.norm(dim=-1, keepdim=True) + 1e-9)

    # Ray origins: camera position in world for every pixel
    ray_origins_world = translation.expand_as(ray_directions_world)  # [W, H, 3]

    # Flatten to [H*W, 3] with row-major (y,x) order
    ray_origins_world = ray_origins_world.permute(1, 0, 2).reshape(-1, 3).contiguous()
    ray_directions_world = ray_directions_world.permute(1, 0, 2).reshape(-1, 3).contiguous()

    # If not transforming to NDC coordinates, return
    if not as_ndc:
        return ray_origins_world, ray_directions_world

    # Normalized Device Coordinate transform: Transforms the scene into [-1,1]
    # cube with the near plane at z=0 and far plane at z=1.
    # Compute translation to bring the ray origins to the near plane
    translation_ndc = -(near_plane + ray_directions_world[..., 2])/ray_directions_world[..., 2]
    ray_origins_world = ray_origins_world + translation_ndc[..., None]*ray_directions_world

    # Compute ray origin components via projection
    ray_origins_ndc_x = -1.0/(image_w/(2.0*f_x))*ray_origins_world[..., 0]/ray_origins_world[..., 2]
    ray_origins_ndc_y = -1.0/(image_h/(2.0*f_y))*ray_origins_world[..., 1]/ray_origins_world[..., 2]
    ray_origins_ndc_z = 1.0 + 2.0*near_plane/ray_origins_world[..., 2]

    # Compute ray direction components via projection
    ray_directions_ndc_x = -1.0/(image_w/(2.0*f_x))*(
        ray_directions_world[..., 0]/ray_directions_world[..., 2] - ray_origins_world[..., 0]/ray_origins_world[..., 2]
    )
    ray_directions_ndc_y = -1.0/(image_h/(2.0*f_y))*(
        ray_directions_world[..., 1]/ray_directions_world[..., 2] - ray_origins_world[..., 1]/ray_origins_world[..., 2]
    )
    ray_directions_ndc_z = -2.0*near_plane/ray_origins_world[..., 2]

    # Stack components and return
    ray_origins_ndc = torch.stack([ray_origins_ndc_x, ray_origins_ndc_y, ray_origins_ndc_z], -1)
    ray_directions_ndc = torch.stack([ray_directions_ndc_x, ray_directions_ndc_y, ray_directions_ndc_z], -1)

    return ray_origins_ndc, ray_directions_ndc
