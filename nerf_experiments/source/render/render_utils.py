"""Contains utilities for rendering rays"""

import torch

def volume_render_rays(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_depths: torch.Tensor,
    white_bkgd: bool = False,
    eps: float = 1e-10
) -> tuple[torch.Tensor]:
    """
    Uses the MLP output of RGB and volume density at sampled depths along a batch of rays to
    render the composited RGB values, along with other semantically meaningful outputs used
    downstream.

    Parameters
    ----------
    rgb : torch.Tensor
        Output of the MLP representing the RGB value at the sampled points along the rays.
        Has shape (num_rays, num_ray_pts, 3).
    sigma : torch.Tensor
        Output of the MLP representing the volume density at the sampled points along the rays.
        Has shape (num_rays, num_ray_pts).
    z_depths : torch.Tensor
        Sorted depths sampled along the ray where the MLP output is defined. Used as integration
        steps for rendering and defines the bins of the returned weights. Has shape
        (num_rays, num_ray_pts).
    white_bkgd : bool, optional
        Whether to insert a white background. Defaults to False
    eps : float, optional
        A tiny factor to insert when taking a cumulative product to avoid zeros.

    Returns
    -------
    composite_rgb : torch.Tensor
        The rendered color for each ray after integrating the MLP-predicted color and volume
        densities along the rays at the sampled depths. Has shape (num_rays, num_ray_pts, 3).
    weights : torch.Tensor
        Probability of a ray terminating at a volume element. Used for re-sampling new positions
        along the rays. Has shape (num_rays, num_ray_pts).
    accumulated_opacity : torch.Tensor
        The sum of the weights along the rays. Has shape (num_rays).
    depth : torch.Tensor
        The depth of the rendered volume along each ray. Has shape (num_rays).
    """

    # Step 1. Compute integration bins
    deltas = z_depths[..., 1:] - z_depths[..., :-1] # shape (num_rays, num_ray_pts-1)
    # Add a large final bin
    deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[..., :1])], dim=-1)   # shape (num_rays, num_ray_pts)

    # Step 2. Compute opacities at each sample point
    alphas = 1.0 - torch.exp(-sigma * deltas).clamp_min(0.0) # shape (num_rays, num_ray_pts)

    # Step 3. Compute transmittance as a cumulative product over (1 - alpha)
    shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1.0 - alphas + eps], dim=-1)
    transmittance = torch.cumprod(shifted, dim=-1)[..., :-1]    # shape (num_rays, num_ray_pts)

    # Step 4. Compute the weights as the product of the transmittance and opacities,
    # the accumulated opacity as the sum of the weights, and the depths as the sum of
    # the weighted sample points normalized by the accumulated opacity
    weights = transmittance * alphas    # shape (num_rays, num_ray_pts)
    accumulated_opacity = weights.sum(dim=-1)    # shape (num_rays)
    depths = (weights * z_depths).sum(dim=-1) / (accumulated_opacity + eps)  # shape (num_rays)

    # Step 5. Compute the composite color as the weighted sum of the MLP-predicted RGB color
    composite_rgb = (weights[..., None] * rgb).sum(dim=-2)   # shape (num_rays, 3)

    # If inserting a white background, add 1 - accumulated_opacity to the composited color
    if white_bkgd:
        composite_rgb = composite_rgb + (1.0 - accumulated_opacity)[..., None]

    return composite_rgb, weights, accumulated_opacity, depths
