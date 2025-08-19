"""Contains utilities for rendering views"""

import torch

def render_ray(ray_samples, densities, colors):
    """
    Performs volumetric rendering along a ray.
    Args:
        ray_samples: Points sampled along the ray (N, 3).
        densities: Predicted density for each point (N, 1).
        colors: Predicted RGB color for each point (N, 3).
    Returns:
        Rendered pixel color (3,).
    """
    # Compute transmittance (1 - accumulated opacity)
    alphas = 1.0 - torch.exp(-densities)
    weights = alphas * torch.cumprod(torch.cat([torch.ones(1), 1.0 - alphas[:-1] + 1e-10]), dim=0)
    rendered_color = torch.sum(weights[:, None] * colors, dim=0)
    return rendered_color

