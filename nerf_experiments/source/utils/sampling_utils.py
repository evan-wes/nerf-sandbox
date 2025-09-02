"""Contains utilities for sampling along rays using probability density functions"""

import torch

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    deterministic: bool = False
) -> torch.Tensor:
    """
    Uses a piecewise-constant histogram representing surface location probabilities
    along a ray given by the input bins and weights tensor to construct a probability
    density function, and samples from it using inverse-transform sampling. Supports
    batched inputs.

    Parameters
    ----------
    bins : torch.Tensor
        Input bin edges. Shape is (num_rays, num_bins + 1)
    weights : torch.Tensor
        Constant value inside each bin. Shape is (num_rays, num_bins)
    n_samples : int
        Number of samples to return
    deterministic : bool, optional
        A flag that controls whether the sampling is deterministic or random

    Returns
    -------
    torch.Tensor :
        The sampled locations along the ray based on the input histogram.
    """

    # Get the device
    device = weights.device

    # Avoid zeros in demonminators by adding a small epsilon to the input weights
    epsilon = 1e-5
    weights = weights + epsilon

    # Create the probability density function by normalizing the weights
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Sum the PDF to create the cumulative distribution function,
    # prepending 0 as the first value
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # shape (num_rays, len(bins))

    # Create the positions along the ray to sample from
    if deterministic:
        # Create a uniform set of points from 0 to 1 and expand to shape (num_rays, n_samples)
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
        u = u.expand(*cdf.shape[:-1], n_samples) # shape (num_rays, n_samples)
    else:
        # Create a random set of points between 0 to 1 with shape (num_rays, n_samples)
        u = torch.rand(*cdf.shape[:-1], n_samples, device=device) # shape (num_rays, n_samples)


    # Get the positions in the CDF that bracket each of the sampled points between 0 and 1
    cdf = cdf.contiguous()
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)
    below = (indices - 1).clamp(min=0)
    above = indices.clamp(max=cdf.shape[-1] - 1)

    # Gather the indices into sets of [lower_index, upper_index] spanning the bin containing each sampled point
    index_bins = torch.stack([below, above], dim=-1)  # shape (num_rays, n_samples, 2)
    # Gather the CDF values and bins
    gather_shape = [*index_bins.shape[:-1], cdf.shape[-1]]
    cdf_values = torch.gather(cdf.unsqueeze(1).expand(gather_shape), 2, index_bins)
    bins_values = torch.gather(bins.unsqueeze(1).expand(gather_shape), 2, index_bins)

    # Invert the CDF for inverse-transform sampling by linear interpolation
    denominator = (cdf_values[..., 1] - cdf_values[..., 0]).clamp(min=epsilon)
    t = (u - cdf_values[..., 0])/denominator
    samples = bins_values[..., 0] + t * (bins_values[..., 1] - bins_values[..., 0]) # shape (num_rays, n_samples)

    return samples
