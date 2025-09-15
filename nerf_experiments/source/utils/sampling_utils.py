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

    # Get the device (align to bins to avoid cross-device issues)
    device = bins.device

    # Avoid zeros in denominators and tiny negatives in weights
    epsilon = 1e-5
    weights = weights.to(device=device, dtype=bins.dtype).clamp_min(0) + epsilon
    weights_sum = weights.sum(dim=-1, keepdim=True)

    # If sum is still ~0 (all near-zero), fall back to uniform
    use_uniform_samples = (weights_sum <= 1e-6)
    weights = torch.where(use_uniform_samples, torch.ones_like(weights), weights)

    # Create the probability density function by normalizing the weights
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True).clamp_min(epsilon)

    # Sum the PDF to create the cumulative distribution function,
    # prepending 0 as the first value
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # shape (num_rays, num_bins+1)

    # Create the positions along the ray to sample from
    if deterministic:
        # Use interval centers to reduce edge bias
        u = (torch.arange(n_samples, device=device, dtype=bins.dtype) + 0.5) / n_samples
        u = u.unsqueeze(0).expand(*cdf.shape[:-1], n_samples)  # shape (num_rays, n_samples)
    else:
        # Create a random set of points between 0 to 1 with shape (num_rays, n_samples)
        u = torch.rand(*cdf.shape[:-1], n_samples, device=device, dtype=bins.dtype)
    u = u.clamp(0.0, 1.0 - 1e-8)

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
    t = (u - cdf_values[..., 0]) / denominator
    samples = bins_values[..., 0] + t * (bins_values[..., 1] - bins_values[..., 0])  # (num_rays, n_samples)

    # If uniform fallback case, just stratified uniform within [bins.min, bins.max]
    if use_uniform_samples.any():
        low = bins[:, :1]
        high = bins[:, -1:]
        if deterministic:
            u2 = (torch.arange(n_samples, device=device, dtype=bins.dtype) + 0.5) / n_samples
            u2 = u2.unsqueeze(0).expand_as(samples)
        else:
            u2 = torch.rand_like(samples)
        uniform = low + (high - low) * u2
        samples = torch.where(use_uniform_samples, uniform, samples)

    return samples
