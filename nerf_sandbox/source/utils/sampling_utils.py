"""Contains utilities for sampling along rays using probability density functions"""

import torch

@torch.no_grad()
def sample_pdf(
    bins: torch.Tensor,          # (B, M) midpoints OR (B, M+1) edges
    weights: torch.Tensor,       # (B, M)
    n_samples: int,
    *,
    deterministic: bool = False
) -> torch.Tensor:
    """
    Hierarchical sampling from a piecewise-constant PDF.
    Returns (B, n_samples).
    """
    if bins.ndim != 2 or weights.ndim != 2:
        raise ValueError(f"Expected (B,·) tensors: bins={bins.shape}, weights={weights.shape}")
    B, M = weights.shape

    # --- Accept edges or midpoints; build edges (B, M+1)
    if bins.shape[-1] == M + 1:
        edges = bins.contiguous()
    elif bins.shape[-1] == M:
        mids = bins.contiguous()
        if M == 1:
            d = torch.full_like(mids, 1e-3)
            edges = torch.cat([mids - 0.5 * d, mids + 0.5 * d], dim=-1)
        else:
            lo = mids[:, :1]  - 0.5 * (mids[:, 1:2]   - mids[:, :1])
            hi = mids[:, -1:] + 0.5 * (mids[:, -1:]   - mids[:, -2:-1])
            inter = 0.5 * (mids[:, 1:] + mids[:, :-1])
            edges = torch.cat([lo, inter, hi], dim=-1)
    else:
        raise ValueError(f"Incompatible shapes: bins={bins.shape}, weights={weights.shape}")

    # --- PDF/CDF (add 1e-5 before normalize)
    w = (weights + 1e-5).clamp_min(0)
    pdf = w / torch.sum(w, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)                             # (B, M)
    cdf = torch.cat([torch.zeros(B, 1, device=cdf.device, dtype=cdf.dtype), cdf], dim=-1)  # (B, M+1)

    # --- Target CDF samples u: official det = linspace(0..1) endpoints included
    if deterministic:
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.unsqueeze(0).expand(B, -1).contiguous()
    else:
        u = torch.rand(B, n_samples, device=cdf.device, dtype=cdf.dtype).contiguous()

    # --- Invert CDF
    inds  = torch.searchsorted(cdf, u, right=True)        # (B, n_samples) in [1..M]
    below = (inds - 1).clamp(min=0, max=cdf.shape[-1]-1)
    above = inds.clamp(min=1, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], dim=-1)                 # (B, n_samples, 2)

    cdf_g   = torch.gather(cdf.unsqueeze(1).expand(B, n_samples, -1),   -1, inds_g)
    edges_g = torch.gather(edges.unsqueeze(1).expand(B, n_samples, -1), -1, inds_g)

    cdf_lo, cdf_hi = cdf_g[..., 0], cdf_g[..., 1]
    bins_lo, bins_hi = edges_g[..., 0], edges_g[..., 1]
    denom = cdf_hi - cdf_lo
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)  # official guard
    t = (u - cdf_lo) / denom
    return bins_lo + t * (bins_hi - bins_lo)
