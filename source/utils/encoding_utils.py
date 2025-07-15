"""Contains utilities for encoding positional data"""

import torch
import math

def positional_encoding(x, num_frequencies=10):
    """
    Maps input coordinates into a higher-dimensional space using sine and cosine functions.
    Args:
        x: Tensor of shape (N, 3) or (N, 2) for spatial or directional inputs.
        num_frequencies: Number of frequencies for encoding.
    Returns:
        Encoded tensor of shape (N, num_frequencies * 2 * dim(x)).
    """
    frequencies = [2 ** i for i in range(num_frequencies)]
    encoding = []
    for freq in frequencies:
        encoding.append(torch.sin(freq * x))
        encoding.append(torch.cos(freq * x))
    return torch.cat(encoding, dim=-1)
