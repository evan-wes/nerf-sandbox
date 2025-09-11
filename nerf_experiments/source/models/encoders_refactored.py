"""
Positional encoders for NeRF with a convenience factory for vanilla defaults.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self,
                 input_dims: int = 3,
                 num_freqs: int = 10,
                 include_input: bool = True,
                 log_spaced: bool = True,
                 use_two_pi: bool = False):
        super().__init__()
        self.input_dims = int(input_dims)
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        self.log_spaced = bool(log_spaced)
        self.use_two_pi = bool(use_two_pi)

        if self.log_spaced:
            self.freq_bands = 2 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** (self.num_freqs - 1), self.num_freqs)

        L = self.num_freqs
        in_dim = self.input_dims
        self.out_dim = (in_dim if include_input else 0) + in_dim * L * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_dims)
        returns: (..., out_dim)
        """
        orig_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = []
        if self.include_input:
            out.append(x)
        c = (2.0 * math.pi) if self.use_two_pi else 1.0
        for f in self.freq_bands.to(x.device):
            out.append(torch.sin(c * f * x))
            out.append(torch.cos(c * f * x))
        y = torch.cat(out, dim=-1)
        y = y.view(*orig_shape, -1)
        return y


# ------------------- Vanilla NeRF helpers -------------------

def get_vanilla_nerf_encoders():
    """
    Return (pos_encoder, dir_encoder) using the official NeRF defaults:
    - positions: L=10 (out_dim = 3 + 3*10*2 = 63)
    - viewdirs:  L=4  (out_dim = 3 + 3*4*2  = 27)
    - include_input=True, log-spaced bands, no 2π factor
    """
    pos_enc = PositionalEncoder(input_dims=3, num_freqs=10,
                                include_input=True, log_spaced=True, use_two_pi=False)
    dir_enc = PositionalEncoder(input_dims=3, num_freqs=4,
                                include_input=True, log_spaced=True, use_two_pi=False)
    return pos_enc, dir_enc
