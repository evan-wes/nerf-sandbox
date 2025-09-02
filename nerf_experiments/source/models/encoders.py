"""Contains utilities for encoding data"""

import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    Positional encoding using sine and cosine bases:
    gamma(x) = [x (optional), sin(2^k x), cos(2^k x)] for k = 0, 1, ... num_freqs-1
    """
    def __init__(
        self,
        input_dims: int = 3,
        num_freqs: int = 10,
        include_input: bool = True,
        log_spaced: bool = True,
        min_freq_log2: int | None = None,
        max_freq_log2: int | None = None,
        use_two_pi: bool = False
    ) -> None:
        """
        Initialization for the PositionalEncoder class.

        Parameters
        ----------
        input_dims : int, optional
            Number of dimensions of the input vectors. Defaults to 3
        num_freqs : int, optional
            Number of frequency bands to use in the encoding. Defaults to 10.
        include_input : bool, optional
            Whether to include the input vector in the encoded input. Defaults to True.
        log_spaced : bool, optional
            Whether logarithmically space the frequency bands, as opposed to linearly.
            Defaults to True.
        min_freq_log2 : int | None, optional
            The smallest exponent of 2 in the frequency bands. If not provided, the smallest
            frequency is 0. Defaults to None.
        max_freq_log2 : int | None, optional
            The largest exponent of 2 in the frequency bands. If not provided, the largest
            frequency is given by num_freqs - 1. Defaults to None.
        use_two_pi : bool, optional
            Whether include a factor of 2pi in the frequencies or not. Defaults to False.

        Returns
        -------
        None
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.use_two_pi = use_two_pi

        if min_freq_log2 is None:
            min_freq_log2 = 0
        if max_freq_log2 is None:
            max_freq_log2 = num_freqs - 1

        if log_spaced:
            # Logarithmically spaced [2^0, 2^1, ..., 2^max_freq_log2]
            freq_bands = 2.0 ** torch.linspace(float(min_freq_log2), float(max_freq_log2), steps=num_freqs)
        else:
            # Linearly spaced between 2^0 and 2^max_freq_log2
            freq_bands = torch.linspace(2.0**float(min_freq_log2), 2.0**float(max_freq_log2), steps=num_freqs)

        # Register as buffer so it follows the module's device/dtype
        self.register_buffer('freq_bands', freq_bands, persistent=False)

        self.out_dim = (self.input_dims if self.include_input else 0) + self.input_dims * self.num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (num_rays, D), where D = input_dims provided to the class init

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (num_rays, self.out_dim)
        """

        # [num_rays, None, D] * [F] -> [num_rays, F, D]
        scale = (2 * torch.pi) if self.use_two_pi else 1.0
        xb = x.unsqueeze(-2) * (self.freq_bands * scale).unsqueeze(-1)  # broadcast

        # [num_rays, F, D] -> [num_rays, F, D] each
        sin_feats = torch.sin(xb)
        cos_feats = torch.cos(xb)

        # Flatten freq and dim: [num_rays, F*D]
        enc = torch.cat([sin_feats, cos_feats], dim=-2).reshape(*x.shape[:-1], -1)

        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)

        return enc
