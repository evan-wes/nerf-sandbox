"""Contains the PositionalEncoder class for encoding data"""

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
        use_two_pi: bool = False,
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
        self.input_dims = int(input_dims)
        self.num_freqs = int(num_freqs)
        self.include_input = bool(include_input)
        self.use_two_pi = bool(use_two_pi)

        if min_freq_log2 is None:
            min_freq_log2 = 0
        if max_freq_log2 is None:
            max_freq_log2 = self.num_freqs - 1

        if log_spaced:
            # 2^{min..max}, num_freqs samples
            freq_bands = 2.0 ** torch.linspace(float(min_freq_log2), float(max_freq_log2), steps=self.num_freqs)
        else:
            # Linear in frequency but same endpoints
            freq_bands = torch.linspace(2.0 ** float(min_freq_log2),
                                        2.0 ** float(max_freq_log2),
                                        steps=self.num_freqs)

        # Register as a buffer so .to(device) on the module moves it
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        self.out_dim = (self.input_dims if self.include_input else 0) + self.input_dims * self.num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the encoding to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., D), where D = input_dims provided to the class init

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (..., self.out_dim)
        """
        # Ensure buffer uses the same device & dtype as x
        fb = self.freq_bands.to(device=x.device, dtype=x.dtype)

        # Make scale a tensor on x's device/dtype (avoid CPU scalar with CUDA tensor)
        scale_val = 2 * torch.pi if self.use_two_pi else 1.0
        scale = torch.as_tensor(scale_val, device=x.device, dtype=x.dtype)

        # (..., 1, D) * (F, 1) -> (..., F, D)
        xb = x.unsqueeze(-2) * (fb * scale).unsqueeze(-1)

        sin_feats = torch.sin(xb)   # (..., F, D)
        cos_feats = torch.cos(xb)   # (..., F, D)

        # concat along freq axis, then flatten freq & dim: (..., 2F*D)
        enc = torch.cat([sin_feats, cos_feats], dim=-2).reshape(*x.shape[:-1], -1)

        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)

        return enc

def get_vanilla_nerf_encoders():
    """
    Return (pos_encoder, dir_encoder) using the official NeRF defaults:
      - positions: L=10, include_input=True, log-spaced, no 2π
      - viewdirs:  L=4,  include_input=True, log-spaced, no 2π
    out_dims: pos=3+3*10*2=63, dir=3+3*4*2=27
    """
    pos_enc = PositionalEncoder(
        input_dims=3, num_freqs=10,
        include_input=True, log_spaced=True, use_two_pi=False
    )
    dir_enc = PositionalEncoder(
        input_dims=3, num_freqs=4,
        include_input=True, log_spaced=True, use_two_pi=False
    )
    return pos_enc, dir_enc
