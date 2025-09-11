"""
NeRF MLP, configurable and with a vanilla factory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    MLP with:
      - trunk on encoded positions with one skip concat of γ(x)
      - sigma and feature heads from trunk
      - color head that takes [feature, γ(d)]
    """
    VANILLA_SKIP_AT = 3  # concat γ(x) AFTER layer idx=3 → layer 4 sees skip

    @classmethod
    def from_vanilla_defaults(cls, enc_pos_dim: int, enc_dir_dim: int) -> "NeRF":
        return cls(enc_pos_dim=enc_pos_dim,
                   enc_dir_dim=enc_dir_dim,
                   n_layers=8,
                   hidden_dim=256,
                   skip_pos=NeRF.VANILLA_SKIP_AT,
                   acc0=0.0)

    def __init__(self,
                 enc_pos_dim: int,
                 enc_dir_dim: int,
                 n_layers: int = 8,
                 hidden_dim: int = 256,
                 skip_pos: int = 5,
                 acc0: float = 0.0,
                 near: float = 2.0,
                 far: float = 6.0):
        super().__init__()
        self.enc_pos_dim = int(enc_pos_dim)
        self.enc_dir_dim = int(enc_dir_dim)
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.skip_pos = int(skip_pos)
        self.acc0 = float(acc0)
        self.near = float(near); self.far = float(far)

        # Trunk over encoded positions
        layers = []
        in_dim = self.enc_pos_dim
        for i in range(self.n_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, self.hidden_dim))
            elif i == (self.skip_pos + 1):
                layers.append(nn.Linear(self.hidden_dim + self.enc_pos_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.trunk = nn.ModuleList(layers)

        # Sigma and feature heads
        self.sigma_out = nn.Linear(self.hidden_dim, 1)
        self.feat_out = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Color head (condition on direction encoding)
        self.color_fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.enc_dir_dim, self.hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim // 2, 3),
        )

        # Optional bias init to pre-bake opacity (disabled by default for vanilla)
        b = self._sigma_bias_for_initial_acc(self.acc0, near=self.near, far=self.far)
        with torch.no_grad():
            if self.acc0 and self.acc0 > 0:
                self.sigma_out.bias.fill_(b)

    @staticmethod
    def _sigma_bias_for_initial_acc(acc0: float, near: float, far: float) -> float:
        if acc0 <= 0.0:
            return 0.0
        # Approximate bias to get a target integrated alpha on a uniform ray
        L = (far - near)
        return float(torch.log(torch.tensor(acc0 / max(1e-6, 1 - acc0))) / max(1e-6, L))

    def forward(self, enc_pos: torch.Tensor, enc_dir: torch.Tensor) -> torch.Tensor:
        """
        enc_pos: (B*N, enc_pos_dim)
        enc_dir: (B*N, enc_dir_dim)
        returns: (B*N, 4) -> [rgb_raw(3), sigma_raw(1)]
        """
        x = enc_pos
        for i, fc in enumerate(self.trunk):
            x = fc(x)
            x = torch.relu(x)
            if i == self.skip_pos:
                x = torch.cat([x, enc_pos], dim=-1)

        sigma_raw = self.sigma_out(x)[..., 0]  # (B*N,)
        feat = self.feat_out(x)
        h = torch.cat([feat, enc_dir], dim=-1)
        rgb_raw = self.color_fc(h)
        return torch.cat([rgb_raw, sigma_raw[..., None]], dim=-1)
