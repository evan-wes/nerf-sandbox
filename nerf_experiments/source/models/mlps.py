"""
Contains implementations of multi-layer perceptrons.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    """
    Implements the multi-layer perceptron archicture as described in
    Figure 7. of the original NeRF paper (Mildenhall et al. 2020).
    """

    def __init__(
        self,
        enc_pos_dim: int,
        enc_dir_dim: int,
        n_layers: int = 8,
        hidden_dim: int = 256,
        skip_pos: int = 5,
        acc0: float = 0.05,
        near: float = 2.0,
        far: float = 6.0
    ) -> None:
        """
        Constructor for the class. Builds the sequential layer architecture, including
        a skip connection where the encoded positional input is concatenated with the
        activation of the fifth layer.

        Parameters
        ----------
        enc_pos_dim : int
            Dimension of the encoded position input
        enc_dir_dim : int
            Dimension of the encoded direction input
        n_layers : int, optional
            The number of fully-connected hidden layers. Default is 8.
        hidden_dim : int, optional
            The output dimension of the hidden layers. Default is 256.
        skip_pos : int, optional
            The index of the layer whose activation we concatenate the input encoded
            position vector. Default is 5, i.e. the encoded position is concatenated
            with the fifth layer's activation output vector
        acc0 : float, optional
            Initial accumulated opacity. Used to initialize sigma. Default is 5 percent
        near : float, optional
            Position of the near plane. Default is 2.0
        far : float, optional
            Position of the far plane. Default is 6.0
        """
        super(NeRF, self).__init__()

        self.enc_pos_dim = enc_pos_dim
        self.enc_dir_dim = enc_dir_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.skip_pos = skip_pos

        #----- Multi-layer perceptron trunk -----#
        layers = []
        in_dim = self.enc_pos_dim
        for idx in range(self.n_layers):
            # Create the current layer with the current input dimension
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            if idx == self.skip_pos:
                # Adjust input dimension to account for concatenation for the next layer
                in_dim = self.hidden_dim + self.enc_pos_dim
            else:
                in_dim = self.hidden_dim

        self.mlp = nn.ModuleList(layers)

        #----- Feature vector layer -----#
        self.feature = nn.Linear(self.hidden_dim, self.hidden_dim)

        #----- Volume density branch (RAW) -----#
        self.sigma_out = nn.Linear(self.hidden_dim, 1)   # raw sigma logits

        #----- Color branch -----#
        # Receives the MLP output concatenated with the encoded direction,
        # produces an intermediate vector half the size of the hidden layers,
        # and then the color
        self.color_fc = nn.Linear(self.hidden_dim + self.enc_dir_dim, self.hidden_dim//2)
        self.color_out = nn.Linear(self.hidden_dim//2, 3)  # raw rgb logits

        # -------- Recommended initializations --------#
        b = self._sigma_bias_for_initial_acc(acc0, near=near, far=far)
        with torch.no_grad():
            self.sigma_out.bias.fill_(b)
            # self.sigma_out.weight.mul_(0.1)
            self.color_out.bias.zero_()
            self.color_out.weight.mul_(0.1)

        # -------- Recommended initialization --------
        # with torch.no_grad():
        #     # Start very transparent so acc << 1 at init
        #     self.sigma_out.bias.fill_(-10.0)     # try -8 to -12; -10 is a good default for far-near ≈ 4
        #     self.sigma_out.weight.mul_(0.1)      # small σ head weights
        #     self.color_out.bias.zero_()
        #     self.color_out.weight.mul_(0.1)      # small RGB head weights

        self._init_mlp()

    def _sigma_bias_for_initial_acc(self, acc0: float, near: float, far: float) -> float:
        """Solve for b in softplus(b) so that 1-exp(-softplus(b)*(far-near)) ~= acc0."""
        acc0 = max(1e-6, min(0.99, acc0))
        L = float(far - near)
        # target avg sigma
        sig = -math.log(1.0 - acc0) / max(1e-8, L)
        # softplus^{-1}(sig) = log(exp(sig)-1)
        return float(math.log(math.expm1(sig)))

    def _init_mlp(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.kaiming_uniform_(self.feature.weight, nonlinearity="linear")
        nn.init.zeros_(self.feature.bias)
        nn.init.kaiming_uniform_(self.color_fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.color_fc.bias)

    def forward(self, enc_pos: torch.Tensor, enc_dir: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network

        Parameters
        ----------
        enc_pos : torch.Tensor
            Tensor of encoded positions, shape (num_rays, enc_pos_dim)
        enc_dir : torch.Tensor
            Tensor of encoded directions, shape (num_rays, enc_dir_dim)

        Returns
        -------
        torch.Tensor
            Predicted color and volume density tensor or dimension [num_rays, 4]
            where the outputs are [Red, Green, Blue, sigma]
        """

        hidden_tensor = enc_pos

        # MLP trunk
        for idx, layer in enumerate(self.mlp):
            # Pass the current hidden tensor through the layer and activation
            hidden_tensor = F.relu(layer(hidden_tensor))
            if idx == self.skip_pos:
                # Concatenate the current hidden tensor with the encoded position input
                hidden_tensor = torch.cat([hidden_tensor, enc_pos], dim=-1)

        # Volume density (RAW logits; no activation here)
        sigma_raw = self.sigma_out(hidden_tensor)

        # Predict the feature vector without any activation
        feature = self.feature(hidden_tensor)

        # Color prediction
        color_input = torch.cat([feature, enc_dir], dim=-1)
        color_hidden = F.relu(self.color_fc(color_input))
        color_raw = self.color_out(color_hidden)  # RAW logits; no sigmoid here

        # Construct the output tensor and return
        output_tensor = torch.cat([color_raw, sigma_raw], dim=-1)

        return output_tensor


if __name__ == "__main__":

    nerf = NeRF()

    print(nerf)