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
        near: float = 2.0,
        far: float = 6.0,
        initial_acc_opacity: float | None = None,
        sigma_activation: str = "softplus"
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
        near : float, optional
            Position of the near plane. Default is 2.0
        far : float, optional
            Position of the far plane. Default is 6.0
        initial_acc_opacity : float, optional
            Initial accumulated opacity, used to initialize sigma. Defaults to None
        sigma_activation : str, optional
            Type of activation to be applied to the raw sigma output. Used to solve
            for a proper initialization, if initial_acc_opacity is provided. Defaults
            to "softplus"
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

        # -------- Initialize sigma bias for accumulated opacity --------#
        if initial_acc_opacity is not None:
            sigma_bias = self._sigma_bias_for_initial_acc_opacity(
                initial_acc_opacity,
                near=near,
                far=far,
                activation=sigma_activation
            )
            with torch.no_grad():
                self.sigma_out.bias.fill_(sigma_bias)
                # self.sigma_out.weight.mul_(0.1)
                self.color_out.bias.zero_()
                self.color_out.weight.mul_(0.1)

        #----- Initialize the MLP layers -----#
        self._init_mlp()

    def _sigma_bias_for_initial_acc_opacity(
        self,
        initial_acc_opacity: float,
        near: float,
        far: float,
        activation: str = "softplus"
    ) -> float:
        """
        Choose a bias 'b' so that activation(b) integrates to the desired initial
        accumulated opacity over a uniform ray from [near, far].

        We solve for a target per-sample density sigma* such that:
            1 - exp(-sigma* * (far - near)) = initial_acc_opacity
        and then pick 'b' so that activation(b) = sigma*.

        Supported:
        - activation == "softplus":  b = log(exp(sigma*) - 1)
        - activation == "relu":      b = sigma*     (since ReLU(b) = b for b>0)
        - fallback (unknown/linear): b = sigma*
        """
        # Clamp the requested opacity into a safe open interval
        p = float(max(1e-6, min(0.99, initial_acc_opacity)))

        # Uniform path length
        L = float(max(1e-8, far - near))

        # Target density so that the integrated transmittance matches p
        sigma_star = -math.log(1.0 - p) / L  # > 0

        act = (activation or "softplus").lower()

        if act == "softplus":
            # inverse softplus: x = log(exp(y) - 1)  (use expm1 for stability)
            return float(math.log(math.expm1(sigma_star)))
        elif act == "relu":
            # ReLU(bias) = sigma_star  → bias = sigma_star (positive)
            return float(sigma_star)
        else:
            # Sensible fallback: assume linear/identity activation on sigma_raw
            # so that sigma_raw ≈ sigma_star.
            return float(sigma_star)

    def _init_mlp(self):
        """
        Gives recommended initializations for the main MLP trunk, and feature
        and color layers
        """
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