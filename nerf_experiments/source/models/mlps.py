"""
Contains implementations of multi-layer perceptrons.
"""

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
        skip_pos: int = 5
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

        #----- Volume density branch -----#
        # Receives the MLP output and produces a scalar
        self.sigma_out = nn.Linear(self.hidden_dim, 1)

        #----- Color branch -----#
        # Receives the MLP output concatenated with the encoded direction,
        # produces an intermediate vector half the size of the hidden layers,
        # and then the color
        self.color_fc = nn.Linear(self.hidden_dim + self.enc_dir_dim, self.hidden_dim//2)
        self.color_out = nn.Linear(self.hidden_dim//2, 3)

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

        # Volume density prediction (additional ReLU ensures non-negativity)
        sigma = F.relu(self.sigma_out(hidden_tensor))

        # Predict the feature vector without any activation
        feature = self.feature(hidden_tensor)

        # Color prediction
        color_input = torch.cat([feature, enc_dir], dim=-1)
        color_hidden = F.relu(self.color_fc(color_input))
        color_rgb = torch.sigmoid(self.color_out(color_hidden))

        # Construct the output tensor and return
        output_tensor = torch.cat([color_rgb, sigma], dim=-1)

        return output_tensor


if __name__ == "__main__":

    nerf = NeRF()

    print(nerf)