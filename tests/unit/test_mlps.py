import math
import pytest
import torch
import torch.nn as nn

from nerf_experiments.source.models.mlps import NeRF


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_constructor_in_features_sequence():
    """
    Enforce the skip-connection contract:

    If we concatenate enc_pos AFTER layer `skip_pos`, then the NEXT layer (skip_pos+1)
    must have in_features = hidden_dim + enc_pos_dim. All other layers (except the
    very first) should have in_features = hidden_dim. The very first layer must take
    enc_pos_dim.
    """
    enc_pos_dim = 15
    enc_dir_dim = 6
    n_layers = 8
    hidden = 64
    skip_pos = 3  # zero-based index

    nerf = NeRF(
        enc_pos_dim=enc_pos_dim,
        enc_dir_dim=enc_dir_dim,
        n_layers=n_layers,
        hidden_dim=hidden,
        skip_pos=skip_pos,
    )

    assert len(nerf.mlp) == n_layers

    # Layer 0 must consume encoded positions
    assert nerf.mlp[0].in_features == enc_pos_dim
    assert nerf.mlp[0].out_features == hidden

    # Layers 1..n_layers-1 default to hidden -> hidden, EXCEPT layer skip_pos+1
    for i in range(1, n_layers):
        if i == skip_pos + 1:
            # The layer AFTER the skip must consume hidden + enc_pos_dim
            assert nerf.mlp[i].in_features == hidden + enc_pos_dim, (
                f"Layer {i} should accept hidden+enc_pos_dim because "
                f"we concatenate AFTER layer {skip_pos}"
            )
        else:
            assert nerf.mlp[i].in_features == hidden, (
                f"Layer {i} should accept hidden only (no concat here)"
            )
        assert nerf.mlp[i].out_features == hidden


def test_color_and_sigma_branch_dims():
    enc_pos_dim = 12
    enc_dir_dim = 8
    hidden = 128

    nerf = NeRF(enc_pos_dim, enc_dir_dim, n_layers=6, hidden_dim=hidden, skip_pos=2)

    # Feature layer operates on hidden
    assert nerf.feature.in_features == hidden
    assert nerf.feature.out_features == hidden

    # Color branch takes feature (hidden) + encoded dir
    assert nerf.color_fc.in_features == hidden + enc_dir_dim
    assert nerf.color_fc.out_features == hidden // 2
    assert nerf.color_out.in_features == hidden // 2
    assert nerf.color_out.out_features == 3

    # Sigma head outputs 1 (density)
    assert nerf.sigma_out.in_features == hidden
    assert nerf.sigma_out.out_features == 1


def test_forward_shapes_and_ranges():
    """
    Forward pass should:
      - accept [B*N, enc_pos_dim], [B*N, enc_dir_dim]
      - return [B*N, 4] (rgb, sigma)
      - rgb in [0,1] (sigmoid)
      - sigma >= 0 (ReLU)
    """
    torch.manual_seed(0)
    B, N = 4, 5
    enc_pos_dim = 27
    enc_dir_dim = 9
    hidden = 64

    nerf = NeRF(enc_pos_dim, enc_dir_dim, n_layers=5, hidden_dim=hidden, skip_pos=2)
    x = torch.randn(B * N, enc_pos_dim)
    d = torch.randn(B * N, enc_dir_dim)

    out = nerf(x, d)
    assert out.shape == (B * N, 4)

    rgb = out[..., :3]
    sigma = out[..., 3]

    assert torch.all(rgb >= 0) and torch.all(rgb <= 1), "rgb must be in [0,1] due to sigmoid"
    assert torch.all(sigma >= 0), "sigma must be non-negative (ReLU)"


def test_grad_flow():
    """
    Backprop should produce gradients for all parameters.
    """
    torch.manual_seed(0)
    enc_pos_dim = 21
    enc_dir_dim = 6
    nerf = NeRF(enc_pos_dim, enc_dir_dim, n_layers=4, hidden_dim=32, skip_pos=1)

    x = torch.randn(32, enc_pos_dim, requires_grad=True)
    d = torch.randn(32, enc_dir_dim, requires_grad=True)
    out = nerf(x, d).sum()
    out.backward()

    for name, p in nerf.named_parameters():
        assert p.grad is not None, f"Param {name} has no gradient"


def test_mismatch_raises_runtime_error():
    """
    If enc_pos/enc_dir tensors don't match the declared dims, forward should fail.
    """
    enc_pos_dim = 10
    enc_dir_dim = 6
    nerf = NeRF(enc_pos_dim, enc_dir_dim, n_layers=3, hidden_dim=16, skip_pos=1)

    # Wrong enc_pos dim
    x_bad = torch.randn(7, enc_pos_dim + 1)
    d_ok = torch.randn(7, enc_dir_dim)
    with pytest.raises(RuntimeError):
        _ = nerf(x_bad, d_ok)

    # Wrong enc_dir dim
    x_ok = torch.randn(7, enc_pos_dim)
    d_bad = torch.randn(7, enc_dir_dim + 2)
    with pytest.raises(RuntimeError):
        _ = nerf(x_ok, d_bad)


def test_parameter_count_stability():
    """
    Sanity check: increasing hidden_dim or n_layers should increase params.
    (Not meant to be a golden exact number; guards wild regressions.)
    """
    a = NeRF(enc_pos_dim=12, enc_dir_dim=6, n_layers=4, hidden_dim=64, skip_pos=1)
    b = NeRF(enc_pos_dim=12, enc_dir_dim=6, n_layers=6, hidden_dim=64, skip_pos=1)
    c = NeRF(enc_pos_dim=12, enc_dir_dim=6, n_layers=6, hidden_dim=128, skip_pos=1)

    pa = _count_params(a)
    pb = _count_params(b)
    pc = _count_params(c)

    assert pb > pa, "More layers should increase params"
    assert pc > pb, "Wider layers should increase params"
