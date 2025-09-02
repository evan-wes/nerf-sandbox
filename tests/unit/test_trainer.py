import json
from dataclasses import FrozenInstanceError, is_dataclass, fields
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


from nerf_experiments.source.config.config_utils import parse_nerf_config, NerfConfig, DataConfig, CameraConfig, TrainSection, RenderConfig, EncodingConfig, PEConfig
from nerf_experiments.source.config.runtime_config import to_runtime_train_config, RuntimeTrainConfig
from nerf_experiments.source.train.trainer import Trainer


# ---------- helpers ----------

def _write_png(path: Path, arr: np.ndarray) -> None:
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)

def _make_transforms(path: Path, frames, camera_angle_x=np.deg2rad(60.0)) -> None:
    data = {
        "camera_angle_x": float(camera_angle_x),
        "frames": [
            {"file_path": f"./{name}", "transform_matrix": T.tolist()}
            for name, T in frames
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def tiny_blender(tmp_path: Path):
    """
    Build a tiny Blender-style dataset:
      - 1 frame, 8x8 RGB mid-gray image
      - identity pose
    Returns: (root_dir, H, W)
    """
    H, W = 8, 8
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    _write_png(tmp_path / "im0.png", img)

    T0 = np.eye(4, dtype=np.float32)
    _make_transforms(tmp_path / "transforms_train.json", [("im0", T0)])
    return tmp_path, H, W


# ---------- tests ----------

def test_runtime_config_is_frozen(tiny_blender):
    root, H, W = tiny_blender

    # Build a NerfConfig (sectioned) similar to your YAML
    nc = NerfConfig(
        data=DataConfig(kind="blender", root=str(root), white_bg=True, downscale=1, split="train"),
        camera=CameraConfig(ndc=False, near=0.1, far=2.0, model="pinhole"),
        train=TrainSection(device="cpu", rays_per_batch=64, max_steps=1000, amp=False, val_every=1000, log_every=50, ckpt_every=1000, lr=1e-3, out_dir=str(root / "outs")),
        render=RenderConfig(Nc=8, Nf=4, det_fine=False),
        encoding=EncodingConfig(pos=PEConfig(num_freqs=4, include_input=True), dir=PEConfig(num_freqs=2, include_input=True)),
    )

    rt = to_runtime_train_config(nc, save_dir=nc.train.out_dir, original_yaml=None)
    assert isinstance(rt, RuntimeTrainConfig)
    assert is_dataclass(rt)
    assert fields(rt)[0].repr

    # Frozen: attempts to mutate should fail
    with pytest.raises(FrozenInstanceError):
        rt.lr = 9.99



def test_trainer_init_and_scheduler_cosine(tiny_blender):
    root, H, W = tiny_blender

    nc = NerfConfig(
        data=DataConfig(kind="blender", root=str(root), white_bg=True, downscale=1, split="train"),
        camera=CameraConfig(ndc=False, near=0.1, far=2.0, model="pinhole"),
        train=TrainSection(device="cpu", rays_per_batch=32, max_steps=200, amp=False, val_every=1000, log_every=50, ckpt_every=1000, lr=5e-4, out_dir=str(root / "outs")),
        render=RenderConfig(Nc=4, Nf=2, det_fine=False),
        encoding=EncodingConfig(pos=PEConfig(num_freqs=3, include_input=True), dir=PEConfig(num_freqs=2, include_input=True)),
    )
    rt = to_runtime_train_config(nc, save_dir=None)

    tr = Trainer(rt)
    # models on correct device
    dev = next(tr.nerf_c.parameters()).device
    assert str(dev) == "cpu"

    # scheduler present and configured
    if tr.sched is not None:
        assert hasattr(tr.sched, "T_max")
        assert tr.sched.T_max == rt.max_steps
        assert hasattr(tr.sched, "eta_min")
        assert pytest.approx(tr.sched.eta_min, rel=1e-6) == rt.scheduler_params["eta_min"]


def test_train_step_smoke(tiny_blender):
    root, H, W = tiny_blender

    nc = NerfConfig(
        data=DataConfig(kind="blender", root=str(root), white_bg=True, downscale=1, split="train"),
        camera=CameraConfig(ndc=False, near=0.1, far=2.0, model="pinhole"),
        train=TrainSection(device="cpu", rays_per_batch=16, max_steps=20, amp=False, val_every=1000, log_every=10, ckpt_every=1000, lr=1e-3, out_dir=str(root / "outs")),
        render=RenderConfig(Nc=4, Nf=2, det_fine=False),
        encoding=EncodingConfig(pos=PEConfig(num_freqs=2, include_input=True), dir=PEConfig(num_freqs=1, include_input=True)),
    )
    rt = to_runtime_train_config(nc)
    tr = Trainer(rt)

    it = iter(tr.sampler)
    batch = next(it)
    out = tr.train_step(batch)

    assert "loss" in out and torch.is_tensor(out["loss"])
    assert "psnr" in out and torch.is_tensor(out["psnr"])
    assert out["comp_f"].shape == (rt.rays_per_batch, 3)
    assert out["comp_c"].shape == (rt.rays_per_batch, 3)


def test_validate_full_image_shape(tiny_blender):
    root, H, W = tiny_blender

    nc = NerfConfig(
        data=DataConfig(kind="blender", root=str(root), white_bg=True, downscale=1, split="train"),
        camera=CameraConfig(ndc=False, near=0.1, far=2.0, model="pinhole"),
        train=TrainSection(device="cpu", rays_per_batch=32, max_steps=10, amp=False, val_every=10, log_every=10, ckpt_every=10, lr=1e-3, out_dir=str(root / "outs")),
        render=RenderConfig(Nc=4, Nf=2, det_fine=True),  # det for deterministic val
        encoding=EncodingConfig(pos=PEConfig(num_freqs=2, include_input=True), dir=PEConfig(num_freqs=1, include_input=True)),
    )
    rt = to_runtime_train_config(nc)
    tr = Trainer(rt)

    val = tr.validate_full_image(frame_idx=0)
    assert "rgb" in val
    assert val["rgb"].shape == (H, W, 3)
    assert (val["rgb"] >= 0).all() and (val["rgb"] <= 1).all()


def test_hierarchical_sampling_calls_pdf(monkeypatch, tiny_blender):
    """
    Ensure hierarchical sampling path invokes sample_pdf with expected shapes.
    We monkeypatch trainer.sample_pdf to record arguments.
    """
    root, H, W = tiny_blender

    nc = NerfConfig(
        data=DataConfig(kind="blender", root=str(root), white_bg=True, downscale=1, split="train"),
        camera=CameraConfig(ndc=False, near=0.1, far=2.0, model="pinhole"),
        train=TrainSection(device="cpu", rays_per_batch=8, max_steps=10, amp=False, val_every=10, log_every=10, ckpt_every=10, lr=1e-3, out_dir=str(root / "outs")),
        render=RenderConfig(Nc=6, Nf=3, det_fine=False),
        encoding=EncodingConfig(pos=PEConfig(num_freqs=2, include_input=True), dir=PEConfig(num_freqs=1, include_input=True)),
    )
    rt = to_runtime_train_config(nc)
    tr = Trainer(rt)

    called = {"n": 0, "bins_shape": None, "weights_shape": None, "n_samples": None, "det": None}

    def fake_sample_pdf(bins, weights, n_samples, deterministic=False):
        called["n"] += 1
        called["bins_shape"] = tuple(bins.shape)       # expect [B, Nc-1]
        called["weights_shape"] = tuple(weights.shape) # expect [B, Nc-2] (interior)
        called["n_samples"] = int(n_samples)
        called["det"] = bool(deterministic)
        # return uniform mids for shape correctness
        B, M = bins.shape
        return bins[:, torch.linspace(0, M-1, steps=n_samples).long()]

    # Patch the symbol used inside trainer
    monkeypatch.setattr("nerf_experiments.source.train.trainer.sample_pdf", fake_sample_pdf, raising=True)

    batch = next(iter(tr.sampler))
    _ = tr.train_step(batch)

    assert called["n"] >= 1
    # Nc=6 -> bins shape [B, 5]; weights interior [B, 4]
    assert called["bins_shape"][1] == rt.nc - 1
    assert called["weights_shape"][1] == rt.nc - 2
    assert called["n_samples"] == rt.nf
    # det flag equals Trainer.det_fine in train_step (False here)
    assert called["det"] is rt.det_fine
