import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nerf_sandbox.source.data.scene import Frame, Scene
from nerf_sandbox.source.data.loaders.blender_loader import BlenderSceneLoader
from nerf_sandbox.source.data.samplers import RandomPixelRaySampler


def _write_png(path: Path, arr: np.ndarray) -> None:
    """Write uint8 RGB(A) image."""
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr8).save(path)


def _make_transforms(path: Path, frames, camera_angle_x=np.deg2rad(60.0)) -> None:
    """Write a minimal transforms_*.json file."""
    data = {
        "camera_angle_x": float(camera_angle_x),
        "frames": [
            {"file_path": f"./{name}", "transform_matrix": T.tolist()}
            for name, T in frames
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def tiny_blender_scene(tmp_path: Path):
    """
    Construct a tiny Blender-like scene on disk:
      - two 8x8 images (one RGB, one RGBA)
      - identity C2W + a translated pose
    Returns the loaded Scene.
    """
    H, W = 8, 8

    # Make images: one RGB gray, one RGBA with 50% alpha over white
    rgb = np.full((H, W, 3), 128, dtype=np.uint8)                    # mid-gray
    rgba = np.concatenate(
        [np.full((H, W, 3), 64, dtype=np.uint8),  # darker RGB
         np.full((H, W, 1), 128, dtype=np.uint8)], axis=-1)          # alpha=0.5

    _write_png(tmp_path / "im0.png", rgb)
    _write_png(tmp_path / "im1.png", rgba)

    # Two poses
    T0 = np.eye(4, dtype=np.float32)
    T1 = np.eye(4, dtype=np.float32); T1[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    _make_transforms(tmp_path / "transforms_train.json", [("im0", T0), ("im1", T1)])

    loader = BlenderSceneLoader(tmp_path, downscale=1, white_bkgd=True)
    scene = loader.load("train")
    return scene


def test_sampler_shapes_and_norms(tiny_blender_scene: "Scene"):
    scene = tiny_blender_scene
    B = 256
    sampler = RandomPixelRaySampler(scene, rays_per_batch=B, device="cpu")

    it = iter(sampler)
    batch = next(it)

    assert set(batch.keys()) == {"rays_o", "rays_d", "rgb"}
    rays_o, rays_d, rgb = batch["rays_o"], batch["rays_d"], batch["rgb"]

    # Shapes
    assert rays_o.shape == (B, 3)
    assert rays_d.shape == (B, 3)
    assert rgb.shape == (B, 3)

    # Dtypes/devices
    assert rays_o.dtype == torch.float32
    assert rays_d.dtype == torch.float32
    assert rgb.dtype == torch.float32
    assert str(rays_o.device) == "cpu"

    # Directions are normalized (world space)
    norms = torch.linalg.norm(rays_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_white_bg_compositing_with_rgba(tiny_blender_scene: "Scene"):
    """
    When Scene.white_bkgd=True and source has RGBA, the sampler should composite:
      rgb = rgb_ * alpha + (1-alpha) * 1
    Verify the result lies in [0,1] and is brighter than raw rgb_ for alpha<1.
    """
    scene = tiny_blender_scene
    # Ensure at least one frame is RGBA and one is RGB
    has_rgba = any(getattr(fr.image, "shape", (0,))[-1] == 4 for fr in scene.frames)
    has_rgb = any(getattr(fr.image, "shape", (0,))[-1] == 3 for fr in scene.frames)
    assert has_rgba and has_rgb

    B = 128
    sampler = RandomPixelRaySampler(scene, rays_per_batch=B, device="cpu", white_bg_composite=True)
    batch = next(iter(sampler))
    rgb = batch["rgb"]  # [B,3] float32 in [0,1]

    assert torch.all(rgb >= 0.0) and torch.all(rgb <= 1.0)

    # Heuristic: since one image had alpha=0.5 over white and darker RGB,
    # some sampled pixels should be > raw dark value (64/255 ≈ 0.251)
    assert (rgb > 0.251).float().mean() > 0.1  # at least some fraction brighter


def test_multiple_batches_stream(tiny_blender_scene: "Scene"):
    """Sampler should produce an infinite stream; verify successive batches differ."""
    scene = tiny_blender_scene
    B = 64
    sampler = RandomPixelRaySampler(scene, rays_per_batch=B, device="cpu")
    it = iter(sampler)
    b1 = next(it)
    b2 = next(it)

    # Not guaranteed different, but very likely; check at least directions differ somewhere
    neq = (b1["rays_d"] - b2["rays_d"]).abs().sum(dim=-1) > 1e-6
    assert neq.any()


def test_sampler_on_cuda_if_available(tiny_blender_scene: "Scene"):
    """If CUDA is available, verify sampler places tensors on GPU when requested."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this runner")

    scene = tiny_blender_scene
    B = 32
    sampler = RandomPixelRaySampler(scene, rays_per_batch=B, device="cuda")
    batch = next(iter(sampler))
    for k, v in batch.items():
        assert v.is_cuda, f"{k} not on CUDA"


def test_ray_origins_match_camera_centers(tiny_blender_scene: "Scene"):
    """
    For each sampled ray, its origin should equal the c2w translation of the chosen frame.
    We can't read the internal RNG index directly, but we can verify that all origins
    lie on one of the known camera centers (within float tol).
    """
    scene = tiny_blender_scene
    centers = torch.stack(
        [torch.as_tensor(fr.c2w[:3, 3], dtype=torch.float32) for fr in scene.frames], dim=0
    )  # [N,3]

    sampler = RandomPixelRaySampler(scene, rays_per_batch=200, device="cpu")
    batch = next(iter(sampler))
    rays_o = batch["rays_o"]  # [B,3]

    # For each origin, check it's close to at least one known center
    diffs = rays_o[None, ...] - centers[:, None, :]  # [N,B,3]
    dists = torch.linalg.norm(diffs, dim=-1)         # [N,B]
    min_d = dists.min(dim=0).values                  # [B]
    assert torch.all(min_d < 1e-5)
