import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nerf_experiments.source.data.scene import Frame, Scene
from nerf_experiments.source.data.loaders.blender_loader import BlenderSceneLoader


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


def test_missing_transforms_raises(tmp_path: Path):
    loader = BlenderSceneLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load("train")


def test_load_basic_and_intrinsics_downscale(tmp_path: Path):
    # Create two simple RGB images (6x4) and transforms
    H, W = 4, 6
    img_dir = tmp_path
    img_names = ["r_000", "r_001"]
    for name in img_names:
        _write_png(img_dir / f"{name}.png", np.full((H, W, 3), 128, dtype=np.uint8))

    # Identity c2w poses
    T = np.eye(4, dtype=np.float32)
    transforms_path = tmp_path / "transforms_train.json"
    _make_transforms(transforms_path, [(n, T) for n in img_names], camera_angle_x=np.deg2rad(60))

    # Downscale by 2 → images become (2x3), intrinsics scaled
    loader = BlenderSceneLoader(tmp_path, downscale=2, white_bg=True)
    scene = loader.load("train")

    assert len(scene.frames) == 2
    assert scene.white_bg is True
    # Check image shapes and dtype range
    for fr in scene.frames:
        assert fr.image.shape[:2] == (H // 2, W // 2)
        assert fr.image.dtype == np.float32
        assert np.all(fr.image >= 0.0) and np.all(fr.image <= 1.0)

    # Intrinsics from FOV=60deg at original size, then divided by 2
    # fx = W / (2*tan(fov/2)) at W=6 → fx0
    fx0 = W / (2.0 * np.tan(np.deg2rad(60) * 0.5))
    K = scene.frames[0].K
    assert np.isclose(K[0, 0], fx0 / 2.0, rtol=1e-5)
    assert np.isclose(K[1, 1], fx0 / 2.0, rtol=1e-5)
    assert np.isclose(K[0, 2], (W * 0.5) / 2.0, rtol=1e-5)
    assert np.isclose(K[1, 2], (H * 0.5) / 2.0, rtol=1e-5)


def test_center_origin_recenters_poses(tmp_path: Path):
    # Two images
    H, W = 8, 8
    for name in ["a", "b"]:
        _write_png(tmp_path / f"{name}.png", np.zeros((H, W, 3), dtype=np.uint8))

    T1 = np.eye(4, dtype=np.float32); T1[:3, 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    T2 = np.eye(4, dtype=np.float32); T2[:3, 3] = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    _make_transforms(tmp_path / "transforms_train.json", [("a", T1), ("b", T2)])

    loader = BlenderSceneLoader(tmp_path, center_origin=True)
    scene = loader.load("train")

    centers = np.stack([fr.c2w[:3, 3] for fr in scene.frames], axis=0)
    mean_center = centers.mean(axis=0)
    assert np.allclose(mean_center, np.zeros(3), atol=1e-6)
    # Also check that the relative offset remains (should be ±1 around zero)
    assert np.allclose(centers[1] - centers[0], np.array([2.0, 0.0, 0.0]), atol=1e-6)


def test_scene_scale_scales_translations_and_bounds(tmp_path: Path):
    # One image
    H, W = 8, 8
    _write_png(tmp_path / "img.png", np.zeros((H, W, 3), dtype=np.uint8))

    T = np.eye(4, dtype=np.float32); T[:3, 3] = np.array([4.0, -2.0, 1.0], dtype=np.float32)
    _make_transforms(tmp_path / "transforms_train.json", [("img", T)])

    loader = BlenderSceneLoader(tmp_path, scene_scale=0.25)  # scale by 1/4
    scene = loader.load("train")

    t_scaled = scene.frames[0].c2w[:3, 3]
    assert np.allclose(t_scaled, np.array([1.0, -0.5, 0.25]), atol=1e-6)
    # near/far scaled
    assert np.isclose(scene.near, 2.0 * 0.25)
    assert np.isclose(scene.far, 6.0 * 0.25)


def test_bad_c2w_shape_raises(tmp_path: Path):
    H, W = 4, 4
    _write_png(tmp_path / "im.png", np.zeros((H, W, 3), dtype=np.uint8))
    bad = np.eye(3, dtype=np.float32)  # wrong shape (3x3)

    transforms = {
        "camera_angle_x": float(np.deg2rad(60)),
        "frames": [{"file_path": "./im", "transform_matrix": bad.tolist()}],
    }
    (tmp_path / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")

    loader = BlenderSceneLoader(tmp_path)
    with pytest.raises(ValueError):
        loader.load("train")


def test_resolve_image_path_adds_png(tmp_path: Path):
    loader = BlenderSceneLoader(tmp_path)
    # Create ./foo.png and ensure _resolve_image_path("./foo") yields that path
    _write_png(tmp_path / "foo.png", np.zeros((2, 2, 3), dtype=np.uint8))
    p = loader._resolve_image_path("./foo")
    assert p.suffix == ".png"
    assert p.exists()
