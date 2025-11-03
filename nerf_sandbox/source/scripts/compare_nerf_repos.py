#!/usr/bin/env python3
"""
NeRF repos comparison harness (device-safe version).

Key add-ons in this version:
- All tensors are coerced to the chosen --device immediately after creation.
- Optional --strict_device checks that raise if anything ends up on the wrong device.
- Consistent device handling for official + your code paths, world & NDC rays, PDF sampling,
  and optional end-to-end rendering.

See the bottom for run examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import random

# --------------------------------------------------------------------------------------
# Imports — YOUR repo
# --------------------------------------------------------------------------------------
from nerf_sandbox.source.data.loaders.blender_loader import BlenderSceneLoader as my_BlenderSceneLoader
from nerf_sandbox.source.data.loaders.llff_loader import LLFFSceneLoader as my_LLFFSceneLoader
from nerf_sandbox.source.data.scene import Scene
from nerf_sandbox.source.utils.ray_utils import get_camera_rays as my_get_camera_rays
from nerf_sandbox.source.utils.render_utils import (
    render_image_chunked as my_render_image_chunked,
)
from nerf_sandbox.source.models.encoders import get_vanilla_nerf_encoders as my_get_encoders
from nerf_sandbox.source.models.mlps import NeRF as MyNeRF
from nerf_sandbox.source.utils.sampling_utils import sample_pdf as my_sample_pdf

# --------------------------------------------------------------------------------------
# Imports — OFFICIAL repo (nerf-pytorch)
# --------------------------------------------------------------------------------------
from nerf_sandbox.source.scripts.nerf_pytorch.load_blender import load_blender_data as off_load_blender
from nerf_sandbox.source.scripts.nerf_pytorch.load_llff import load_llff_data as off_load_llff
from nerf_sandbox.source.scripts.nerf_pytorch.run_nerf_helpers import (
    get_rays as off_get_rays,
    ndc_rays as off_ndc_rays,
    sample_pdf as off_sample_pdf,
    get_embedder as off_get_embedder,
    NeRF as OffNeRF
)
from nerf_sandbox.source.scripts.nerf_pytorch.run_nerf import (
    render as off_render,
    run_network as off_run_network,
    raw2outputs as off_raw2outputs
)

# ======================================================================================
# Types & helpers
# ======================================================================================
class DeviceEnforcer:
    def __init__(self, device: torch.device, strict: bool = False):
        self.device = device
        self.strict = strict

    def move(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = []
        for t in tensors:
            if torch.is_tensor(t) and t.device != self.device:
                t = t.to(self.device, non_blocking=False)
            out.append(t)
        return tuple(out)

    def assert_on_device(self, name: str, *tensors: torch.Tensor) -> None:
        if not self.strict:
            return
        for t in tensors:
            if torch.is_tensor(t) and t.device != self.device:
                raise RuntimeError(f"[DeviceError] '{name}' on {t.device}, expected {self.device}.")

def _to_torch(x, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)

def _flatten_hw3(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-1])

def _as_c2w_4x4(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        c4 = np.eye(4, dtype=np.float32)
        c4[:3, :4] = m
        return c4
    raise ValueError(f"Unexpected c2w shape {m.shape}; expected (4,4) or (3,4).")

def _grade(x, warn, fail, higher_is_worse=True):
    """Return ('OK'|'WARN'|'FAIL') for a scalar x."""
    if x is None:
        return "OK"
    if higher_is_worse:
        return "FAIL" if x > fail else ("WARN" if x > warn else "OK")
    else:
        return "FAIL" if x < fail else ("WARN" if x < warn else "OK")


def _fmt_pct(x):
    return f"{100.0*x:.2f}%"

def _cuda_preflight(device: torch.device):
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    # trivial alloc + op
    _ = torch.randn(4, 4, device=device) @ torch.randn(4, 4, device=device)

@dataclass
class DatasetView:
    images: List[np.ndarray]
    Ks: List[np.ndarray]
    c2ws: List[np.ndarray]
    H: int
    W: int
    focal: float | None
    white_bkgd: bool
    name: str

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ======================================================================================
# Section 1 — Dataset builders
# ======================================================================================
def build_view_off_blender(root: str, half_res: bool, testskip: int, white_bkgd: bool) -> DatasetView:
    imgs, poses, render_poses, hwf, i_split = off_load_blender(root, half_res=half_res, testskip=testskip)
    H, W, focal = int(hwf[0]), int(hwf[1]), float(hwf[2])
    Ks: List[np.ndarray] = []
    c2ws: List[np.ndarray] = []
    for i in range(poses.shape[0]):
        K = np.array([[focal, 0.0, 0.5 * W],
                      [0.0,  focal, 0.5 * H],
                      [0.0,  0.0,   1.0]], dtype=np.float32)
        Ks.append(K)
        c4 = np.eye(4, dtype=np.float32); c4[:3, :4] = poses[i, :3, :4]
        c2ws.append(c4)
    return DatasetView(
        images=[imgs[i] for i in range(imgs.shape[0])],
        Ks=Ks, c2ws=c2ws,
        H=H, W=W, focal=focal,
        white_bkgd=bool(white_bkgd),
        name="official_blender",
    )

def build_view_my_blender(root: str, downscale: int, white_bkgd: bool, testskip: int) -> DatasetView:
    """
    Mirror the official blender loader behavior:
      - Load train, val, test
      - Apply testskip to val/test (train always uses skip=1 unless testskip==0)
      - Keep RGBA (no white compositing) to match official output shape
    """
    loader = my_BlenderSceneLoader(
        root=root,
        downscale=downscale,
        white_bkgd=white_bkgd,
        centering="none",
        scene_scale=1.0,
        composite_on_load=False,   # keep RGBA like official
    )

    all_imgs: List[np.ndarray] = []
    all_Ks:   List[np.ndarray] = []
    all_c2w:  List[np.ndarray] = []

    splits = ["train", "val", "test"]
    for s in splits:
        sc: Scene = loader.load(split=s)
        frames = sc.frames

        # Match official skip policy
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = max(1, int(testskip))

        frames = frames[::skip]

        for fr in frames:
            all_imgs.append(np.asarray(fr.image))
            all_Ks.append(np.asarray(fr.K, dtype=np.float32))
            c4 = _as_c2w_4x4(fr.c2w)
            all_c2w.append(c4)

    if len(all_imgs) == 0:
        raise RuntimeError("No frames loaded from Blender dataset (check paths).")

    H, W = all_imgs[0].shape[:2]
    focal = float(all_Ks[0][0, 0])

    return DatasetView(
        images=all_imgs,
        Ks=all_Ks,
        c2ws=all_c2w,
        H=H, W=W, focal=focal,
        white_bkgd=bool(white_bkgd),
        name="my_blender_all_splits",
    )


def build_view_off_llff(root: str, factor: int, recenter: bool, spherify: bool) -> DatasetView:
    images, poses, bds, render_poses, i_test = off_load_llff(root, factor=factor, recenter=recenter, bd_factor=.75, spherify=spherify)
    N = images.shape[0]
    H, W, focal = int(poses[0, 0, -1]), int(poses[0, 1, -1]), float(poses[0, 2, -1])
    Ks: List[np.ndarray] = []
    c2ws: List[np.ndarray] = []
    for i in range(N):
        Ki = np.array([[poses[i, 2, -1], 0.0, 0.5 * poses[i, 1, -1]],
                       [0.0, poses[i, 2, -1], 0.5 * poses[i, 0, -1]],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
        Ks.append(Ki)
        c4 = np.eye(4, dtype=np.float32); c4[:3, :4] = poses[i, :3, :4]
        c2ws.append(c4)
    return DatasetView(
        images=[images[i] for i in range(N)],
        Ks=Ks, c2ws=c2ws,
        H=H, W=W, focal=None,
        white_bkgd=False,
        name="official_llff",
    )

def build_view_my_llff(root: str, downscale: int, white_bkgd: bool) -> DatasetView:
    loader = my_LLFFSceneLoader(root=root, downscale=downscale, white_bkgd=white_bkgd,
                                bd_factor=0.75, use_llff_holdout=False)
    sc: Scene = loader.load(split="train")
    H, W = sc.frames[0].H, sc.frames[0].W
    return DatasetView(
        images=[np.asarray(fr.image) for fr in sc.frames],
        Ks=[np.asarray(fr.K, dtype=np.float32) for fr in sc.frames],
        c2ws=[np.asarray(fr.c2w, dtype=np.float32) for fr in sc.frames],
        H=H, W=W, focal=None,
        white_bkgd=bool(white_bkgd),
        name="my_llff",
    )

# ======================================================================================
# Section 2 — Intrinsics/pose comparison
# ======================================================================================
def compare_intrinsics_pose(a: DatasetView, b: DatasetView, atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    assert len(a.images) == len(b.images), f"Different #images: {len(a.images)} vs {len(b.images)}"
    N = len(a.images)
    results: Dict[str, Any] = {
        "num_images": N,
        "image_shape_a": [int(a.H), int(a.W)],
        "image_shape_b": [int(b.H), int(b.W)],
        "per_frame": [],
    }
    for i in range(N):
        Ki, Kj = a.Ks[i], b.Ks[i]
        cA, cB = a.c2ws[i], b.c2ws[i]
        same_K = np.allclose(Ki, Kj, atol=atol, rtol=rtol)
        same_pose = np.allclose(cA, cB, atol=atol, rtol=rtol)
        results["per_frame"].append({
            "i": i,
            "K_l1": float(np.abs(Ki - Kj).mean()),
            "K_maxabs": float(np.max(np.abs(Ki - Kj))),
            "pose_l1": float(np.abs(cA - cB).mean()),
            "pose_maxabs": float(np.max(np.abs(cA - cB))),
            "same_K": bool(same_K),
            "same_pose": bool(same_pose),
        })
    return results

# ======================================================================================
# Section 3 — Ray generation comparison
# ======================================================================================
def _sample_pixel_indices(H: int, W: int, n_max: int, seed: int) -> np.ndarray:
    total = H * W
    n = min(int(n_max), total) if n_max and n_max > 0 else total
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)
    yy = idx // W
    xx = idx % W
    return np.stack([xx, yy], axis=-1).astype(np.int64)  # (n,2) [x,y]

def make_official_rays_world(H: int, W: int, K: np.ndarray, c2w_3x4: np.ndarray, *, dev: DeviceEnforcer):
    c4 = np.eye(4, dtype=np.float32); c4[:3, :4] = c2w_3x4[:3, :4]
    ro_hw3_dev, rd_hw3_dev = _off_get_rays_safe(H, W, K, c4, dev=dev)
    return _flatten_hw3(ro_hw3_dev), torch.nn.functional.normalize(_flatten_hw3(rd_hw3_dev), dim=-1)


def make_official_rays_ndc(H: int, W: int, K: np.ndarray, near_plane: float,
                           rays_o_hw3: torch.Tensor, rays_d_hw3: torch.Tensor, *, dev: DeviceEnforcer):
    ro_ndc_hw3_dev, rd_ndc_hw3_dev = _off_ndc_rays_safe(H, W, K, near_plane, rays_o_hw3, rays_d_hw3, dev=dev)
    return _flatten_hw3(ro_ndc_hw3_dev), torch.nn.functional.normalize(_flatten_hw3(rd_ndc_hw3_dev), dim=-1)


def _off_get_rays_safe(H: int, W: int, K_np: np.ndarray, c2w_4x4_np: np.ndarray, *, dev: DeviceEnforcer):
    """Call official get_rays on CPU (it builds CPU tensors), then move results to dev.device."""
    K_cpu = _to_torch(K_np, device="cpu")
    c_cpu = _to_torch(c2w_4x4_np[:3, :4], device="cpu")
    ro_cpu, rd_cpu = off_get_rays(H, W, K_cpu, c_cpu)           # CPU inside official code
    ro_dev, rd_dev = dev.move(ro_cpu, rd_cpu)                   # now on chosen device
    dev.assert_on_device("off_get_rays_safe", ro_dev, rd_dev)
    return ro_dev, rd_dev


def _off_ndc_rays_safe(H: int, W: int, K_np: np.ndarray, near_plane: float,
                       rays_o_hw3_dev: torch.Tensor, rays_d_hw3_dev: torch.Tensor, *, dev: DeviceEnforcer):
    """Same idea for official ndc_rays: hop to CPU for the math, then back to dev.device."""
    ro_cpu = rays_o_hw3_dev.to("cpu")
    rd_cpu = rays_d_hw3_dev.to("cpu")
    focal = float(K_np[0, 0])
    ro_ndc_cpu, rd_ndc_cpu = off_ndc_rays(H, W, focal, float(near_plane), ro_cpu, rd_cpu)
    ro_ndc_dev, rd_ndc_dev = dev.move(ro_ndc_cpu, rd_ndc_cpu)
    dev.assert_on_device("off_ndc_rays_safe", ro_ndc_dev, rd_ndc_dev)
    return ro_ndc_dev, rd_ndc_dev

def _off_sample_pdf_safe(bins_edges_dev: torch.Tensor,
                         weights_dev: torch.Tensor,
                         n_samples: int,
                         *,
                         det: bool,
                         dev: DeviceEnforcer) -> torch.Tensor:
    """Call official sample_pdf safely on CPU, then move result back to dev.device."""
    be_cpu = bins_edges_dev.to("cpu", non_blocking=False)
    wt_cpu = weights_dev.to("cpu", non_blocking=False)
    out_cpu = off_sample_pdf(be_cpu, wt_cpu, n_samples, det=det)  # u is CPU inside
    out_dev, = dev.move(out_cpu)
    dev.assert_on_device("off_sample_pdf_safe(output)", out_dev)
    return out_dev


def make_my_rays(H: int, W: int, K: np.ndarray, c2w_4x4: np.ndarray, *, dev: DeviceEnforcer, as_ndc: bool, near_plane: float) -> Tuple[torch.Tensor, torch.Tensor]:
    (
        rays_o_world,
        rays_d_world_unit,
        _r_dw_norm,
        rays_o_march,
        rays_d_march_unit,
        _r_dm_norm,
    ) = my_get_camera_rays(
        image_h=H, image_w=W,
        intrinsic_matrix=K, transform_camera_to_world=c2w_4x4,
        device=dev.device, dtype=torch.float32,
        convention="opengl", pixel_center=False,
        as_ndc=bool(as_ndc), near_plane=float(near_plane),
        pixels_xy=None,
    )
    # Ensure on device (defensive; your function should already honor 'device')
    rays_o_world, rays_d_world_unit = dev.move(rays_o_world, rays_d_world_unit)
    rays_o_march, rays_d_march_unit = dev.move(rays_o_march, rays_d_march_unit)
    if as_ndc:
        dev.assert_on_device("my_get_camera_rays(ndc)", rays_o_march, rays_d_march_unit)
        return _flatten_hw3(rays_o_march), _flatten_hw3(rays_d_march_unit)
    else:
        dev.assert_on_device("my_get_camera_rays(world)", rays_o_world, rays_d_world_unit)
        return _flatten_hw3(rays_o_world), _flatten_hw3(rays_d_world_unit)

def compare_rays_per_image(H: int, W: int, K: np.ndarray, c2w_4x4: np.ndarray, *, dev: DeviceEnforcer, n_pixels: int, ndc: bool, ndc_near_plane: float, seed: int) -> Dict[str, Any]:
    # Official WORLD rays
    r_o_hw3_off, r_d_hw3_off = _off_get_rays_safe(H, W, K, c2w_4x4, dev=dev)
    r_o_hw3_off, r_d_hw3_off = dev.move(r_o_hw3_off, r_d_hw3_off)
    dev.assert_on_device("off_get_rays(world)", r_o_hw3_off, r_d_hw3_off)

    r_o_flat_off = _flatten_hw3(r_o_hw3_off)
    r_d_flat_off = torch.nn.functional.normalize(_flatten_hw3(r_d_hw3_off), dim=-1)

    if ndc:
        ro_off, rd_off = make_official_rays_ndc(H, W, K, near_plane=ndc_near_plane, rays_o_hw3=r_o_hw3_off, rays_d_hw3=r_d_hw3_off, dev=dev)
        ro_my,  rd_my  = make_my_rays(H, W, K, c2w_4x4, dev=dev, as_ndc=True,  near_plane=ndc_near_plane)
    else:
        ro_off, rd_off = r_o_flat_off, r_d_flat_off
        ro_my,  rd_my  = make_my_rays(H, W, K, c2w_4x4, dev=dev, as_ndc=False, near_plane=ndc_near_plane)

    # Same pixel subset
    xy  = _sample_pixel_indices(H, W, n_pixels, seed)
    lin = torch.as_tensor(xy[:, 1] * W + xy[:, 0], device=dev.device, dtype=torch.long)

    ro_off_s = ro_off.index_select(0, lin)
    rd_off_s = rd_off.index_select(0, lin)
    ro_my_s  = ro_my.index_select(0, lin)
    rd_my_s  = rd_my.index_select(0, lin)
    dev.assert_on_device("ray_subset", ro_off_s, rd_off_s, ro_my_s, rd_my_s)

    eps = 1e-9
    cos = torch.sum(rd_off_s * rd_my_s, dim=-1).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(cos + 0.0))
    dir_l2 = torch.norm(rd_off_s - rd_my_s, dim=-1)
    ori_l2 = torch.norm(ro_off_s - ro_my_s, dim=-1)

    return {
        "n_pixels": int(lin.shape[0]),
        "dir_cos_mean": float(cos.mean().item()),
        "dir_ang_deg_mean": float(ang.mean().item()),
        "dir_ang_deg_p95": float(ang.quantile(0.95).item()),
        "dir_l2_mean": float(dir_l2.mean().item()),
        "ori_l2_mean": float(ori_l2.mean().item()),
        "ori_l2_p95": float(ori_l2.quantile(0.95).item()),
    }

# ======================================================================================
# Section 4 — PDF sampling parity
# ======================================================================================

def _call_my_sample_pdf_compat(bins_edges_t: torch.Tensor,
                               weights_t: torch.Tensor,
                               n_samples: int,
                               *,
                               dev: DeviceEnforcer) -> torch.Tensor:
    bins_edges_t, weights_t = dev.move(bins_edges_t, weights_t)
    bins_edges_t = bins_edges_t.contiguous()
    weights_t    = weights_t.contiguous()

    # Prefer EDGES→WEIGHTS(M) first
    try:
        out = my_sample_pdf(bins_edges_t, weights_t, n_samples, deterministic=True)
        path = "edges"
    except Exception:
        # Fall back to MIDPOINTS→WEIGHTS(M)
        bins_mid_t = 0.5 * (bins_edges_t[..., 1:] + bins_edges_t[..., :-1]).contiguous()
        out = my_sample_pdf(bins_mid_t, weights_t, n_samples, deterministic=True)
        path = "midpoints"

    if isinstance(out, (tuple, list)):
        out = out[0]
    out, = dev.move(out)
    dev.assert_on_device("my_sample_pdf(output)", out)

    # Optional one-liner debug (won’t spam; easy to grep)
    # print(f"[pdf-compat] used {path}: bins={bins_edges_t.shape}→out={out.shape}")
    return out


def pdf_sampling_parity_test(*, dev: DeviceEnforcer,
                             batch: int = 8, n_bins: int = 63,
                             n_samples: int = 64, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    edges_np = np.sort(rng.random((batch, n_bins + 1)).astype(np.float32), axis=-1)
    w_np     = rng.random((batch, n_bins)).astype(np.float32)

    edges_t = torch.as_tensor(edges_np, device=dev.device).contiguous()
    w_t     = torch.as_tensor(w_np,    device=dev.device).contiguous()

    samp_off = _off_sample_pdf_safe(edges_t, w_t, n_samples, det=True, dev=dev)
    samp_off, = dev.move(samp_off)
    dev.assert_on_device("off_sample_pdf(output)", samp_off)

    samp_my = _call_my_sample_pdf_compat(edges_t, w_t, n_samples, dev=dev)

    diff = (samp_off - samp_my).abs()
    return {
        "batch": batch,
        "n_bins": n_bins,
        "n_samples": n_samples,
        "l1_mean": float(diff.mean().item()),
        "l2_mean": float(torch.sqrt((diff**2).mean()).item()),
        "max_abs": float(diff.max().item()),
    }

# ======================================================================================
# Section 5 — MLP architecture comparison
# ======================================================================================
def build_official_mlp(use_viewdirs: bool, multires_xyz: int, multires_dir: int, D: int, W: int, N_importance: int, *, dev: DeviceEnforcer) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    embed_fn, input_ch = off_get_embedder(multires_xyz, i=0)
    embeddirs_fn, input_ch_views = (off_get_embedder(multires_dir, i=0) if use_viewdirs else (None, 0))
    output_ch = 5 if (N_importance and N_importance > 0) else 4
    model = OffNeRF(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views, output_ch=output_ch, skips=[4], use_viewdirs=use_viewdirs).to(dev.device)

    arch = {
        "D": D, "W": W,
        "input_ch": input_ch,
        "input_ch_views": input_ch_views,
        "use_viewdirs": bool(use_viewdirs),
        "pts_linears": [(l.in_features, l.out_features) for l in model.pts_linears],
        "views_linears": [(l.in_features, l.out_features) for l in model.views_linears],
        "feature_linear": (model.feature_linear.in_features, model.feature_linear.out_features) if hasattr(model, "feature_linear") else None,
        "alpha_linear": (model.alpha_linear.in_features, model.alpha_linear.out_features) if hasattr(model, "alpha_linear") else None,
        "rgb_linear": (model.rgb_linear.in_features, model.rgb_linear.out_features) if hasattr(model, "rgb_linear") else None,
    }
    return model, embed_fn, embeddirs_fn, arch

def build_my_mlp(D: int, W: int, *, dev: DeviceEnforcer) -> Tuple[Any, Any, Dict[str, Any]]:
    pos_enc, dir_enc = my_get_encoders()
    pos_enc = pos_enc.to(dev.device)
    dir_enc = dir_enc.to(dev.device)
    nerf = MyNeRF(enc_pos_dim=pos_enc.out_dim, enc_dir_dim=dir_enc.out_dim, n_layers=D, hidden_dim=W, skip_pos=4).to(dev.device)
    arch = {
        "enc_pos_dim": int(pos_enc.out_dim),
        "enc_dir_dim": int(dir_enc.out_dim),
        "n_layers": int(D), "hidden_dim": int(W), "skip_pos": 4,
        "mlp": [(layer.in_features, layer.out_features) for layer in nerf.mlp],
        "feature": (nerf.feature.in_features, nerf.feature.out_features),
        "sigma_out": (nerf.sigma_out.in_features, nerf.sigma_out.out_features),
        "color_fc": (nerf.color_fc.in_features, nerf.color_fc.out_features),
        "color_out": (nerf.color_out.in_features, nerf.color_out.out_features),
    }
    return nerf, (pos_enc, dir_enc), arch

# ======================================================================================
# Section 6 — End-to-end RGB on sampled rays (optional)
# ======================================================================================

# --- Weight tie: official OffNeRF -> your MyNeRF ---
@torch.no_grad()
def tie_official_into_mine(off_model: OffNeRF, my_model: MyNeRF) -> None:
    """
    Copy weights from the official OffNeRF into your MyNeRF, handling the
    skip-layer index offset (official skip at idx=5, yours at idx=4 with D=8).
    Works in general by detecting the unique W+input_ch wide layer on each side.
    Copies across devices/dtypes safely.
    """
    # --- helpers ---
    def _copy_linear(src_l, dst_l, target_dev, target_dtype):
        dst_l.weight.copy_(src_l.weight.to(device=target_dev, dtype=target_dtype))
        dst_l.bias.copy_(src_l.bias.to(device=target_dev, dtype=target_dtype))

    # devices/dtypes
    my_dev   = next(my_model.parameters()).device
    my_dtype = next(my_model.parameters()).dtype

    # dims
    D_off = len(off_model.pts_linears)
    D_my  = len(my_model.mlp)
    assert D_off == D_my, f"Layer count mismatch: off={D_off}, mine={D_my}"

    W_off = off_model.W
    W_my  = my_model.mlp[0].out_features
    in_off = off_model.input_ch            # official positional-encoder dim
    in_my  = my_model.mlp[0].in_features   # your positional-encoder dim

    # find the single wide (skip) layer on each side (in_features == W + input_ch)
    def _find_skip_idx_off():
        for idx, l in enumerate(off_model.pts_linears):
            if l.in_features == (W_off + in_off):
                return idx
        raise AssertionError("Could not find official skip layer (W+input_ch).")

    def _find_skip_idx_my():
        for idx, l in enumerate(my_model.mlp):
            if l.in_features == (W_my + in_my):
                return idx
        raise AssertionError("Could not find my skip layer (W+input_ch).")

    s_off = _find_skip_idx_off()
    s_my  = _find_skip_idx_my()

    # sanity
    assert W_off == W_my, f"Hidden width mismatch: off={W_off}, mine={W_my}"
    assert in_off == in_my, f"Pos enc dim mismatch: off={in_off}, mine={in_my}"

    # --- 1) prefix: layers BEFORE the skip (one-to-one up to the earlier skip) ---
    prefix = min(s_off, s_my)
    for i in range(prefix):
        l_off, l_my = off_model.pts_linears[i], my_model.mlp[i]
        # shapes must match here
        assert (l_off.in_features, l_off.out_features) == (l_my.in_features, l_my.out_features), \
            f"prefix layer {i} shape mismatch: off {l_off.in_features}->{l_off.out_features} vs my {l_my.in_features}->{l_my.out_features}"
        _copy_linear(l_off, l_my, my_dev, my_dtype)

    # --- 2) the skip layers themselves ---
    _copy_linear(off_model.pts_linears[s_off], my_model.mlp[s_my], my_dev, my_dtype)

    # --- 3) the remaining layers AFTER we've accounted for the skip ---
    # Build the remaining index lists in-order, excluding the skip index on each side.
    off_rest = [i for i in range(prefix, D_off) if i != s_off]
    my_rest  = [i for i in range(prefix, D_my)  if i != s_my]

    # They can differ by one in relative position; we still map in-order.
    assert len(off_rest) == len(my_rest), f"Tail length mismatch: off={len(off_rest)} vs my={len(my_rest)}"

    for io, im in zip(off_rest, my_rest):
        l_off, l_my = off_model.pts_linears[io], my_model.mlp[im]
        # All remaining should be plain W->W layers
        assert l_off.out_features == l_my.out_features == W_my, \
            f"out_features mismatch: off[{io}]={l_off.out_features}, my[{im}]={l_my.out_features}"
        assert l_off.in_features == l_my.in_features == W_my, \
            f"in_features mismatch: off[{io}]={l_off.in_features}, my[{im}]={l_my.in_features}"
        _copy_linear(l_off, l_my, my_dev, my_dtype)

    # --- 4) feature/alpha/view/rgb heads ---
    _copy_linear(off_model.feature_linear, my_model.feature,    my_dev, my_dtype)
    _copy_linear(off_model.alpha_linear,   my_model.sigma_out,  my_dev, my_dtype)

    v_off = off_model.views_linears[0]
    _copy_linear(v_off,                  my_model.color_fc,   my_dev, my_dtype)
    _copy_linear(off_model.rgb_linear,   my_model.color_out,  my_dev, my_dtype)



def _off_raw2outputs_safe(raw_dev: torch.Tensor,
                          z_vals_dev: torch.Tensor,
                          rays_d_dev: torch.Tensor,
                          white_bkgd: bool,
                          dev: DeviceEnforcer):
    """
    Call official raw2outputs safely on CPU, then move results to dev.device.
    Supports variants that return 4 or 5 tensors.
    Returns: rgb, disp, acc, weights, depth
    """
    # move inputs to CPU for the official helper (it allocates CPU tensors inside)
    raw_cpu    = raw_dev.to("cpu", non_blocking=False)
    z_vals_cpu = z_vals_dev.to("cpu", non_blocking=False)
    rays_d_cpu = rays_d_dev.to("cpu", non_blocking=False)

    out = off_raw2outputs(raw_cpu, z_vals_cpu, rays_d_cpu,
                          raw_noise_std=0, white_bkgd=bool(white_bkgd), pytest=False)

    if isinstance(out, (tuple, list)):
        if len(out) == 5:
            rgb_cpu, disp_cpu, acc_cpu, weights_cpu, depth_cpu = out
        elif len(out) == 4:
            rgb_cpu, disp_cpu, acc_cpu, weights_cpu = out
            # synthesize depth same as official formula
            depth_cpu = torch.sum(weights_cpu * z_vals_cpu, dim=-1)
        else:
            raise RuntimeError(f"Unexpected raw2outputs return length: {len(out)}")
    else:
        raise RuntimeError("raw2outputs returned a non-sequence result")

    rgb, disp, acc, weights, depth = dev.move(rgb_cpu, disp_cpu, acc_cpu, weights_cpu, depth_cpu)
    dev.assert_on_device("off_raw2outputs_safe(outputs)", rgb, disp, acc, weights, depth)
    return rgb, disp, acc, weights, depth

@torch.no_grad()
def integrator_parity_test(*,
                           dev: DeviceEnforcer,
                           n_rays: int = 256,
                           n_samples: int = 64,
                           white_bkgd: bool = True,
                           near: float = 2.0,
                           far: float = 6.0,
                           seed: int = 0) -> Dict[str, float]:
    rng = torch.Generator(device=dev.device).manual_seed(seed)

    # z_vals and rays_d on the chosen device
    z_vals = torch.linspace(near, far, steps=n_samples, device=dev.device).expand(n_rays, n_samples)
    rays_d = torch.ones(n_rays, 3, device=dev.device)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

    # Controlled raw: rgb in [0,1], sigma >= 0
    rgb = torch.sigmoid(torch.randn(n_rays, n_samples, 3, generator=rng, device=dev.device))
    sigma = torch.relu(torch.randn(n_rays, n_samples, 1, generator=rng, device=dev.device) * 0.2)
    # Make official raw2outputs see the exact same RGB we use (it applies sigmoid to raw[..., :3])
    eps = 1e-6
    rgb_clamped = rgb.clamp(eps, 1 - eps)
    logit_rgb = torch.log(rgb_clamped) - torch.log(1.0 - rgb_clamped)  # numerically stable logit
    # Keep sigma as-is (official uses ReLU on raw[..., 3], and sigma ≥ 0 already)
    raw = torch.cat([logit_rgb, sigma], dim=-1)

    # Official integrator (CPU-safe wrapper)
    rgb_o, disp_o, acc_o, w_o, depth_o = _off_raw2outputs_safe(raw, z_vals, rays_d, white_bkgd, dev)

    # Your integrator
    from nerf_sandbox.source.utils.render_utils import volume_render_rays as my_vol
    rgb_m, w_m, acc_m, depth_m = my_vol(
        rgb=rgb,
        sigma=sigma.squeeze(-1),
        z_depths=z_vals,
        white_bkgd=bool(white_bkgd),
        infinite_last_bin=True,          # <<< match official
    )
    assert torch.allclose(w_m.sum(dim=-1), acc_m.squeeze(-1), atol=1e-6), "acc ≠ sum(weights) on my path"

    # Compare apples-to-apples: official depth_o vs. your unnormalized depth
    depth_m_sum = torch.sum(w_m * z_vals, dim=-1)               # [N]
    depth_l1 = float(torch.mean(torch.abs(depth_o - depth_m_sum)).item())

    # Compare on device, then move to CPU for numpy diffs
    def _l1(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.mean(torch.abs(a.reshape(n_rays, -1) - b.reshape(n_rays, -1))).item())

    return {
        "rgb_l1":   _l1(rgb_o,   rgb_m),
        "acc_l1":   _l1(acc_o,   acc_m),
        "depth_l1": depth_l1,
        "w_l1":     _l1(w_o,     w_m),
    }

def render_official_on_rays(
    H: int, W: int, K: np.ndarray, c2w_4x4: np.ndarray, *,
    dev: DeviceEnforcer,
    use_ndc: bool, near: float, far: float,
    n_coarse: int, n_fine: int,
    netchunk: int, chunk_rays: int,
    white_bkgd: bool, n_rays: int, seed: int,
    # optional reuse (for weight-tie / parity)
    model_c: OffNeRF | None = None,
    model_f: OffNeRF | None = None,
    embed_fn=None,
    embeddirs_fn=None,
):
    """
    OFFICIAL path rendered on CPU; returns stats and (model_c, model_f, embed_fn, embeddirs_fn)
    so the caller can reuse them across images.
    """
    # -- WORLD rays on CPU via official helper --
    Kt_cpu = _to_torch(K, device="cpu")
    c_cpu  = _to_torch(c2w_4x4[:3, :4], device="cpu")
    ro_hw3_cpu, rd_hw3_cpu = off_get_rays(H, W, Kt_cpu, c_cpu)

    ro_cpu = _flatten_hw3(ro_hw3_cpu)
    rd_cpu = _flatten_hw3(rd_hw3_cpu)

    # -- choose subset (CPU) --
    xy = _sample_pixel_indices(H, W, n_rays, seed)
    lin_cpu = torch.as_tensor(xy[:, 1] * W + xy[:, 0], device="cpu", dtype=torch.long)
    ro_s_cpu = ro_cpu.index_select(0, lin_cpu)
    rd_s_cpu = rd_cpu.index_select(0, lin_cpu)
    batch_rays_cpu = torch.stack([ro_s_cpu, rd_s_cpu], dim=0)  # (2, N, 3) CPU

    # -- encoders & models (CPU) --
    use_viewdirs = True

    def _infer_embed_out_dim(fn) -> int:
        with torch.no_grad():
            return int(fn(torch.zeros(1, 3)).shape[-1])

    if embed_fn is None or embeddirs_fn is None:
        embed_fn, input_ch = off_get_embedder(10, i=0)
        embeddirs_fn, input_ch_views = off_get_embedder(4, i=0)
    else:
        input_ch = _infer_embed_out_dim(embed_fn)
        input_ch_views = _infer_embed_out_dim(embeddirs_fn)

    output_ch = 5 if (n_fine and n_fine > 0) else 4
    if model_c is None:
        model_c = OffNeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,
                          output_ch=output_ch, skips=[4], use_viewdirs=use_viewdirs)
    if (n_fine and n_fine > 0) and model_f is None:
        model_f = OffNeRF(D=8, W=256, input_ch=input_ch, input_ch_views=input_ch_views,
                          output_ch=output_ch, skips=[4], use_viewdirs=use_viewdirs)

    model_c = model_c.to("cpu")
    if model_f is not None:
        model_f = model_f.to("cpu")

    def network_query_fn(inputs, viewdirs, network_fn):
        return off_run_network(inputs, viewdirs, network_fn,
                               embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
                               netchunk=netchunk)

    render_kwargs = {
        "network_query_fn": network_query_fn,
        "perturb": 0.0,
        "N_importance": int(n_fine or 0),
        "network_fine": model_f,
        "N_samples": int(n_coarse),
        "network_fn": model_c,
        "white_bkgd": bool(white_bkgd),
        "raw_noise_std": 0.0,
        "lindisp": False,
    }

    with torch.no_grad():
        rgb_cpu, disp_cpu, acc_cpu, extras = off_render(
            H, W, Kt_cpu, chunk=chunk_rays, rays=batch_rays_cpu, c2w=None,
            ndc=bool(use_ndc), near=float(near), far=float(far),
            use_viewdirs=use_viewdirs, **render_kwargs
        )

    # move outputs back to target device for uniform handling
    rgb_dev, disp_dev, acc_dev = dev.move(rgb_cpu, disp_cpu, acc_cpu)
    dev.assert_on_device("off_render(outputs)", rgb_dev, disp_dev, acc_dev)

    rgb_flat  = rgb_dev.reshape(-1, 3).cpu().numpy()
    acc_flat  = acc_dev.reshape(-1).cpu().numpy()
    disp_flat = disp_dev.reshape(-1).cpu().numpy()

    stats = {
        "n_rays": int(rgb_flat.shape[0]),
        "rgb_mean": [float(rgb_flat[:, i].mean()) for i in range(3)],
        "rgb_std":  [float(rgb_flat[:, i].std())  for i in range(3)],
        "acc_mean": float(acc_flat.mean()),
        "acc_std":  float(acc_flat.std()),
        "disp_mean": float(disp_flat.mean()),
        "disp_std":  float(disp_flat.std()),
    }
    # IMPORTANT: return exactly five values (matches your unpack)
    return stats, model_c, model_f, embed_fn, embeddirs_fn




def render_my_on_rays(
    H: int, W: int, K: np.ndarray, c2w_4x4: np.ndarray, *,
    dev: DeviceEnforcer,
    use_ndc: bool, near: float, far: float,
    n_coarse: int, n_fine: int,
    chunk_pts: int,
    white_bkgd: bool,
    n_rays: int, seed: int,
    # optional reuse
    nerf_c: MyNeRF | None = None,
    nerf_f: MyNeRF | None = None,
    pos_enc=None,
    dir_enc=None,
) -> Dict[str, Any]:
    """
    YOUR path on dev.device. Returns only stats (your call site expects a single dict).
    """
    (
        r_o_world,
        r_d_world_unit,
        r_d_world_norm,
        r_o_march,
        r_d_march_unit,
        r_d_march_norm,
    ) = my_get_camera_rays(
        image_h=H, image_w=W,
        intrinsic_matrix=K, transform_camera_to_world=c2w_4x4,
        device=dev.device, dtype=torch.float32,
        convention="opengl", pixel_center=False,
        as_ndc=bool(use_ndc), near_plane=(1.0 if use_ndc else near), pixels_xy=None,
    )
    r_o_world, r_d_world_unit, r_d_world_norm = dev.move(r_o_world, r_d_world_unit, r_d_world_norm)
    r_o_march, r_d_march_unit, r_d_march_norm = dev.move(r_o_march, r_d_march_unit, r_d_march_norm)

    viewdirs_world = r_d_world_unit
    if use_ndc:
        rays_o, rays_d_u, ray_norms = r_o_march, r_d_march_unit, r_d_march_norm
        s_near, s_far = 0.0, 1.0
    else:
        rays_o, rays_d_u, ray_norms = r_o_world, r_d_world_unit, r_d_world_norm
        s_near, s_far = near, far

    xy  = _sample_pixel_indices(H, W, n_rays, seed)
    lin = torch.as_tensor(xy[:, 1] * W + xy[:, 0], device=dev.device, dtype=torch.long)
    ro_s = rays_o.reshape(-1, 3).index_select(0, lin)
    rd_s = rays_d_u.reshape(-1, 3).index_select(0, lin)
    rn_s = ray_norms.reshape(-1, 1).index_select(0, lin)
    vdir_s = viewdirs_world.reshape(-1, 3).index_select(0, lin)
    dev.assert_on_device("my_rays_subset", ro_s, rd_s, rn_s, vdir_s)

    if (pos_enc is None) or (dir_enc is None):
        pos_enc, dir_enc = my_get_encoders()
    pos_enc = pos_enc.to(dev.device)
    dir_enc = dir_enc.to(dev.device)

    if nerf_c is None:
        nerf_c = MyNeRF(enc_pos_dim=pos_enc.out_dim, enc_dir_dim=dir_enc.out_dim,
                        n_layers=8, hidden_dim=256, skip_pos=4).to(dev.device)
    else:
        nerf_c = nerf_c.to(dev.device)

    if (n_fine and n_fine > 0):
        if nerf_f is None:
            nerf_f = MyNeRF(enc_pos_dim=pos_enc.out_dim, enc_dir_dim=dir_enc.out_dim,
                            n_layers=8, hidden_dim=256, skip_pos=4).to(dev.device)
        else:
            nerf_f = nerf_f.to(dev.device)
    else:
        nerf_f = None

    with torch.no_grad():
        res = my_render_image_chunked(
            rays_o=ro_s, rays_d_unit=rd_s, ray_norms=rn_s,
            H=1, W=int(ro_s.shape[0]),
            near=s_near, far=s_far,
            pos_enc=pos_enc, dir_enc=dir_enc,
            nerf_c=nerf_c, nerf_f=nerf_f,
            nc_eval=int(n_coarse), nf_eval=int(n_fine or 0),
            white_bkgd=bool(white_bkgd), device=dev.device,
            eval_chunk=max(1024, int(chunk_pts)), perturb=False,
            viewdirs_world_unit=vdir_s,
            infinite_last_bin=True,
        )

    rgb_t, acc_t, depth_t = res["rgb"], res["acc"], res["depth"]
    rgb_t, acc_t, depth_t = dev.move(rgb_t, acc_t, depth_t)
    dev.assert_on_device("my_render_image_chunked", rgb_t, acc_t, depth_t)

    rgb   = rgb_t.reshape(-1, 3).cpu().numpy()
    acc   = acc_t.reshape(-1).cpu().numpy()
    depth = depth_t.reshape(-1).cpu().numpy()

    return {
        "n_rays": int(rgb.shape[0]),
        "rgb_mean": [float(rgb[:, i].mean()) for i in range(3)],
        "rgb_std":  [float(rgb[:, i].std())  for i in range(3)],
        "acc_mean": float(acc.mean()),
        "acc_std":  float(acc.std()),
        "depth_mean": float(depth.mean()),
        "depth_std":  float(depth.std()),
    }


# ======================================================================================
# Section 7 — NDC comparison
# ======================================================================================

@torch.no_grad()
def check_ndc_ray_parity(
    H, W, K_np, c2w_np,
    *,
    near_world: float,
    n_rays: int,
    seed: int,
    dev,
    pixel_center: bool = False,       # keep default matching nerf-pytorch (no +0.5)
    topk_print: int = 5,
    angle_thr_deg: float = 0.05,
    origin_thr: float = 1e-6,
    norm_thr: float = 1e-6,
) -> dict:
    """
    Compare official NDC rays (off_get_rays -> off_ndc_rays) with your my_get_camera_rays(as_ndc=True).

    Returns a dict with metrics + pass/fail flags. Always logs and returns; never raises.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    HW = H * W
    n  = min(int(n_rays), HW)

    # --- OFFICIAL: world -> ndc ---
    Kt_cpu = torch.as_tensor(K_np, dtype=torch.float32, device="cpu")
    c_cpu  = torch.as_tensor(c2w_np[:3, :4], dtype=torch.float32, device="cpu")

    ro_hw3_cpu, rd_hw3_cpu = off_get_rays(H, W, Kt_cpu, c_cpu)     # world
    ro_cpu = ro_hw3_cpu.reshape(-1, 3).contiguous()                # (HW,3)
    rd_cpu = rd_hw3_cpu.reshape(-1, 3).contiguous()                # (HW,3)

    fx = float(K_np[0, 0])  # nerf-pytorch uses scalar focal→use fx
    ro_ndc_off, rd_ndc_off = off_ndc_rays(H, W, fx, float(near_world), ro_cpu, rd_cpu)  # (HW,3)

    # --- YOUR: direct ndc ---
    (
        _oW, _dWu, _dWn,
        ro_ndc_my, rd_ndc_unit_my, rd_ndc_norm_my
    ) = my_get_camera_rays(
        image_h=H, image_w=W,
        intrinsic_matrix=K_np, transform_camera_to_world=c2w_np,
        device=dev.device, dtype=torch.float32,
        convention="opengl",
        pixel_center=bool(pixel_center),
        as_ndc=True, near_plane=float(near_world),
        pixels_xy=None
    )
    ro_ndc_my  = ro_ndc_my.reshape(-1, 3).to("cpu")
    rd_u_my    = rd_ndc_unit_my.reshape(-1, 3).to("cpu")
    rn_my      = rd_ndc_norm_my.reshape(-1, 1).to("cpu")

    # --- identical random subset ---
    idx = torch.randperm(HW, generator=rng)[:n]
    ro_off = ro_ndc_off.index_select(0, idx)
    rd_off = rd_ndc_off.index_select(0, idx)
    ro_my  = ro_ndc_my.index_select(0, idx)
    rd_u_m = rd_u_my.index_select(0, idx)
    rn_m   = rn_my.index_select(0, idx)

    # normalize official dirs in NDC
    rd_u_off = torch.nn.functional.normalize(rd_off, dim=-1)
    rd_n_off = rd_off.norm(dim=-1, keepdim=True)

    # metrics
    def deg(a, b):
        dot = (a * b).sum(-1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(dot))

    ang   = deg(rd_u_off, rd_u_m)               # (n,)
    o_l2  = (ro_off - ro_my).norm(dim=-1)       # (n,)
    n_abs = (rd_n_off - rn_m).abs().squeeze(-1) # (n,)

    # robust p95
    def p95(x):
        if len(x) == 0: return 0.0
        k = max(1, int(0.95 * len(x)))
        return float(x.kthvalue(k).values.item())

    out = {
        "N": int(n),
        "dir_angle_deg_mean": float(ang.mean().item()),
        "dir_angle_deg_p95":  p95(ang),
        "origin_L2_mean":     float(o_l2.mean().item()),
        "origin_L2_p95":      p95(o_l2),
        "norm_absdiff_mean":  float(n_abs.mean().item()),
        "norm_absdiff_p95":   p95(n_abs),
        "thresholds": {
            "dir_angle_deg_p95": angle_thr_deg,
            "origin_L2_p95": origin_thr,
            "norm_absdiff_p95": norm_thr,
        },
    }
    out["pass_dir_angle"] = (out["dir_angle_deg_p95"] <= angle_thr_deg)
    out["pass_origin"]    = (out["origin_L2_p95"]     <= origin_thr)
    out["pass_norm"]      = (out["norm_absdiff_p95"]  <= norm_thr)
    out["pass_all"]       = bool(out["pass_dir_angle"] and out["pass_origin"] and out["pass_norm"])

    status = "PASS" if out["pass_all"] else "FAIL"
    print(
        f"[NDC parity {status}] N={out['N']} | "
        f"ang_p95={out['dir_angle_deg_p95']:.4f}° (≤{angle_thr_deg})  "
        f"oL2_p95={out['origin_L2_p95']:.3e} (≤{origin_thr:.0e})  "
        f"|d|Δ_p95={out['norm_absdiff_p95']:.3e} (≤{norm_thr:.0e})"
    )

    # Log the worst offenders (top-k by angle) with pixel coords + a few values
    if topk_print and n > 0:
        k = min(int(topk_print), int(n))
        worst = torch.topk(ang, k=k, largest=True).indices
        xs = (idx % W)[worst]
        ys = (idx // W)[worst]
        print("  [NDC worst] idx  (x,y)      ang_deg   oL2       |d|_off   |d|_my")
        for j in range(k):
            ii = int(worst[j])
            print(f"  [#{j+1:02d}] {int(idx[ii]):06d} ({int(xs[j]):3d},{int(ys[j]):3d})  "
                  f"{float(ang[ii]):8.4f}  {float(o_l2[ii]):.3e}  "
                  f"{float(rd_n_off[ii]):.6f}  {float(rn_m[ii]):.6f}")

    return out



def _ndc_with_K_generic(H, W, K, near, rays_o, rays_d):
    # Same algebra as your as_ndc branch, factored for CPU tensors
    fx, fy = K[0,0], K[1,1]
    # shift to the near plane in camera space:
    # need rays in cam coords; reconstruct minimal R,t is not available here,
    # so this fallback is only used if off_ndc_rays is missing; prefer off_ndc_rays.
    raise RuntimeError("off_ndc_rays not found; please import from the official repo.")


# ======================================================================================
# Analysis / Summary
# ======================================================================================

def _diagnose_K_scale(aK: np.ndarray, bK: np.ndarray) -> Optional[float]:
    """Return scale s such that bK≈scale*aK (based on fx), else None."""
    fx_a, fx_b = aK[0,0], bK[0,0]
    if fx_a > 0 and fx_b > 0:
        return fx_b / fx_a
    return None

def analyze_results(report: dict, *, out_dir: Path, topk: int = 5) -> dict:
    """
    Analyze dataset_compare.json contents and print key discrepancies.
    Also writes analysis.txt and analysis.json into out_dir.
    """
    lines: list[str] = []
    add = lines.append

    ds = report.get("dataset_kind")
    add(f"=== NeRF Repo Comparison Analysis ({ds}) ===")
    add(f"images: {report.get('num_images')}  | do_render: {report.get('render_stats') is not None}")

    # ---------------- Intrinsics / Pose ----------------
    ip = report.get("intrinsics_pose", {})
    per = ip.get("per_frame", [])
    bad_K = []
    bad_pose = []
    for item in per:
        if not item.get("same_K", True):
            bad_K.append((item["i"], item["K_maxabs"], item["K_l1"]))
        if not item.get("same_pose", True):
            bad_pose.append((item["i"], item["pose_maxabs"], item["pose_l1"]))

    bad_K.sort(key=lambda t: t[1], reverse=True)
    bad_pose.sort(key=lambda t: t[1], reverse=True)

    add("\n-- Intrinsics/Pose --")
    add(f"Frames compared: {ip.get('num_images', len(per))}")
    if bad_K:
        # try to guess a simple scale mismatch
        i0 = bad_K[0][0]
        scale_guess = _diagnose_K_scale(
            np.asarray(report["intrinsics_pose"]["per_frame"][i0]["K_lhs"]) if False else a.Ks[i0],  # if you don't retain Ks in report, use views directly here instead
            np.asarray(b.Ks[i0])  # or pass the Ks you have in scope when calling analyze_results
        )
        if scale_guess is not None:
            if abs(scale_guess - 0.5) < 0.05 or abs(scale_guess - 2.0) < 0.05:
                add(f"Hint: K scale ~{scale_guess:.3f} → likely resolution mismatch (half_res vs downscale).")
                add("      Fix: set my_downscale=2 when --half_res is used (auto-coupled in this patch).")
    else:
        add("K: all equal (within 1e-5).")

    if bad_pose:
        add(f"Different pose: {len(bad_pose)} frame(s). Worst {min(topk,len(bad_pose))}:")
        for i, mx, l1 in bad_pose[:topk]:
            status = _grade(mx, warn=1e-5, fail=1e-3)
            add(f"  [Pose] i={i:03d} maxabs={mx:.3e} l1={l1:.3e} -> {status}")
    else:
        add("Poses: all equal (within 1e-5).")

    # ---------------- Ray metrics ----------------
    rms = report.get("ray_metrics", [])
    add("\n-- Ray Parity (subset) --")
    if not rms:
        add("No ray metrics found.")
    else:
        # rank by direction angle mean and ori_l2_p95
        worst_ang = sorted(
            [(r["i"], r.get("dir_ang_deg_mean", 0.0), r.get("dir_ang_deg_p95", 0.0)) for r in rms],
            key=lambda t: t[1], reverse=True,
        )
        worst_ori = sorted(
            [(r["i"], r.get("ori_l2_p95", 0.0), r.get("ori_l2_mean", 0.0)) for r in rms],
            key=lambda t: t[1], reverse=True,
        )
        add(f"Images evaluated: {len(rms)}")

        add("Direction angle mean (deg), worst:")
        for i, mean_deg, p95 in worst_ang[:topk]:
            st = _grade(mean_deg, warn=0.05, fail=0.5)  # 0.05° warn, 0.5° fail
            add(f"  i={i:03d} mean={mean_deg:.4f}°, p95={p95:.4f}° -> {st}")

        add("Origin L2 (p95), worst:")
        for i, p95, mean in worst_ori[:topk]:
            st = _grade(p95, warn=1e-3, fail=1e-2)  # meters (assuming unit scene)
            add(f"  i={i:03d} p95={p95:.3e}, mean={mean:.3e} -> {st}")

    # ---------------- PDF sampling parity ----------------
    pdf = report.get("pdf_sampling_parity", {})
    add("\n-- PDF Sampling Parity --")
    if pdf:
        max_abs = pdf.get("max_abs", None)
        l1 = pdf.get("l1_mean", None)
        st = _grade(max_abs, warn=1e-5, fail=1e-3) if max_abs is not None else "OK"
        add(f"max_abs={max_abs:.3e}  l1_mean={l1:.3e}  -> {st}")
    else:
        add("No PDF parity stats found.")

    # ---------------- MLP architecture sanity ----------------
    off = report.get("mlp_official", {})
    mine = report.get("mlp_mine", {})
    add("\n-- MLP Architecture --")
    issues = []

    # compare depths/widths
    D_off, W_off = off.get("D"), off.get("W")
    D_my,  W_my  = mine.get("n_layers"), mine.get("hidden_dim")
    if (D_off is not None and D_my is not None) and D_off != D_my:
        issues.append(f"D mismatch: official {D_off} vs mine {D_my}")
    if (W_off is not None and W_my is not None) and W_off != W_my:
        issues.append(f"W mismatch: official {W_off} vs mine {W_my}")

    # enc dims
    enc_pos_off = off.get("input_ch")
    enc_dir_off = off.get("input_ch_views", 0)
    enc_pos_my  = mine.get("enc_pos_dim")
    enc_dir_my  = mine.get("enc_dir_dim")
    if enc_pos_off is not None and enc_pos_my is not None and enc_pos_off != enc_pos_my:
        issues.append(f"pos-encoding dim: official {enc_pos_off} vs mine {enc_pos_my}")
    if enc_dir_off is not None and enc_dir_my is not None and enc_dir_off != enc_dir_my:
        issues.append(f"dir-encoding dim: official {enc_dir_off} vs mine {enc_dir_my}")

    if issues:
        for m in issues:
            add(f"  {m}")
    else:
        add("Architectures appear consistent (D/W and encoding dims).")

    # ---------------- Render stats (optional) ----------------
    rs = report.get("render_stats")
    add("\n-- End-to-end Rendering (sampled rays) --")
    if not rs:
        add("Skipped (run with --do_render to include).")
    else:
        # Compare RGB/acc means between official and mine
        deltas = []
        for item in rs:
            i = item["i"]
            o = item["official"]; m = item["mine"]
            # RGB mean L1
            rgb_o = np.array(o["rgb_mean"], dtype=float)
            rgb_m = np.array(m["rgb_mean"], dtype=float)
            rgb_l1 = float(np.abs(rgb_o - rgb_m).mean())
            # Acc mean abs diff
            acc_diff = abs(float(o["acc_mean"]) - float(m["acc_mean"]))
            deltas.append((i, rgb_l1, acc_diff))

        worst_rgb = sorted(deltas, key=lambda t: t[1], reverse=True)[:topk]
        worst_acc = sorted(deltas, key=lambda t: t[2], reverse=True)[:topk]

        add("RGB mean |Δ| (0..1), worst:")
        for i, rgb_l1, _ in worst_rgb:
            st = _grade(rgb_l1, warn=0.02, fail=0.05)  # 2% warn, 5% fail
            add(f"  i={i:03d} Δrgb_mean={rgb_l1:.4f} -> {st}")

        add("Opacity mean |Δ| (0..1), worst:")
        for i, _, acc_diff in worst_acc:
            st = _grade(acc_diff, warn=0.05, fail=0.10)
            add(f"  i={i:03d} Δacc_mean={acc_diff:.4f} -> {st}")

        add("Note: official reports disparity 'disp', yours reports 'depth'—not directly comparable.")


    # ---------------- Integrator Parity (optional) ----------------
    integ = report.get("integrator_parity")
    add("\n-- Integrator Parity (official raw2outputs vs mine) --")
    if not integ:
        add("Skipped (run with --diag_integrator to include).")
    else:
        # thresholds: very tight equality → OK; small drift → WARN; bigger → FAIL
        # tweak if you prefer stricter/looser gates
        thresholds = {
            "rgb_l1":   (1e-3, 1e-2),
            "acc_l1":   (1e-3, 1e-2),
            "depth_l1": (1e-3, 1e-2),  # depth↔depth (not disparity)
            "w_l1":     (1e-3, 1e-2),
        }

        status_by_metric = {}
        for k in ("rgb_l1", "acc_l1", "depth_l1", "w_l1"):
            v = float(integ.get(k, float("nan")))
            warn, fail = thresholds[k]
            st = _grade(v, warn=warn, fail=fail, higher_is_worse=True) if np.isfinite(v) else "FAIL"
            status_by_metric[k] = st
            add(f"{k}: {v:.6e} -> {st}")

        # High-level diagnosis hints if something failed
        if "FAIL" in status_by_metric.values():
            add("Hints for discrepancies:")
            if status_by_metric.get("w_l1") == "FAIL" or status_by_metric.get("acc_l1") == "FAIL":
                add("  • Check transmittance recursion: cumprod(1-α+1e-10) vs your ε; and σ activation (ReLU vs Softplus).")
                add("  • Ensure Δ = z[i+1]-z[i] is scaled by ||rays_d|| (official multiplies by ray length).")
                add("  • Verify last bin handling matches official (they append a huge Δ ~1e10).")
            if status_by_metric.get("rgb_l1") == "FAIL":
                add("  • If weights differ, RGB will differ. Also confirm white background add: rgb += (1-acc) when white_bkgd=True.")
            if status_by_metric.get("depth_l1") == "FAIL":
                add("  • Compare DEPTH with DEPTH (not disparity). Depth is ∑ w·z. Ensure identical z_vals and weights.")
        elif "WARN" in status_by_metric.values():
            add("Minor drift detected (WARN). If you need bitwise parity, mirror the official ε’s and last-bin logic exactly.")


    # ---------------- Finalize ----------------
    text = "\n".join(lines)
    (out_dir / "analysis.txt").write_text(text)

    # Simple machine-readable summary
    summary = {
        "dataset_kind": ds,
        "num_images": report.get("num_images"),
        "counts": {
            "intrinsics_diff": len(bad_K),
            "pose_diff": len(bad_pose),
            "ray_images": len(rms),
            "render_images": len(rs or []),
        },
        "pdf_parity": pdf,
        "mlp_consistency_issues": issues,
        "notes_path": str(out_dir / "analysis.txt"),
    }

    # Include integrator parity status (if present)
    integ = report.get("integrator_parity")
    if integ:
        thresholds = {
            "rgb_l1":   (1e-3, 1e-2),
            "acc_l1":   (1e-3, 1e-2),
            "depth_l1": (1e-3, 1e-2),
            "w_l1":     (1e-3, 1e-2),
        }
        status_by_metric = {}
        for k, (warn, fail) in thresholds.items():
            v = float(integ.get(k, float("nan")))
            status_by_metric[k] = _grade(v, warn=warn, fail=fail, higher_is_worse=True) if np.isfinite(v) else "FAIL"
        summary["integrator_parity"] = {
            "metrics": integ,
            "status": status_by_metric,
        }

    (out_dir / "analysis.json").write_text(json.dumps(summary, indent=2))

    print("\n" + text + "\n")
    print(f"[OK] Wrote {(out_dir / 'analysis.txt')} and {(out_dir / 'analysis.json')}")
    return summary


# ======================================================================================
# CLI / Orchestration
# ======================================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_kind", choices=["blender", "llff"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="./nerf_compare_out")
    p.add_argument("--official_repo_root", default=None)
    p.add_argument("--my_repo_root", default=None)

    # Loader knobs
    p.add_argument("--half_res", action="store_true", help="Blender only: half resolution")
    p.add_argument("--testskip", type=int, default=1, help="Blender only")
    p.add_argument("--llff_factor", type=int, default=8, help="LLFF downsample factor (official)")
    p.add_argument("--my_downscale", type=int, default=1, help="Your loaders downscale")

    # Compare controls
    p.add_argument("--max_images", type=int, default=0, help="0=all")
    p.add_argument("--max_pixels", type=int, default=8192, help="per-image max pixels for ray parity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--strict_device", action="store_true", help="Assert tensors remain on the chosen device")

    # Rendering (optional)
    p.add_argument("--do_render", action="store_true", help="Run end-to-end RGB on sampled rays")
    p.add_argument("--tie_weights", action="store_true", default=True,
               help="Tie MyNeRF weights from OffNeRF for end-to-end parity.")
    p.add_argument("--n_coarse", type=int, default=64)
    p.add_argument("--n_fine", type=int, default=64)
    p.add_argument("--render_rays_per_image", type=int, default=2048)
    p.add_argument("--off_chunk_rays", type=int, default=32*1024)
    p.add_argument("--off_netchunk", type=int, default=64*1024)
    p.add_argument("--my_chunk_pts", type=int, default=8192)
    p.add_argument("--diag_integrator", action="store_true", default=True, help="Run raw2outputs vs volume_render_rays parity test")

    args = p.parse_args()

    set_all_seeds(args.seed)
    device = torch.device(args.device)
    dev = DeviceEnforcer(device=device, strict=args.strict_device)
    _cuda_preflight(device)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Data loading ----------------
    if args.dataset_kind == "blender":
        off_view = build_view_off_blender(
            args.data_root, half_res=args.half_res, testskip=args.testskip, white_bkgd=True
        )
        # Auto-couple downscale to half_res to avoid K mismatches
        my_down = 2 if args.half_res else max(1, args.my_downscale or 1)
        my_view  = build_view_my_blender(
            args.data_root, downscale=my_down, white_bkgd=True, testskip=args.testskip
        )
        ndc = False
        ndc_near_plane = 1.0
        near, far = 2.0, 6.0
    else:
        off_view = build_view_off_llff(args.data_root, factor=args.llff_factor, recenter=True, spherify=False)
        my_view  = build_view_my_llff(args.data_root, downscale=args.llff_factor, white_bkgd=False)
        ndc = True
        ndc_near_plane = 1.0
        near, far = 0.0, 1.0

    # Limit images if requested
    N = len(off_view.images)
    if args.max_images and args.max_images > 0:
        sel = list(range(min(args.max_images, N)))
        for v in (off_view, my_view):
            v.images = [v.images[i] for i in sel]
            v.Ks     = [v.Ks[i] for i in sel]
            v.c2ws   = [v.c2ws[i] for i in sel]

    # ---------------- Poses / intrinsics comparison ----------------
    pose_intr = compare_intrinsics_pose(off_view, my_view)

    # ---------------- Ray parity per image ----------------
    ray_metrics: List[Dict[str, Any]] = []
    for i in range(len(off_view.images)):
        H, W = off_view.H, off_view.W
        Ki = off_view.Ks[i]
        cA = off_view.c2ws[i]
        c_ref = cA  # use official pose as reference for ray-gen on both
        r = compare_rays_per_image(H, W, Ki, c_ref, dev=dev, n_pixels=args.max_pixels, ndc=ndc, ndc_near_plane=ndc_near_plane, seed=args.seed + i)
        ray_metrics.append({"i": i, **r})

    # ---------------- PDF sampling parity ----------------
    pdf_parity = pdf_sampling_parity_test(dev=dev, batch=8, n_bins=63, n_samples=args.n_fine or 64, seed=args.seed)


    # ---------------- End-to-end rendering stats on sampled rays ----------------
    # --- MLPs for parity / architecture ---
    off_c, off_embed, off_embdirs, off_arch = build_official_mlp(use_viewdirs=True, multires_xyz=10,
                                                                multires_dir=4, D=8, W=256,
                                                                N_importance=args.n_fine, dev=dev)
    my_c, (my_pos_enc, my_dir_enc), my_arch = build_my_mlp(D=8, W=256, dev=dev)

    # Make "fine" models (if needed)
    off_f = OffNeRF(D=8, W=256, input_ch=off_arch["input_ch"], input_ch_views=off_arch["input_ch_views"],
                    output_ch=(5 if (args.n_fine and args.n_fine > 0) else 4),
                    skips=[4], use_viewdirs=True).to(device) if (args.n_fine and args.n_fine > 0) else None
    my_f = MyNeRF(enc_pos_dim=my_pos_enc.out_dim, enc_dir_dim=my_dir_enc.out_dim,
                n_layers=8, hidden_dim=256, skip_pos=4).to(device) if (args.n_fine and args.n_fine > 0) else None

    # Optional: tie weights
    if args.tie_weights:
        tie_official_into_mine(off_c, my_c)
        if off_f is not None and my_f is not None:
            tie_official_into_mine(off_f, my_f)

    # --- Rendering stats (reuse models/encoders) ---
    render_stats = []
    if args.do_render:
        for i in range(len(off_view.images)):
            H, W = off_view.H, off_view.W
            Ki = off_view.Ks[i]
            c2w_i = off_view.c2ws[i]
            # Official
            off_stats, _, _, _embed_fn, _embdirs_fn = render_official_on_rays(
                H, W, Ki, c2w_i, dev=dev, use_ndc=ndc,
                near=near, far=far, n_coarse=args.n_coarse, n_fine=args.n_fine,
                netchunk=args.off_netchunk, chunk_rays=args.off_chunk_rays,
                white_bkgd=off_view.white_bkgd, n_rays=args.render_rays_per_image,
                seed=args.seed + 17 + i,
                model_c=off_c, model_f=off_f, embed_fn=off_embed, embeddirs_fn=off_embdirs,
            )
            # Yours (reuse + tied weights if requested)
            my_stats = render_my_on_rays(
                H, W, Ki, c2w_i, dev=dev, use_ndc=ndc,
                near=near, far=far, n_coarse=args.n_coarse, n_fine=args.n_fine,
                chunk_pts=args.my_chunk_pts, white_bkgd=my_view.white_bkgd,
                n_rays=args.render_rays_per_image, seed=args.seed + 37 + i,
                nerf_c=my_c, nerf_f=my_f, pos_enc=my_pos_enc, dir_enc=my_dir_enc,
            )
            render_stats.append({"i": i, "official": off_stats, "mine": my_stats})

    if args.diag_integrator:
        integ = integrator_parity_test(
            dev=dev, n_rays=512, n_samples=args.n_coarse, white_bkgd=off_view.white_bkgd,
            near=near, far=far, seed=args.seed
        )
        print(f"\n-- Integrator Parity (official vs mine) --")
        for k, v in integ.items():
            print(f"{k}: {v:.6f}")

    # --- NDC parity
    print(f"\n-- NDC Parity (official vs mine) --")
    ndc_metrics = []
    if args.dataset_kind == "llff":
        for i in range(len(off_view.images)):
            H, W = off_view.H, off_view.W
            Ki   = off_view.Ks[i]
            c2w  = off_view.c2ws[i]
            r = check_ndc_ray_parity(
                H, W, Ki, c2w,
                near_world=ndc_near_plane,
                n_rays=8192,
                seed=args.seed,
                dev=dev,
                pixel_center=False,    # official get_rays does NOT add 0.5
                topk_print=5
            )
            ndc_metrics.append({"i": i, **r})

    # ---------------- Write JSON report ----------------
    out = {
        "dataset_kind": args.dataset_kind,
        "num_images": len(off_view.images),
        "intrinsics_pose": pose_intr,
        "ray_metrics": ray_metrics,
        "pdf_sampling_parity": pdf_parity,
        "mlp_official": off_arch,
        "mlp_mine": my_arch,
        "render_stats": (render_stats if args.do_render else None),
        "integrator_parity": (integ if args.diag_integrator else None),
        "ndc_metrics": ndc_metrics,
        "device": str(device),
        "strict_device": bool(args.strict_device),
    }
    (out_dir / f"dataset_compare_{args.dataset_kind}.json").write_text(json.dumps(out, indent=2))
    print(f"[OK] Wrote {out_dir / f'dataset_compare_{args.dataset_kind}.json'}")



    analyze_results(out, out_dir=out_dir, topk=5)

if __name__ == "__main__":
    main()
