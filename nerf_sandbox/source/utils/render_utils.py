"""
Rendering utilities for NeRF:

- sRGB <-> linear conversions
- PNG saving + quick MP4 export
- Camera ray generation (get_camera_rays)
- Stateless NeRF forward (coarse+fine) and full-image rendering
- Convenience: render a single pose
- Camera paths: generate_camera_path_poses (convention-aligned)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from tqdm import tqdm

from nerf_sandbox.source.data.scene import Scene
from nerf_sandbox.source.utils.sampling_utils import sample_pdf
from nerf_sandbox.source.utils.ray_utils import get_camera_rays


# ------------------------- IO -------------------------

@torch.no_grad()
def save_rgb_png(t: torch.Tensor, path: str | Path) -> None:
    """Save HxWx3 float tensor in [0,1] to PNG (uint8)."""
    path = Path(path)
    arr = (t.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    imageio.imwrite(path, arr)

@torch.no_grad()
def save_gray_png(img: torch.Tensor | np.ndarray, path: str | Path) -> None:
    path = Path(path)
    if isinstance(img, torch.Tensor):
        arr = img.clamp(0, 1).cpu().numpy()
    else:
        arr = np.clip(img, 0, 1)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    imageio.imwrite(path, arr)

@torch.no_grad()
def export_validation_video(exp_dir: str | Path,
                            src_glob: Optional[str] = "preview/step_*.png",
                            out_path: Optional[str | Path] = None,
                            fps: int = 24,
                            pad_to_mod: int = 16,
                            cancel_flag_getter: Optional[Callable[[], bool]] = None
                            ) -> Optional[Path]:
    """
    Pack preview PNGs into an MP4. By default uses only the main RGB frames
    saved as 'preview/step_*.png' (avoids acc_*.png / depth_*.png).

    Args:
        exp_dir: experiment directory that contains the 'preview/' folder.
        src_glob: pattern relative to exp_dir. Default: 'preview/step_*.png'.
        out_path: explicit output path; default: '<exp_dir>/val_preview.mp4'.
        fps: frames per second for the output video.
        pad_to_mod: pad H and W to be multiples of this value (better for codecs).
        cancel_flag_getter: optional callback to allow early cancel; if it
            returns True the function exits and returns None.
    """
    import re
    exp_dir = Path(exp_dir)
    pattern = src_glob or "preview/step_*.png"

    # Collect only the intended RGB frames (e.g., step_0000100.png)
    frames = list(exp_dir.glob(pattern))
    if not frames:
        return None

    # Natural numeric sort by step index; fallback to name if no digits
    def _step_key(p: Path) -> tuple[int, str]:
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else -1, p.name)

    frames.sort(key=_step_key)

    out = Path(out_path) if out_path else (exp_dir / "val_preview.mp4")

    imgs = []
    for p in frames:
        if cancel_flag_getter and cancel_flag_getter():
            return None
        img = imageio.imread(p)
        if pad_to_mod:
            H, W = img.shape[:2]
            pad_h = (pad_to_mod - (H % pad_to_mod)) % pad_to_mod
            pad_w = (pad_to_mod - (W % pad_to_mod)) % pad_to_mod
            if pad_h or pad_w:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        imgs.append(img)

    imageio.mimwrite(out, imgs, fps=fps, codec="libx264", quality=8)
    return out


# ------------------------- Volume rendering -------------------------


def volume_render_rays(
    rgb: torch.Tensor,                   # (B, N, 3)
    sigma: torch.Tensor,                 # (B, N)  -- after activation (relu/softplus)
    z_depths: torch.Tensor,              # (B, N)  -- sorted along each ray
    ray_norm: torch.Tensor | None = None,# (B,1) or (B,)  -- ||d_raw||, to scale Δz into metric Δs
    white_bkgd: bool = False,
    eps: float = 1e-10,
    infinite_last_bin: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Volume rendering with alpha compositing.

    Returns:
        composite_rgb       : (B, 3)   (same color space as input rgb; no gamma applied)
        weights             : (B, N)
        accumulated_opacity : (B, 1)
        depth               : (B, 1)   in same units as z_depths ([0,1] for NDC; [near,far] otherwise)
    """
    B, N = z_depths.shape
    # Optional: enforce sorted samples
    # assert torch.all(z_depths[..., 1:] >= z_depths[..., :-1]), "z_depths must be sorted"

    # 1) Bin lengths
    deltas_finite = z_depths[..., 1:] - z_depths[..., :-1]                # (B, N-1)
    if infinite_last_bin:
        delta_last = torch.full_like(deltas_finite[..., :1], 1e10)
    else:
        delta_last = torch.zeros_like(deltas_finite[..., :1])
    deltas = torch.cat([deltas_finite, delta_last], dim=-1)               # (B, N)

    # Scale Δz by ||d_raw|| to get metric Δs (both NDC and non-NDC)
    if ray_norm is not None:
        ray_norm = ray_norm.view(B, 1).to(deltas.dtype)                   # (B,1)
        deltas = deltas * ray_norm                                        # (B, N)

    # 2) Per-sample opacity α = 1 - exp(-σΔ)   (keep the sdt clamp for numerical safety only)
    sdt = (sigma * deltas).clamp_min(0.0).clamp_max(60.0)
    alphas = 1.0 - torch.exp(-sdt)                         # no explicit clamp

    # 3) Transmittance T_j = Π_{k<j} (1 - α_k + 1e-10) with EXCLUSIVE cumprod
    shifted = torch.cat([torch.ones_like(alphas[..., :1]),
                        1.0 - alphas + eps], dim=-1)      # add eps; do NOT clamp
    transmittance = torch.cumprod(shifted, dim=-1)[..., :-1]

    # 4) Weights, opacity, depth
    weights = (transmittance * alphas)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)  # (B, N)

    accumulated_opacity = weights.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)   # (B,1)
    depth = (weights * z_depths).sum(dim=-1, keepdim=True) / (accumulated_opacity + eps)  # (B,1)

    # 5) Composite color
    composite_rgb = (weights[..., None] * rgb).sum(dim=-2)                # (B,3)
    if white_bkgd:
        composite_rgb = composite_rgb + (1.0 - accumulated_opacity)

    # Optional: ensure finite outputs
    composite_rgb = torch.nan_to_num(composite_rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    return composite_rgb, weights, accumulated_opacity, depth

# ------------------------- Stateless NeRF forward -------------------------

def nerf_forward_pass(
    rays_o: torch.Tensor,                    # (B, 3)
    rays_d_unit: torch.Tensor,               # (B, 3) -- unit directions in marching space (world or NDC)
    z_vals: torch.Tensor,                    # (B, N) -- parameter samples along the ray (sorted)
    *,
    pos_enc,                                  # callable: (Q,3)->(Q,Dx)
    dir_enc,                                  # callable: (Q,3)->(Q,Dd)
    nerf,                                     # MLP: (enc_pos, enc_dir) -> {...} or [...,4]
    white_bkgd: bool,
    ray_norms: torch.Tensor | None = None,    # (B,1) or (B,) -- ||d_raw|| BEFORE normalization, or None to disable scaling
    viewdirs_world_unit: torch.Tensor | None = None,  # (B,3) -- unit world/cam dirs for MLP
    sigma_activation: str = "relu",
    raw_noise_std: float = 0.0,               # >0 only during training
    training: bool = False,
    mlp_chunk: int = 0,                       # cap encoder+MLP queries (0 = no cap)
    infinite_last_bin: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stateless NeRF march + composite at fixed z-samples.

    Returns:
        composite_rgb : (B, 3)  -- in [0,1]
        weights       : (B, N)
        acc           : (B, 1)
        depth         : (B, 1)  -- weighted average of z_vals (same units as z_vals)
    """
    # ---- basic shape/finite checks (fail fast) ----
    assert rays_o.shape == rays_d_unit.shape and rays_o.shape[-1] == 3, \
        f"bad ray shapes {rays_o.shape} / {rays_d_unit.shape}"
    B, N = z_vals.shape
    if ray_norms is not None:
        # accept (B,) or (B,1); reshape to (B,1)
        assert ray_norms.shape[:1] == (B,), f"ray_norms {ray_norms.shape} must broadcast with batch {B}"
        ray_norms = ray_norms.view(B, 1)
    assert torch.all(torch.isfinite(rays_o)) \
        and torch.all(torch.isfinite(rays_d_unit)) \
        and torch.all(torch.isfinite(z_vals)), "non-finite inputs"

    # ---- points along rays (metric distance in the marching space) ----
    # If ray_norms is None: use z as-is (no scaling). Otherwise s = z * ||d_raw||.
    if ray_norms is None:
        z_metric = z_vals
    else:
        z_metric = z_vals * ray_norms                                         # (B, N)
    pts = rays_o[:, None, :] + rays_d_unit[:, None, :] * z_metric[..., None] # (B, N, 3)

    # ---- view directions for shading (always WORLD/CAMERA unit dirs) ----
    if viewdirs_world_unit is not None:
        vdirs_unit = F.normalize(viewdirs_world_unit, dim=-1)                 # (B, 3)
    else:
        # Fallback: marching dirs (already unit) — OK in world, suboptimal in NDC but safe
        vdirs_unit = F.normalize(rays_d_unit, dim=-1)
    vdirs = vdirs_unit[:, None, :].expand_as(pts)                             # (B, N, 3)

    # ---- flatten -> encode -> MLP ----
    flat_pts  = pts.reshape(-1, 3)
    flat_dirs = vdirs.reshape(-1, 3)
    Q = flat_pts.shape[0]

    def _mlp_forward(epos: torch.Tensor, edir: torch.Tensor):
        out = nerf(epos, edir)
        if isinstance(out, dict):
            rgb_flat   = torch.sigmoid(out["rgb"])           # (Q,3)
            sigma_flat = out["sigma"].reshape(-1)            # (Q,)
        else:
            rgb_flat   = torch.sigmoid(out[..., :3])
            sigma_flat = out[..., 3].reshape(-1)

        if training and (raw_noise_std > 0.0):
            noise_fp32 = torch.randn(sigma_flat.shape, device=sigma_flat.device, dtype=torch.float32) * float(raw_noise_std)
            sigma_flat = sigma_flat + noise_fp32.to(sigma_flat.dtype)

        if sigma_activation == "softplus":
            sigma_flat = F.softplus(sigma_flat)
        else:
            sigma_flat = F.relu(sigma_flat)
        return rgb_flat, sigma_flat

    if mlp_chunk and mlp_chunk > 0 and Q > mlp_chunk:
        rgb_chunks, sig_chunks = [], []
        for s in range(0, Q, mlp_chunk):
            epos = pos_enc(flat_pts[s:s+mlp_chunk])
            edir = dir_enc(flat_dirs[s:s+mlp_chunk])
            r, sgm = _mlp_forward(epos, edir)
            rgb_chunks.append(r); sig_chunks.append(sgm)
        rgb_flat   = torch.cat(rgb_chunks, dim=0)
        sigma_flat = torch.cat(sig_chunks, dim=0)
    else:
        epos = pos_enc(flat_pts)
        edir = dir_enc(flat_dirs)
        rgb_flat, sigma_flat = _mlp_forward(epos, edir)

    # Back to (B,N,·)
    rgb   = rgb_flat.reshape(B, N, 3)
    sigma = sigma_flat.reshape(B, N)

    # ---- volume rendering ----
    # Pass ray_norm=None to disable Δ scaling in the compositor if it was None here.
    composite_rgb, weights, acc_map, depth_map = volume_render_rays(
        rgb=rgb,
        sigma=sigma,
        z_depths=z_vals,                   # keep unscaled so 'depth' stays in z units
        white_bkgd=white_bkgd,
        ray_norm=(ray_norms if ray_norms is not None else None),
        infinite_last_bin=infinite_last_bin,
    )

    # Stabilize shapes
    if weights.dim() == 1:   weights   = weights.view(B, N)
    if acc_map.dim() == 1:   acc_map   = acc_map.view(B, 1)
    if depth_map.dim() == 1: depth_map = depth_map.view(B, 1)

    return composite_rgb, weights, acc_map, depth_map

@torch.no_grad()
def render_image_chunked(
    rays_o: torch.Tensor,                 # (H*W, 3)
    rays_d_unit: torch.Tensor,            # (H*W, 3)  unit marching dirs (world or NDC)
    ray_norms: torch.Tensor,              # (H*W, 1)  ||d_raw|| BEFORE normalization
    H: int, W: int,
    near: float, far: float,
    pos_enc, dir_enc,
    nerf_c, nerf_f,
    nc_eval: int, nf_eval: int,
    white_bkgd: bool, device: torch.device,
    eval_chunk: int = 8192,
    perturb: bool = False,
    sigma_activation: str = "relu",
    *,
    viewdirs_world_unit: torch.Tensor | None = None,  # (H*W, 3) unit WORLD/CAM dirs for the MLP
    infinite_last_bin: bool = False
) -> dict:
    """
    Renders an image by tiling rays into chunks.
    Coarse (uniform) + hierarchical fine (midpoint-only PDF sampler).

    Important: `rays_d_unit` must be unit length; `ray_norms` contains the
    pre-normalization norms for metric step lengths during alpha compositing.
    """

    # Flatten (already flat in your use, but keep this robust)
    total_num_rays = H * W
    rays_origins_flat    = rays_o.reshape(total_num_rays, 3).to(device)
    rays_directions_flat = rays_d_unit.reshape(total_num_rays, 3).to(device)
    ray_norms_flat       = ray_norms.reshape(total_num_rays, 1).to(device)
    render_dtype = rays_origins_flat.dtype

    # Optional world viewdirs for the MLP
    if viewdirs_world_unit is not None:
        vdirs_world_flat = viewdirs_world_unit.reshape(total_num_rays, 3).to(device)
    else:
        vdirs_world_flat = None  # falls back to rays_d_unit in nerf_forward_pass (OK for non-NDC; suboptimal for NDC)

    # Output buffers (flat)
    output_rgb_flat   = torch.empty((total_num_rays, 3), dtype=render_dtype, device=device)
    output_acc_flat   = torch.empty((total_num_rays, 1), dtype=render_dtype, device=device)
    output_depth_flat = torch.empty((total_num_rays, 1), dtype=render_dtype, device=device)

    # Coarse sampling (near→far)
    t_lin = torch.linspace(0.0, 1.0, steps=nc_eval, device=device, dtype=render_dtype)
    z_coarse_template = near * (1.0 - t_lin) + far * t_lin  # (nc_eval,)

    # Autocast
    device_type_str = device.type if isinstance(device, torch.device) else str(device)
    use_autocast = device_type_str.startswith("cuda")

    for start_idx in range(0, total_num_rays, eval_chunk):
        end_idx = min(total_num_rays, start_idx + eval_chunk)

        # Slice this chunk
        rays_origins_chunk    = rays_origins_flat[start_idx:end_idx]         # (B,3)
        rays_directions_chunk = rays_directions_flat[start_idx:end_idx]      # (B,3) unit
        ray_norms_chunk       = ray_norms_flat[start_idx:end_idx]            # (B,1)
        batch_size_chunk = rays_origins_chunk.shape[0]

        vdirs_chunk = None
        if vdirs_world_flat is not None:
            vdirs_chunk = vdirs_world_flat[start_idx:end_idx]                # (B,3) unit WORLD/CAM dirs

        # ----- Coarse: uniform z -----
        z_coarse = z_coarse_template.expand(batch_size_chunk, nc_eval).contiguous()

        if perturb:
            midpoints   = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])
            lower_edges = torch.cat([z_coarse[:, :1], midpoints], dim=-1)
            upper_edges = torch.cat([midpoints, z_coarse[:, -1:]], dim=-1)
            z_coarse = lower_edges + (upper_edges - lower_edges) * torch.rand_like(z_coarse)
            z_coarse = torch.sort(z_coarse, dim=-1).values

        # ----- Coarse render pass -----
        with torch.amp.autocast(device_type_str, enabled=use_autocast):
            composite_rgb_coarse, weights_coarse, acc_coarse, depth_coarse = nerf_forward_pass(
                rays_o=rays_origins_chunk,
                rays_d_unit=rays_directions_chunk,                 # unit
                z_vals=z_coarse,
                pos_enc=pos_enc, dir_enc=dir_enc, nerf=nerf_c,
                white_bkgd=white_bkgd,
                ray_norms=ray_norms_chunk,                    # (B,1) metric scale
                viewdirs_world_unit=vdirs_chunk,              # unit WORLD/CAM dirs for MLP
                sigma_activation=sigma_activation,
                raw_noise_std=0.0,                            # eval
                training=False,
                mlp_chunk=eval_chunk,
                infinite_last_bin=infinite_last_bin
            )

        # Make weights shape predictable
        weights_coarse = weights_coarse.reshape(batch_size_chunk, -1)

        # If no fine pass, write and continue
        if (nf_eval is None) or (int(nf_eval) <= 0) or (nerf_f is None):
            output_rgb_flat[start_idx:end_idx]   = composite_rgb_coarse
            output_acc_flat[start_idx:end_idx]   = acc_coarse.reshape(-1, 1)
            output_depth_flat[start_idx:end_idx] = depth_coarse.reshape(-1, 1)
            continue

        # ----- Hierarchical fine sampling (midpoints + interval weights) -----
        bins_mid     = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])                 # (B, Nc-1)
        weights_bins = 0.5 * (weights_coarse[:, 1:] + weights_coarse[:, :-1])     # (B, Nc-1)
        weights_bins = (weights_bins.detach() + 1e-5)

        assert bins_mid.shape == weights_bins.shape, "bins_mid and weights_bins must align"

        z_fine = sample_pdf(bins_mid, weights_bins, n_samples=nf_eval, deterministic=True)
        z_merged = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1).values  # (B, Nc+Nf)

        # ----- Fine render pass -----
        with torch.amp.autocast(device_type_str, enabled=use_autocast):
            composite_rgb_fine, _, acc_fine, depth_fine = nerf_forward_pass(
                rays_o=rays_origins_chunk,
                rays_d_unit=rays_directions_chunk,                 # unit
                z_vals=z_merged,
                pos_enc=pos_enc, dir_enc=dir_enc, nerf=nerf_f,
                white_bkgd=white_bkgd,
                ray_norms=ray_norms_chunk,                    # (B,1)
                viewdirs_world_unit=vdirs_chunk,
                sigma_activation=sigma_activation,
                raw_noise_std=0.0,
                training=False,
                mlp_chunk=eval_chunk,
                infinite_last_bin=infinite_last_bin
            )

        # Write chunk results
        output_rgb_flat[start_idx:end_idx]   = composite_rgb_fine
        output_acc_flat[start_idx:end_idx]   = acc_fine.reshape(-1, 1)
        output_depth_flat[start_idx:end_idx] = depth_fine.reshape(-1, 1)

    # Reshape to images
    return {
        "rgb":   output_rgb_flat.reshape(H, W, 3),
        "acc":   output_acc_flat.reshape(H, W, 1),
        "depth": output_depth_flat.reshape(H, W, 1),
    }

@torch.no_grad()
def render_pose(
    c2w, H, W, K,
    near: float, far: float,             # world-space sampling range when not using NDC
    pos_enc, dir_enc, nerf_c, nerf_f,
    device,
    white_bkgd: bool = True,
    nc_eval: int = 64, nf_eval: int = 128, eval_chunk: int = 8192,
    perturb: bool = False,
    sigma_activation: str = "relu",
    *,
    use_ndc: bool = False,
    convention: str = "opengl",
    near_plane: float | None = None,     # world-space plane used for the NDC warp
    samp_near: float | None = None,      # sampling start override
    samp_far: float | None = None,       # sampling end override
    infinite_last_bin: bool = False,
):
    """
    Render a single pose with explicit WORLD vs MARCHING ray spaces.

    WORLD rays (always computed, as_ndc=False):
      - used only for the MLP's directional encoding: unit WORLD/CAMERA viewdirs

    MARCHING rays (branch):
      - if use_ndc=True: NDC-warped rays; z-sampling in [0,1]
      - else: world rays; z-sampling in [near,far]
    """

    # -------- WORLD/CAMERA rays (for the MLP's directional encoding) --------
    (
        rays_o_world,            # (H*W, 3)
        rays_d_world_unit,       # (H*W, 3)
        rays_d_world_norm,       # (H*W, 1)
        _rays_o_march_IGN,       # unused in this branch
        _rays_d_march_unit_IGN,  # unused in this branch
        _rays_d_march_norm_IGN,  # unused in this branch
    ) = get_camera_rays(
        image_h=H, image_w=W,
        intrinsic_matrix=K, transform_camera_to_world=c2w,
        device=device, dtype=torch.float32,
        convention=convention,
        pixel_center=True,                 # keep consistent with how K is defined
        as_ndc=False,                       # WORLD only in this call
        near_plane=near,                    # unused here but kept for signature symmetry
        pixels_xy=None,
    )

    # -------- MARCHING rays (NDC if requested, else WORLD) --------
    if use_ndc:
        ndc_near = float(near if (near_plane is None) else near_plane)
        (
            _r_o_w, _r_d_w_u, _r_d_w_n,                    # ignored (WORLD copies)
            rays_o_marching,                               # (H*W, 3)  NDC origins
            rays_d_marching_unit,                          # (H*W, 3)  NDC unit dirs
            rays_d_marching_norm,                          # (H*W, 1)  ||d_raw|| in NDC
        ) = get_camera_rays(
            image_h=H, image_w=W,
            intrinsic_matrix=K, transform_camera_to_world=c2w,
            device=device, dtype=torch.float32,
            convention=convention,
            pixel_center=True,
            as_ndc=True,                                   # <<< NDC marching
            near_plane=ndc_near,
            pixels_xy=None,
        )
        samp_near_render = 0.0 if samp_near is None else float(samp_near)
        samp_far_render  = 1.0 if samp_far  is None else float(samp_far)
    else:
        # march in WORLD: reuse WORLD rays
        rays_o_marching      = rays_o_world
        rays_d_marching_unit = rays_d_world_unit
        rays_d_marching_norm = rays_d_world_norm
        samp_near_render = near if samp_near is None else float(samp_near)
        samp_far_render  = far  if samp_far  is None else float(samp_far)

    # -------- Render full image in chunks (coarse + fine) --------
    out = render_image_chunked(
        # marching rays (unit dirs + norms)
        rays_o=rays_o_marching,
        rays_d_unit=rays_d_marching_unit,
        ray_norms=rays_d_marching_norm,

        H=H, W=W,
        near=samp_near_render, far=samp_far_render,

        pos_enc=pos_enc, dir_enc=dir_enc,
        nerf_c=nerf_c, nerf_f=nerf_f,

        nc_eval=nc_eval, nf_eval=nf_eval,
        white_bkgd=white_bkgd, device=device,
        eval_chunk=eval_chunk, perturb=perturb,
        sigma_activation=sigma_activation,

        # WORLD unit viewdirs fed to the MLP's directional branch
        viewdirs_world_unit=rays_d_world_unit,

        infinite_last_bin=infinite_last_bin,
    )

    return out
