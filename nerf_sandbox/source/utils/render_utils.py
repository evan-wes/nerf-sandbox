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
from typing import Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
from tqdm import tqdm

from nerf_sandbox.source.data.scene import Scene
from nerf_sandbox.source.utils.sampling_utils import sample_pdf
from nerf_sandbox.source.utils.ray_utils import get_camera_rays


# ------------------------- Colorspace -------------------------

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    a = 0.055
    return torch.where(
        x <= 0.04045, x / 12.92,
        ((x + a) / (1 + a)).pow(2.4)
    )

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    a = 0.055
    return torch.where(
        x <= 0.0031308, x * 12.92,
        (1 + a) * x.pow(1 / 2.4) - a
    )


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
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_depths: torch.Tensor,
    ray_norm: torch.Tensor | None = None,
    white_bkgd: bool = False,
    eps: float = 1e-10,
    infinite_last_bin: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Uses the MLP output of RGB and volume density at sampled depths along a batch of rays to
    render the composited RGB values, along with other semantically meaningful outputs used
    downstream.

    Parameters
    ----------
    rgb : torch.Tensor
        (B, N, 3) color at sampled points along rays.
    sigma : torch.Tensor
        (B, N) volume density at sampled points (after activation).
    z_depths : torch.Tensor
        (B, N) sorted depths along each ray.
    ray_norm : torch.Tensor | None
        Optional per-ray direction norms of shape [B,1] or [B] used to scale bin lengths
        when rays were sampled with non-unit directions. If None, deltas are used as-is.
    white_bkgd : bool
        Whether to composite against white background in linear space.
    eps : float
        Small value to stabilize products/divisions.
    infinite_last_bin : bool
        If True, sets the last bin length to a very large dtype-safe value so that
        any remaining transmittance is captured by the last interval (AMP-safe).
        If False, remaining mass goes to the explicit background term.

    Returns
    -------
    composite_rgb : torch.Tensor
        The rendered color for each ray after integrating the MLP-predicted color and volume
        densities along the rays at the sampled depths. Has shape (num_rays, 3).
    weights : torch.Tensor
        Probability of a ray terminating at a volume element. Used for re-sampling new positions
        along the rays. Has shape (num_rays, num_ray_pts).
    accumulated_opacity : torch.Tensor
        The sum of the weights along the rays. Has shape (num_rays,).
    depth : torch.Tensor
        The depth of the rendered volume along each ray. Has shape (num_rays,).
    """

    # 1) Bin lengths (finite bins only by default; remaining transmittance goes to background)
    deltas_finite = z_depths[..., 1:] - z_depths[..., :-1]               # [B, N-1]
    if infinite_last_bin:
        # AMP-safe "infinite" last interval so alpha_last ≈ 1 for nonzero sigma
        delta_last = torch.full_like(deltas_finite[..., :1],
                                     torch.finfo(sigma.dtype).max)
    else:
        delta_last = torch.zeros_like(deltas_finite[..., :1])
    deltas = torch.cat([deltas_finite, delta_last], dim=-1)              # [B, N]

    # If rays were sampled with NON-unit directions, convert param Δz to metric distances.
    if ray_norm is not None:
        deltas = deltas * (ray_norm if ray_norm.ndim == deltas.ndim else ray_norm[..., None])

    # 2) Per-sample opacity α = 1 - exp(-σ Δ)
    #    (No activation here; 'sigma' is already after the chosen activation.)
    alphas = 1.0 - torch.exp(-sigma * deltas)                            # [B, N]
    alphas = alphas.clamp(0.0, 1.0)

    # 3) Transmittance T = Π_k<j (1 - α_k)
    one_minus = (1.0 - alphas).clamp_min(eps)                            # avoid zeros
    # Prepend 1 so that transmittance[j] is product up to (but not including) j
    shifted = torch.cat([torch.ones_like(one_minus[..., :1]), one_minus], dim=-1)
    transmittance = torch.cumprod(shifted, dim=-1)[..., :-1]             # [B, N]

    # 4) Weights and accumulators
    weights = (transmittance * alphas).clamp_min(0.0)                    # [B, N]
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    accumulated_opacity = weights.sum(dim=-1).clamp(0.0, 1.0)            # [B]
    depth = (weights * z_depths).sum(dim=-1) / (accumulated_opacity + eps)  # [B]

    # 5) Composite color
    composite_rgb = (weights[..., None] * rgb).sum(dim=-2)               # [B, 3]
    if white_bkgd:
        # White background in linear space
        composite_rgb = composite_rgb + (1.0 - accumulated_opacity)[..., None]

    return composite_rgb, weights, accumulated_opacity, depth


# ------------------------- Stateless NeRF forward -------------------------


@torch.no_grad()
def _forward_nerf_once(rays_o: torch.Tensor, rays_d: torch.Tensor, z: torch.Tensor,
                       pos_enc, dir_enc, model,
                       white_bkgd: bool,
                       sigma_activation: str = "relu",
                       raw_noise_std: float = 0.0,
                       training: bool = False,
                       use_ray_norm: bool = True):
    """
    Stateless coarse/fine pass over given z samples.
    Returns composite RGB, weights, acc, depth (all batch-shaped).
    """
    B, N = z.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]
    viewdirs = F.normalize(rays_d, dim=-1)
    vdirs = viewdirs[:, None, :].expand(B, N, 3)

    enc_pos = pos_enc(pts.reshape(-1, 3))
    enc_dir = dir_enc(vdirs.reshape(-1, 3))

    pred = model(enc_pos, enc_dir)
    rgb_raw, sigma_raw = pred[..., :3], pred[..., 3]
    rgb = torch.sigmoid(rgb_raw).reshape(B, N, 3)

    if raw_noise_std and training:
        sigma_raw = sigma_raw + torch.randn_like(sigma_raw) * float(raw_noise_std)
    if str(sigma_activation).lower() == "relu":
        sigma = torch.relu(sigma_raw).reshape(B, N)
    else:
        sigma = F.softplus(sigma_raw, beta=1.0).reshape(B, N)

    rn = rays_d.norm(dim=-1, keepdim=True) if use_ray_norm else None
    return volume_render_rays(rgb, sigma, z, rn, white_bkgd=white_bkgd)


@torch.no_grad()
def render_image_chunked(rays_o: torch.Tensor, rays_d: torch.Tensor,
                         H: int, W: int,
                         near: float, far: float,
                         pos_enc, dir_enc,
                         nerf_c, nerf_f,
                         nc_eval: int, nf_eval: int,
                         white_bkgd: bool, device: torch.device,
                         eval_chunk: int = 8192,
                         perturb: bool = False,
                         raw_noise_std: float = 0.0,
                         sigma_activation: str = "relu") -> dict:
    """
    Render a full image by tiling rays into chunks to control memory usage.
    Performs a coarse pass (uniform samples) followed by hierarchical resampling
    for the fine pass, then composites to RGB/acc/depth.

    Parameters
    ----------
    rays_o, rays_d : (H*W, 3) or (H, W, 3)
        Ray origins and directions.
    H, W : int
        Image height and width.
    near, far : float
        Near/far bounds for sampling along the ray.
    pos_enc, dir_enc :
        Positional and directional encoders (modules with __call__/forward).
    nerf_c, nerf_f :
        Coarse and fine NeRF MLPs.
    nc_eval, nf_eval : int
        Number of coarse and fine samples per ray.
    white_bkgd : bool
        Whether to composite against white background in linear space.
    device : torch.device
        Device to run the rendering on.
    eval_chunk : int
        Number of rays to process per chunk.
    perturb : bool
        If True, apply stratified jittering to coarse samples.
    raw_noise_std : float
        Std-dev of Gaussian noise added to raw sigma (density) logits.
        If > 0, the noise path is enabled (even under no_grad) to stress-test eval.
    sigma_activation : {"relu"|"softplus"}
        Sigma activation used inside the forward pass.

    Returns
    -------
    dict with:
      - "rgb"   : (H, W, 3) composite color in linear [0,1]
      - "acc"   : (H, W, 1) accumulated opacity
      - "depth" : (H, W, 1) expected depth
    """

    # Flatten rays to a (num_rays, 3) layout and move to the requested device
    total_num_rays = H * W
    rays_origins_flat = rays_o.reshape(total_num_rays, 3).to(device)
    rays_directions_flat = rays_d.reshape(total_num_rays, 3).to(device)
    render_dtype = rays_origins_flat.dtype

    # Output buffers (flat); we’ll reshape to (H, W, ·) at the end
    output_rgb_flat   = torch.empty((total_num_rays, 3), dtype=render_dtype, device=device)
    output_acc_flat   = torch.empty((total_num_rays, 1), dtype=render_dtype, device=device)
    output_depth_flat = torch.empty((total_num_rays, 1), dtype=render_dtype, device=device)

    # Sampling along rays (coarse): linearly interpolate in z from near→far
    # We’ll expand per chunk, so we keep the base 1D template here.
    t_lin = torch.linspace(0.0, 1.0, steps=nc_eval, device=device, dtype=render_dtype)
    z_coarse_template = near * (1.0 - t_lin) + far * t_lin  # (nc_eval,)

    # # AUTO: if we’re sampling in NDC ([0,1]), don’t scale bins by ‖dir‖
    # ndc_sampling = (abs(float(near)) <= 1e-8) and (abs(float(far) - 1.0) <= 1e-8)
    # use_ray_norm = not ndc_sampling

    use_ray_norm = True

    # Whether to add training-style noise to sigma during eval
    use_density_noise = float(raw_noise_std) > 0.0

    # Set up autocast for CUDA to reduce activation memory during eval
    device_type_str = device.type if isinstance(device, torch.device) else str(device)
    use_autocast = device_type_str.startswith("cuda")

    for start_idx in range(0, total_num_rays, eval_chunk):
        end_idx = min(total_num_rays, start_idx + eval_chunk)

        # Slice the current chunk of rays
        rays_origins_chunk = rays_origins_flat[start_idx:end_idx]     # (B, 3)
        rays_directions_chunk = rays_directions_flat[start_idx:end_idx]  # (B, 3)
        batch_size_chunk = rays_origins_chunk.shape[0]

        # ----- Coarse sampling (uniform in depth) -----
        # Expand z template to all rays in this chunk: (B, nc_eval)
        z_coarse = z_coarse_template.expand(batch_size_chunk, nc_eval).contiguous()

        # Optional stratified jittering for coarse samples
        if perturb:
            midpoints = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])  # (B, nc_eval-1)
            lower_edges = torch.cat([z_coarse[:, :1], midpoints], dim=-1)
            upper_edges = torch.cat([midpoints, z_coarse[:, -1:]], dim=-1)
            z_coarse = lower_edges + (upper_edges - lower_edges) * torch.rand_like(z_coarse)
            z_coarse = torch.sort(z_coarse, dim=-1).values  # keep ascending

        # Coarse render pass
        with torch.amp.autocast(device_type_str, enabled=use_autocast):
            composite_rgb_coarse, weights_coarse, acc_coarse, _ = _forward_nerf_once(
                rays_origins_chunk,
                rays_directions_chunk,
                z_coarse,
                pos_enc,
                dir_enc,
                nerf_c,
                white_bkgd,
                sigma_activation=sigma_activation,
                raw_noise_std=raw_noise_std,
                training=use_density_noise,  # enable noise path if requested
                use_ray_norm=use_ray_norm
            )

        # ----- Hierarchical resampling for fine pass -----
        # Build mid-point bins and use weights (excluding the first/last) as the histogram
        # for inverse CDF sampling. Detach to prevent grads flowing through the sampler.
        z_midpoints = 0.5 * (z_coarse[:, 1:] + z_coarse[:, :-1])       # (B, nc_eval-1)
        weights_mid = weights_coarse[:, 1:-1].detach()                 # (B, nc_eval-2)

        # Sample nf_eval points from the piecewise-constant PDF; deterministic=True for eval
        z_fine = sample_pdf(z_midpoints, weights_mid, n_samples=nf_eval, deterministic=True)  # (B, nf_eval)

        # Merge and sort all z-samples for the fine pass
        z_merged = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1).values  # (B, nc_eval + nf_eval)

        # Fine render pass
        with torch.amp.autocast(device_type_str, enabled=use_autocast):
            composite_rgb_fine, _, acc_fine, depth_fine = _forward_nerf_once(
                rays_origins_chunk,
                rays_directions_chunk,
                z_merged,
                pos_enc,
                dir_enc,
                nerf_f,
                white_bkgd,
                sigma_activation=sigma_activation,
                raw_noise_std=raw_noise_std,
                training=use_density_noise,  # mirror noise choice for fine pass
                use_ray_norm=use_ray_norm
            )

        # Ensure fine outputs have shape (batch, 1) for assignment into preallocated (B,1) buffers.
        if acc_fine.dim() == 1:
            acc_fine = acc_fine.unsqueeze(-1)
        if depth_fine.dim() == 1:
            depth_fine = depth_fine.unsqueeze(-1)

        # Write chunk results back to flat output buffers
        output_rgb_flat[start_idx:end_idx]   = composite_rgb_fine
        output_acc_flat[start_idx:end_idx]   = acc_fine
        output_depth_flat[start_idx:end_idx] = depth_fine

    # Reshape flat outputs to images
    return {
        "rgb":   output_rgb_flat.reshape(H, W, 3),
        "acc":   output_acc_flat.reshape(H, W, 1),
        "depth": output_depth_flat.reshape(H, W, 1),
    }


@torch.no_grad()
def render_pose(
    c2w, H, W, K,
    near: float, far: float,             # keep for backward-compat (world-space)
    pos_enc, dir_enc, nerf_c, nerf_f,
    device,
    white_bkgd=True, nc_eval=64, nf_eval=128, eval_chunk=8192,
    perturb=False, raw_noise_std=0.0, sigma_activation="relu",
    *,
    use_ndc: bool = False,
    convention: str = "opengl",
    near_plane: float | None = None,     # NEW: world-space plane for NDC warp
    samp_near: float | None = None,      # NEW: sampling start override
    samp_far: float | None = None        # NEW: sampling end override
):
    # 1) Choose near-plane for ray generation
    ndc_near = near if near_plane is None else near_plane

    rays_o, rays_d = get_camera_rays(
        image_h=H, image_w=W, intrinsic_matrix=K, transform_camera_to_world=c2w,
        device=device, dtype=torch.float32, normalize_dirs=True,
        as_ndc=use_ndc, near_plane=ndc_near, convention=convention
    )

    # 2) Choose sampling bounds for the integrator
    if use_ndc:
        s_near = 0.0 if samp_near is None else samp_near
        s_far  = 1.0 if samp_far  is None else samp_far
    else:
        s_near = near if samp_near is None else samp_near
        s_far  = far  if samp_far  is None else samp_far

    out = render_image_chunked(
        rays_o, rays_d, H, W,
        near=s_near, far=s_far,
        pos_enc=pos_enc, dir_enc=dir_enc,
        nerf_c=nerf_c, nerf_f=nerf_f,
        nc_eval=nc_eval, nf_eval=nf_eval,
        white_bkgd=white_bkgd, device=device,
        eval_chunk=eval_chunk, perturb=perturb,
        raw_noise_std=raw_noise_std, sigma_activation=sigma_activation
    )
    return out["rgb"]

# ------------------------- Camera path rendering -------------------------

def _look_at(
    eye: np.ndarray,
    target: np.ndarray = np.zeros(3),
    up: np.ndarray = np.array([0, 1, 0]),
    convention: str = "opengl"
) -> np.ndarray:
    """
    Build a camera-to-world (c2w) transform with a selectable camera convention.

    Conventions (camera axes):
      - 'opengl'/'blender'/'nerf' : +X right, +Y up,     camera forward = -Z
      - 'opencv'/'colmap'         : +X right, +Y down,   camera forward = +Z
      - 'pytorch3d'/'p3d'         : +X right, +Y up,     camera forward = +Z

    IMPORTANT: Use the **same** convention here and in get_camera_rays().

    Parameters
    ----------
    eye : np.ndarray
        World position of the camera, shape (3,)
    target : np.ndarray
        World point the camera looks at, shape (3,)
    up : np.ndarray, optional
        Approximate world up direction, shape (3,). Defaults to [0, 1, 0]
    convention : str, optional
        One of {'opengl'|'blender'|'nerf', 'opencv'|'colmap', 'pytorch3d'|'p3d'}. Defaults to
        "opengl".

    Returns
    -------
    c2w : np.ndarray
        Array with shape (4,4) and dtype float32 camera-to-world transform mapping camera
        coords -> world coords
    """
    eps = 1e-8

    # Ensure float32 arrays
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    # Forward direction from eye to target (world space), unit-length
    forward_vec = target - eye
    forward_norm = np.linalg.norm(forward_vec)
    if not np.isfinite(forward_norm) or forward_norm < eps:
        forward_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward_norm = 1.0
    forward_dir = forward_vec / forward_norm

    # Normalize the provided up vector
    up_norm = np.linalg.norm(up)
    if not np.isfinite(up_norm) or up_norm < eps:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_norm = 1.0
    up_dir = up / up_norm

    # Right/side vector: cross(forward, up). If nearly parallel, choose an alternate up.
    side_vec = np.cross(forward_dir, up_dir)
    side_norm = np.linalg.norm(side_vec)
    if not np.isfinite(side_norm) or side_norm < eps:
        alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(forward_dir[1]) < 0.999 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        side_vec = np.cross(forward_dir, alt_up)
        side_norm = np.linalg.norm(side_vec)
        if side_norm < eps:
            side_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            side_norm = 1.0
    side_dir = side_vec / side_norm

    # True up: orthogonalize via cross(side, forward)
    up_true = np.cross(side_dir, forward_dir)
    up_true_norm = np.linalg.norm(up_true)
    if up_true_norm < eps:
        up_true = up_dir
        up_true_norm = np.linalg.norm(up_true) + eps
    up_true = up_true / up_true_norm

    conv = (convention or "opengl").lower()
    if conv in ("opengl", "blender", "nerf"):
        # camera basis in world: x=right, y=up, z=-forward
        x_cam_world = side_dir
        y_cam_world = up_true
        z_cam_world = -forward_dir
    elif conv in ("opencv", "colmap"):
        # camera basis in world: x=right, y=down (so -up_true), z=+forward
        x_cam_world = side_dir
        y_cam_world = -up_true
        z_cam_world = forward_dir
    elif conv in ("pytorch3d", "p3d"):
        # camera basis in world: x=right, y=up, z=+forward
        x_cam_world = side_dir
        y_cam_world = up_true
        z_cam_world = forward_dir
    else:
        raise ValueError(
            f"Unknown camera convention '{convention}'. "
            "Choose from: 'opengl'|'blender'|'nerf', 'opencv'|'colmap', 'pytorch3d'|'p3d'."
        )

    # Assemble rotation with columns = camera basis vectors in world coordinates
    rotation_c2w = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=1).astype(np.float32)  # (3,3)

    # Compose full 4x4 camera-to-world
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation_c2w
    c2w[:3, 3] = eye
    return c2w

def generate_camera_path_poses(
    val_scene,
    n_frames: int,
    path_type: str = "circle",            # {"circle","spiral"}
    radius: float | None = None,          # if None, derived from median cam distance
    elevation_deg: float = 0.0,
    yaw_range_deg: float = 360.0,
    res_scale: float = 1.0,
    world_up: np.ndarray | None = None,   # if None, auto-detect from dataset
    convention: str = "opengl",           # must match get_camera_rays convention
) -> tuple[list[np.ndarray], int, int, np.ndarray]:
    """
    Build a *smooth*, convention-aligned camera path poses list, plus (H,W,K).

    - Uses the same `_look_at(..., convention=...)` as rendering, so axes match
      your ray generation exactly.
    - Auto-detects world 'up' from dataset (average c2w +Y) if not provided.
    - Supports "circle" and a gentle "spiral" (radius and elevation ramp).
    - Returns poses only; actual rendering can be done elsewhere.

    Returns:
      poses: list[(4,4) float32 c2w], length n_frames
      H, W : ints (after `res_scale`)
      K    : (3,3) float32 intrinsics (scaled)
    """
    import numpy as np

    # ---------- Base resolution + intrinsics ----------
    base = val_scene.frames[0]
    H0, W0 = int(base.image.shape[0]), int(base.image.shape[1])
    K0 = np.array(base.K, dtype=np.float32)

    if float(res_scale) != 1.0:
        H = max(1, int(round(H0 * float(res_scale))))
        W = max(1, int(round(W0 * float(res_scale))))
        K = K0.copy()
        s = float(res_scale)
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] *= s; K[1, 2] *= s
    else:
        H, W, K = H0, W0, K0

    # ---------- Radius inference (median camera distance) ----------
    if radius is None:
        cams = np.stack([np.array(f.c2w, dtype=np.float32)[:3, 3] for f in val_scene.frames], axis=0)
        r = np.median(np.linalg.norm(cams, axis=1))
        if not np.isfinite(r) or r < 1e-3:
            r = 4.0
        radius = float(r)

    # ---------- World 'up' (auto) ----------
    if world_up is None:
        ups = []
        for f in val_scene.frames:
            C = np.array(f.c2w, dtype=np.float32)
            y = C[:3, 1] / (np.linalg.norm(C[:3, 1]) + 1e-8)
            ups.append(y)
        up_vec = np.mean(np.stack(ups, axis=0), axis=0)
        if np.linalg.norm(up_vec) < 1e-6:
            up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        world_up = (up_vec / (np.linalg.norm(up_vec) + 1e-8)).astype(np.float32)
    else:
        world_up = np.asarray(world_up, dtype=np.float32)
        world_up /= (np.linalg.norm(world_up) + 1e-8)

    # ---------- Orthonormal basis around 'up' for smooth circles ----------
    # Choose a helper axis not parallel to up; then build {right, forward_plane}
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(world_up @ tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(world_up, tmp); right /= (np.linalg.norm(right) + 1e-8)
    forward_plane = np.cross(right, world_up); forward_plane /= (np.linalg.norm(forward_plane) + 1e-8)

    # ---------- Angles & spiral parameters ----------
    start_elev = np.deg2rad(float(elevation_deg))
    total_yaw  = np.deg2rad(float(yaw_range_deg))

    # gentle ramps
    spiral_radius_growth_fraction = 0.15   # final radius ≈ radius * (1 + 0.15)
    spiral_elevation_delta_deg    = 15.0   # final elevation ≈ elevation_deg + 15°
    end_elev = np.deg2rad(elevation_deg + spiral_elevation_delta_deg)

    path_type_lc = str(path_type).lower()
    if path_type_lc not in ("circle", "spiral"):
        raise ValueError(f"Unsupported path_type '{path_type}'. Use 'circle' or 'spiral'.")

    # ---------- Build poses (constant angular speed) ----------
    poses: list[np.ndarray] = []
    for i in range(int(n_frames)):
        t01 = i / max(1, (n_frames - 1))

        # negative yaw → clockwise (keeps parity with prior renders)
        yaw = -total_yaw * t01

        if path_type_lc == "spiral":
            r = float(radius) * (1.0 + spiral_radius_growth_fraction * t01)
            elev_i = (1.0 - t01) * start_elev + t01 * end_elev
        else:  # circle
            r = float(radius)
            elev_i = start_elev

        r_in_plane = r * np.cos(elev_i)
        h_along_up = r * np.sin(elev_i)
        eye = (r_in_plane * (np.cos(yaw) * right + np.sin(yaw) * forward_plane)
               + h_along_up * world_up)

        # Critically: use the SAME robust look-at with convention as ray gen.
        c2w = _look_at(eye=eye, target=np.zeros(3, dtype=np.float32), up=world_up, convention=convention)
        poses.append(c2w)

    return poses, H, W, K
