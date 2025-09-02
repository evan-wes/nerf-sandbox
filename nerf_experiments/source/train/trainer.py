"""Contains code for training an MLP"""


from __future__ import annotations
import random
import math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from nerf_experiments.source.data.loaders.blender_loader import BlenderSceneLoader
from nerf_experiments.source.data.samplers import RandomPixelRaySampler
from nerf_experiments.source.utils.ray_utils import get_camera_rays
from nerf_experiments.source.utils.pose_utils import look_at

from nerf_experiments.source.models.encoders import PositionalEncoder
from nerf_experiments.source.models.mlps import NeRF
from nerf_experiments.source.render.render_utils import volume_render_rays
from nerf_experiments.source.utils.sampling_utils import sample_pdf

from nerf_experiments.source.config.runtime_config import RuntimeTrainConfig


# =============== helpers ===============

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility tradeoffs:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # keep perf

def mse2psnr(x: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(x.clamp_min(1e-10))

def stratified_samples(near: float, far: float, n_samples: int, B: int, device, dtype, perturb: bool=True) -> torch.Tensor:
    t = torch.linspace(0., 1., steps=n_samples, device=device, dtype=dtype)  # [n]
    z = near * (1. - t) + far * t
    z = z.expand(B, n_samples).contiguous()
    if perturb:
        mids = 0.5 * (z[:, 1:] + z[:, :-1])
        lower = torch.cat([z[:, :1], mids], dim=-1)
        upper = torch.cat([mids, z[:, -1:]], dim=-1)
        z = lower + (upper - lower) * torch.rand_like(z)
    return torch.sort(z, dim=-1).values

def save_rgb_png(t: torch.Tensor, path: Path):
    arr = (t.clamp(0,1).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)

def make_scheduler(opt: optim.Optimizer, name: str, params: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler | None:
    name = (name or "none").lower()
    if name in ("none", "constant"):
        return None
    if name == "cosine":
        T_max = int(params.get("T_max"))
        eta_min = float(params.get("eta_min", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    # fallback: warn + none
    print(f"[scheduler] Unknown scheduler '{name}', using constant LR.")
    return None


# =============== trainer ===============

class Trainer:
    def __init__(self, cfg: RuntimeTrainConfig) -> None:
        self.cfg = cfg
        set_global_seed(cfg.seed)

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        # 1) Load scenes (train + val for previews)  (Blender path here; extend for LLFF/COLMAP in your loader later)
        loader = BlenderSceneLoader(
            root=cfg.data_root,
            downscale=cfg.downscale,
            white_bg=cfg.white_bg,
            scene_scale=1.0,
            center_origin=False,
        )
        # training scene
        self.train_scene = loader.load(cfg.split)

        # validation scene (prefer "val", fallback → "test", then → train)
        self.val_scene = None
        for split_name in ("val", "test"):
            try:
                self.val_scene = loader.load(split_name)
                break
            except FileNotFoundError:
                pass
        if self.val_scene is None:
            # fallback for smoke tests if val/test not present
            self.val_scene = self.train_scene
        # Which frame to render for previews (can be overridden by CLI)
        self.val_frame_idx = getattr(self, "val_frame_idx", 0)

        # 2) Near/Far (RuntimeTrainConfig already resolved these)
        self.near = float(cfg.near)
        self.far  = float(cfg.far)

        # 3) Sampler
        self.sampler = RandomPixelRaySampler(
            self.train_scene,
            rays_per_batch=cfg.rays_per_batch,
            device=self.device,
            white_bg_composite=cfg.white_bg,
        )

        # 4) Encoders
        self.pos_enc = PositionalEncoder(
            input_dims=3,
            num_freqs=cfg.pos_num_freqs,
            include_input=cfg.pos_include_input,
        ).to(self.device)
        self.dir_enc = PositionalEncoder(
            input_dims=3,
            num_freqs=cfg.dir_num_freqs,
            include_input=cfg.dir_include_input,
        ).to(self.device)

        # 5) Models (coarse + fine)
        self.nerf_c = NeRF(
            enc_pos_dim=self.pos_enc.out_dim,
            enc_dir_dim=self.dir_enc.out_dim,
        ).to(self.device)
        self.nerf_f = NeRF(
            enc_pos_dim=self.pos_enc.out_dim,
            enc_dir_dim=self.dir_enc.out_dim,
        ).to(self.device)

        # 6) Optimizer + scheduler
        self.opt = optim.Adam(
            list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
            lr=cfg.lr,
        )
        self.sched = make_scheduler(self.opt, cfg.scheduler, cfg.scheduler_params)

        # 7) AMP + IO
        self.scaler = torch.amp.GradScaler(enabled=(cfg.amp and self.device.type == "cuda"))
        self.out_dir = Path(cfg.out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

        # 8) Cache render params
        self.nc = int(cfg.nc)
        self.nf = int(cfg.nf)
        self.det_fine = bool(cfg.det_fine)

    def _forward_nerf(self, rays_o: torch.Tensor, rays_d: torch.Tensor, z: torch.Tensor, model: nn.Module):
        B, N = z.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]     # [B,N,3]
        viewdirs = F.normalize(rays_d, dim=-1)
        vdirs = viewdirs[:, None, :].expand(B, N, 3)                      # [B,N,3]

        enc_pos = self.pos_enc(pts.reshape(-1, 3))
        enc_dir = self.dir_enc(vdirs.reshape(-1, 3))

        pred = model(enc_pos, enc_dir)                                    # [B*N, 4]
        rgb = pred[..., :3].reshape(B, N, 3)
        sigma = pred[..., 3].reshape(B, N)

        comp_rgb, weights, acc, depth = volume_render_rays(
            rgb=rgb, sigma=sigma, z_depths=z, white_bkgd=self.cfg.white_bg
        )
        return comp_rgb, weights, acc, depth

    def train_step(self, batch) -> dict[str, torch.Tensor]:
        rays_o, rays_d, target = batch["rays_o"], batch["rays_d"], batch["rgb"]
        B = rays_o.shape[0]
        dtype = rays_o.dtype

        with torch.amp.autocast(self.device.type, enabled=self.cfg.amp):
            # --- Coarse ---
            zc = stratified_samples(self.near, self.far, self.nc, B, device=self.device, dtype=dtype, perturb=True)
            comp_c, w_c, _, _ = self._forward_nerf(rays_o, rays_d, zc, self.nerf_c)

            # --- Hierarchical ---
            mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
            w_mid = w_c[:, 1:-1].detach()
            zf = sample_pdf(mids, w_mid, n_samples=self.nf, deterministic=self.det_fine)  # [B, Nf]
            z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

            # --- Fine ---
            comp_f, _, _, _ = self._forward_nerf(rays_o, rays_d, z_all, self.nerf_f)

            # Losses
            loss_c = F.mse_loss(comp_c, target)
            loss_f = F.mse_loss(comp_f, target)
            loss = loss_c + loss_f
            psnr = mse2psnr(loss_f.detach())

        return {"loss": loss, "psnr": psnr, "comp_f": comp_f.detach(), "comp_c": comp_c.detach()}

    @torch.no_grad()
    def validate_full_image(self, frame_idx: int = 0) -> dict[str, torch.Tensor]:
        # Clamp index into available frames
        frames = self.val_scene.frames
        if len(frames) == 0:
            raise RuntimeError("Validation scene has no frames.")
        fid = int(max(0, min(frame_idx, len(frames) - 1)))
        frame = frames[fid]
        H, W = frame.image.shape[:2]
        rays_o, rays_d = get_camera_rays(
            image_h=H, image_w=W, intrinsic_matrix=frame.K, transform_camera_to_world=frame.c2w,
            device=self.device, dtype=torch.float32, normalize_dirs=True, as_ndc=False, near_plane=self.near,
            pixels_xy=None,
        )  # [H*W,3]

        # Chunked rendering
        chunk = 8192
        out_rgb = []
        for i in range(0, H*W, chunk):
            ro = rays_o[i:i+chunk]; rd = rays_d[i:i+chunk]
            Bc = ro.shape[0]

            # coarse
            zc = stratified_samples(self.near, self.far, self.nc, Bc, device=self.device, dtype=ro.dtype, perturb=False)
            comp_c, w_c, _, _ = self._forward_nerf(ro, rd, zc, self.nerf_c)
            # pdf → fine
            mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
            zf = sample_pdf(mids, w_c[:, 1:-1], n_samples=self.nf, deterministic=True)
            z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values
            comp_f, _, _, _ = self._forward_nerf(ro, rd, z_all, self.nerf_f)

            out_rgb.append(comp_f)
        img = torch.cat(out_rgb, dim=0).reshape(H, W, 3)
        return {"rgb": img}

    def train(self):
        it = iter(self.sampler)
        self.nerf_c.train(); self.nerf_f.train()

        t0 = time.time()
        for step in range(1, self.cfg.max_steps + 1):
            batch = next(it)
            self.opt.zero_grad(set_to_none=True)

            out = self.train_step(batch)
            loss = out["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.sched is not None:
                self.sched.step()

            if step % self.cfg.log_every == 0:
                dt = time.time() - t0
                lr_now = self.opt.param_groups[0]["lr"]
                print(f"[{step:>7d}] loss={loss.item():.6f} psnr={out['psnr'].item():.2f} lr={lr_now:.2e} ({dt:.2f}s)")
                t0 = time.time()

            if step % self.cfg.val_every == 0:
                self.nerf_c.eval(); self.nerf_f.eval()
                with torch.no_grad():
                    val = self.validate_full_image(frame_idx=self.val_frame_idx)
                self.nerf_c.train(); self.nerf_f.train()
                save_rgb_png(val["rgb"], self.out_dir / f"preview_{step}.png")

            if step % self.cfg.ckpt_every == 0:
                self.save_ckpt(step)

    def save_ckpt(self, step: int):
        ckpt = {
            "step": step,
            "nerf_c": self.nerf_c.state_dict(),
            "nerf_f": self.nerf_f.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": (self.sched.state_dict() if self.sched is not None else None),
            "cfg": self.cfg.__dict__,  # frozen runtime config fields
        }
        torch.save(ckpt, self.out_dir / f"ckpt_{step}.pt")
        torch.save(ckpt, self.out_dir / "ckpt_latest.pt")

    @torch.no_grad()
    def render_pose(self, c2w: torch.Tensor, H: int, W: int, K: np.ndarray | torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """
        Render RGB for a single camera pose (c2w), returns [H,W,3] in [0,1].
        Uses deterministic sampling for validation-quality output.
        """
        rays_o, rays_d = get_camera_rays(
            image_h=H, image_w=W, intrinsic_matrix=K, transform_camera_to_world=c2w,
            device=self.device, dtype=torch.float32, normalize_dirs=True, as_ndc=False, near_plane=self.near,
            pixels_xy=None,
        )
        out_rgb = []
        for i in range(0, H*W, chunk):
            ro = rays_o[i:i+chunk]; rd = rays_d[i:i+chunk]
            Bc = ro.shape[0]
            # coarse
            zc = stratified_samples(self.near, self.far, self.nc, Bc, device=self.device, dtype=ro.dtype, perturb=False)
            comp_c, w_c, _, _ = self._forward_nerf(ro, rd, zc, self.nerf_c)
            # pdf → fine
            mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
            zf = sample_pdf(mids, w_c[:, 1:-1], n_samples=self.nf, deterministic=True)
            z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values
            comp_f, _, _, _ = self._forward_nerf(ro, rd, z_all, self.nerf_f)
            out_rgb.append(comp_f)
        img = torch.cat(out_rgb, dim=0).reshape(H, W, 3)
        return img


    @torch.no_grad()
    def render_camera_path(
        self,
        out_dir: str | Path,
        n_frames: int = 120,
        fps: int = 30,
        path_type: str = "circle",    # "circle" | "spiral"
        radius: float | None = None,
        elevation_deg: float = 0.0,
        yaw_range_deg: float = 360.0,
        res_scale: float = 1.0,
        look_at_target: torch.Tensor | None = None,
        save_video: bool = True,
    ) -> None:
        """
        Generate a path of c2w poses and render frames with current NeRF.
        Saves PNG sequence and optionally MP4 (if imageio-ffmpeg is available).
        - path_type="circle": camera moves on a horizontal circle at fixed elevation.
        - path_type="spiral": camera moves on a spiral (radius slowly changes).
        """
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        # Use validation intrinsics by default
        frame0 = self.val_scene.frames[0]
        H0, W0 = frame0.image.shape[:2]
        H = int(round(H0 * res_scale))
        W = int(round(W0 * res_scale))
        # round up to nearest multiple of 16
        H = (H + 15) // 16 * 16
        W = (W + 15) // 16 * 16
        K = frame0.K.copy()
        if res_scale != 1.0:
            K = K.copy()
            K[0, 0] *= (W / W0); K[1, 1] *= (H / H0)
            K[0, 2] *= (W / W0); K[1, 2] *= (H / H0)

        # Scene center estimate from train poses
        centers_np = np.stack([f.c2w[:3, 3] for f in self.train_scene.frames], axis=0).astype(np.float32)  # [N,3]
        centers = torch.from_numpy(centers_np).to(self.device)  # [N,3]
        center = centers.mean(dim=0)
        if look_at_target is None:
            look_at_target = center

        # Radius default from mean distance of training cameras
        dists = centers.norm(dim=-1)
        if radius is None:
            radius = float(dists.mean().item())

        elev = math.radians(elevation_deg)
        yaw0 = -0.5 * math.radians(yaw_range_deg)
        yaw1 = +0.5 * math.radians(yaw_range_deg)

        # Generate poses
        c2ws = []
        for t in range(n_frames):
            a = yaw0 + (yaw1 - yaw0) * (t / max(1, n_frames - 1))
            r = radius
            if path_type.lower() == "spiral":
                # small sinusoidal radius modulation for a gentle spiral
                r = radius * (0.9 + 0.1 * math.sin(2 * math.pi * t / n_frames))
            # position in world
            eye = torch.tensor([
                r * math.cos(a) * math.cos(elev),
                r * math.sin(elev),
                r * math.sin(a) * math.cos(elev),
            ], device=self.device, dtype=torch.float32) + center
            c2w = look_at(eye, look_at_target.to(self.device))
            c2ws.append(c2w)

        # Render frames
        png_paths = []
        for idx, pose in enumerate(tqdm(c2ws, desc="Rendering path", unit="frame")):
            img = self.render_pose(pose, H=H, W=W, K=K)
            path = out_dir / f"path_{idx:04d}.png"
            save_rgb_png(img, path)
            png_paths.append(path)

        # Optional MP4
        if save_video:
            try:
                import imageio
                mp4_path = out_dir / "render_path.mp4"
                with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8) as w:
                    for p in png_paths:
                        w.append_data(imageio.v2.imread(p))
                print(f"[render] Wrote {mp4_path}")
            except Exception as e:
                print(f"[render] Could not write MP4 ({e}); PNG sequence saved at {out_dir}")