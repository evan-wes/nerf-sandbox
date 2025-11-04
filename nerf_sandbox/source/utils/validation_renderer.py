from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio
try:
    from PIL import Image as _PILImage
except Exception:
    _PILImage = None

from nerf_sandbox.source.utils.validation_schedule import build_validation_steps
from nerf_sandbox.source.utils.render_utils import (
    render_image_chunked,
    render_pose,
    save_rgb_png,
    save_gray_png
)
from nerf_sandbox.source.utils.ray_utils import (
    get_camera_rays
)
from nerf_sandbox.source.utils.path_pose_generator import PathPoseGenerator

ArrayLike = Union[np.ndarray, torch.Tensor]


class ValidationRenderer:
    """
    Validation + progress-video rendering helper.

    Roles:
      1) Render selected validation frames (RGB/acc/depth) — either one-off
         or step-tagged (so we can export a per-view time-lapse).
      2) DURING training, build up a progress video (camera path) by rendering
         small blocks of pre-computed poses at each validation step, using the
         current model state (no checkpoint reloads).

    Expects `trainer` to expose:
      - device, near_world, far_world, scene_val, pos_enc, dir_enc, nerf_c, nerf_f, out_dir
      - cfg (with white_bkgd, eval_* knobs, etc.)
      - optional TensorBoard helper `tb_logger` with add_image(...) and flush()
    """

    def __init__(self, trainer, out_dir: str | Path | None = None, tb_logger: Any | None = None):
        self.tr = trainer
        self.device = trainer.device
        self.near_world = float(getattr(trainer, "near_world", 2.0))
        self.far_world = float(getattr(trainer, "far_world", 6.0))
        self.use_ndc = bool(getattr(trainer, "use_ndc", False))
        self.ndc_near_plane_world = float(getattr(trainer, "ndc_near_plane_world", self.near_world))
        self.scene = trainer.scene_val
        self.out_dir = Path(out_dir) if out_dir else (Path(trainer.out_dir) / "validation")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tb = tb_logger
        self.convention = getattr(trainer, "camera_convention", "opengl")
        self.path_generator = PathPoseGenerator(debug=True)

        self.infinite_last_bin = getattr(trainer, "infinite_last_bin", False)
        self.composite_on_load = getattr(trainer, "composite_on_load", True)

        # Eval defaults
        self.nc_eval = int(getattr(trainer, "nc_eval", getattr(trainer.cfg, "nc_eval", 0)) or getattr(trainer, "nc", 64))
        self.nf_eval = int(getattr(trainer, "nf_eval", getattr(trainer.cfg, "nf_eval", 0)) or getattr(trainer, "nf", 128))
        self.eval_chunk = int(getattr(trainer, "eval_chunk", getattr(trainer.cfg, "eval_chunk", 2048)))
        self.white_bkgd = bool(getattr(trainer, "white_bkgd", False))
        self.sigma_activation = "relu" if bool(getattr(trainer, "vanilla", False)) \
                               else str(getattr(trainer, "sigma_activation", "softplus"))
        # Validation should be deterministic; do not add sigma noise
        self.raw_noise_std = 0.0
        self.snap_multiple = int(getattr(trainer.cfg, "snap_multiple", 16))

        # ---- Progress video state (camera path) ----
        self._prog_active: bool = False
        self._prog_frames_dir: Optional[Path] = None
        self._prog_total_frames: int = 0
        self._prog_block_sizes: List[int] = []
        self._prog_next_block_idx: int = 0
        self._prog_poses: List[np.ndarray] = []
        self._prog_H: int = 0
        self._prog_W: int = 0
        self._prog_K: np.ndarray = np.eye(3, dtype=np.float32)
        self._prog_fps: int = 24

        print(f"[DEBUG]: ValidationRenderer.white_bkgd: {self.white_bkgd}, ValidationRenderer.infinite_last_bin: {self.infinite_last_bin}")

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _snap_hwk(self, H: int, W: int, K) -> tuple[int, int, np.ndarray]:
        """
        Snap (H, W) up to multiples of self.snap_multiple and scale intrinsics
        so rays stay consistent. If already aligned, returns inputs unchanged.
        """
        m = int(getattr(self, "snap_multiple", 16))
        if m <= 1:
            return H, W, K

        Hs = ((int(H) + m - 1) // m) * m
        Ws = ((int(W) + m - 1) // m) * m
        if Hs == H and Ws == W:
            return H, W, K

        sx = Ws / float(W)
        sy = Hs / float(H)
        K2 = K.copy()
        # preserve FOV by scaling fx, fy and principal point
        K2[0, 0] *= sx  # fx
        K2[1, 1] *= sy  # fy
        K2[0, 2] *= sx  # cx
        K2[1, 2] *= sy  # cy
        return Hs, Ws, K2

    def _auto_world_up(self) -> np.ndarray:
        ups = np.stack([f.c2w[:3, 1] for f in self.scene.frames], axis=0)
        up_vec = ups.mean(axis=0)
        up_vec /= (np.linalg.norm(up_vec) + 1e-8)
        return up_vec.astype(np.float32)

    def _basis_from_up(self, up_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-8)
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(up_vec @ tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(up_vec, tmp); right /= (np.linalg.norm(right) + 1e-8)
        fwdp  = np.cross(right, up_vec); fwdp /= (np.linalg.norm(fwdp) + 1e-8)
        return right, up_vec, fwdp

    def _infer_radius(self) -> float:
        cams = np.stack([f.c2w[:3, 3] for f in self.scene.frames], axis=0)
        r = np.median(np.linalg.norm(cams, axis=1))
        return float(r if np.isfinite(r) and r >= 1e-3 else 4.0)

    def _resolve_frame_indices(
        self, frame_indices: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None
    ) -> List[int]:
        idxs: List[int] = []

        if frame_indices:
            idxs.extend(int(i) for i in frame_indices)

        if filenames:
            # match via Frame.meta.get('file_path' or 'basename')
            meta_paths = []
            for i, f in enumerate(self.scene.frames):
                fp = None
                if isinstance(getattr(f, "meta", None), dict):
                    fp = f.meta.get("file_path") or f.meta.get("basename")
                meta_paths.append((i, fp))
            for target in filenames:
                target = str(target).strip()
                matched = None
                for i, fp in meta_paths:
                    if not fp: continue
                    p = Path(fp)
                    if target in (fp, p.name, p.stem):
                        matched = i; break
                if matched is None:
                    raise ValueError(f"Could not resolve filename '{target}' to a validation frame.")
                idxs.append(matched)

        if not idxs:
            idxs = [0]
        idxs = sorted(set(max(0, min(i, len(self.scene.frames) - 1)) for i in idxs))
        return idxs

    def _compute_psnr(self, pred_rgb_hw3: torch.Tensor,
                  gt_rgb_hw3: torch.Tensor,
                  mask_hw1: torch.Tensor | None = None) -> float:
        """
        pred_rgb_hw3, gt_rgb_hw3: (H,W,3) sRGB in [0,1]
        mask_hw1: optional (H,W,1) in [0,1], 1 = valid pixel
        """
        # Ensure same device/dtype
        device = pred_rgb_hw3.device
        pred = pred_rgb_hw3.clamp(0, 1).to(dtype=torch.float32, device=device)
        gt   = gt_rgb_hw3.clamp(0, 1).to(dtype=torch.float32, device=device)

        if mask_hw1 is not None:
            m = mask_hw1.to(dtype=torch.float32, device=device)
            # Use single-channel mask; broadcast to 3 channels when weighting the error
            if m.shape[-1] != 1:
                m = m[..., :1]
            diff2 = (pred - gt) ** 2
            diff2_weighted = diff2 * m                # broadcast over channel dim
            denom = (m.sum() * pred.shape[-1]).clamp_min(1e-8)  # effective #weighted channels
            mse = diff2_weighted.sum() / denom
        else:
            mse = torch.nn.functional.mse_loss(pred, gt, reduction="mean")

        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        return float(psnr.item())

    def _to_torch_hw3_or_hw4(self, x) -> torch.Tensor:
        """
        Convert x (torch.Tensor | np.ndarray | PIL.Image) to float32 torch tensor in [0,1],
        shape HxWx3 or HxWx4. Accepts CHW or HWC; converts to HWC.
        """
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif (_PILImage is not None) and isinstance(x, _PILImage.Image):
            t = torch.from_numpy(np.array(x))
        else:
            raise TypeError(f"Unsupported GT type: {type(x)}")

        # If CHW, move to HWC
        if t.ndim == 3 and t.shape[0] in (3, 4):
            t = t.permute(1, 2, 0)

        # Now expect HWC
        if t.ndim != 3 or t.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4, got shape {tuple(t.shape)}")

        # Normalize dtype/range
        if t.dtype == torch.uint8:
            t = t.to(torch.float32) / 255.0
        else:
            t = t.to(torch.float32)
            if t.max().item() > 1.5:  # in case it's [0,255] float
                t = t / 255.0

        return t

    def _get_frame_gt(self, scene, frame_idx: int, target_hw: tuple[int, int], *, use_mask: bool):
        """
        Returns (gt_rgb_hw3, mask_hw1_or_None) as float32 in [0,1], resized to target_hw.
        """
        f = scene.frames[frame_idx]

        # Grab any plausible RGB[A] source (adapt attribute names if yours differ)
        src = None
        for attr in ("rgb", "image", "rgb_srgb", "image_srgb", "np_rgb"):
            if hasattr(f, attr):
                src = getattr(f, attr)
                break
        if src is None:
            raise RuntimeError(f"No GT image attribute found for frame {frame_idx}")

        gt = self._to_torch_hw3_or_hw4(src)

        # Split alpha if present
        if gt.shape[-1] == 4:
            rgb = gt[..., :3]
            alpha = gt[..., 3:4]
        else:
            rgb, alpha = gt, None

        # Resize to renderer resolution if needed
        Ht, Wt = target_hw
        if (rgb.shape[0] != Ht) or (rgb.shape[1] != Wt):
            rgb = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0),
                                size=(Ht, Wt), mode="bilinear", align_corners=False
                            ).squeeze(0).permute(1, 2, 0)
            if alpha is not None:
                alpha = F.interpolate(alpha.permute(2, 0, 1).unsqueeze(0),
                                    size=(Ht, Wt), mode="nearest"
                                    ).squeeze(0).permute(1, 2, 0)

        mask = alpha if (use_mask and alpha is not None) else None
        return rgb, mask

    # -------------------------------------------------------------------------
    # Static (one-off) validation frames (RGB/acc/depth), no step tag
    # -------------------------------------------------------------------------


    @torch.no_grad()
    def render_selected_frames(
        self,
        frame_indices: Optional[Sequence[int]] = None,
        filenames: Optional[Sequence[str]] = None,
        res_scale: float = 1.0,
        log_to_tb: bool = False,
        tb_step: Optional[int] = None,
    ) -> List[Path]:
        idxs = self._resolve_frame_indices(frame_indices, filenames)
        out_paths: List[Path] = []

        for fid in tqdm(idxs, desc="[VAL] selected frames"):
            fr = self.scene.frames[fid]
            H0, W0 = fr.image.shape[:2]
            K = np.array(fr.K, dtype=np.float32)

            # Scale intrinsics + resolution together
            if res_scale != 1.0:
                s = float(res_scale)
                H = max(1, int(round(H0 * s)))
                W = max(1, int(round(W0 * s)))
                K[0, 0] *= s; K[1, 1] *= s
                K[0, 2] *= s; K[1, 2] *= s
            else:
                H, W = H0, W0

            # -------- WORLD/CAMERA rays for MLP viewdirs (always as_ndc=False) --------
            (
                rays_o_world,            # (H*W, 3)
                rays_d_world_unit,       # (H*W, 3)
                rays_d_world_norm,       # (H*W, 1)
                _r_o_m_IGN,              # unused here
                _r_d_m_u_IGN,            # unused here
                _r_d_m_n_IGN,            # unused here
            ) = get_camera_rays(
                image_h=H, image_w=W,
                intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                device=self.device, dtype=torch.float32,
                convention=self.convention,
                pixel_center=True,                 # keep consistent with K
                as_ndc=False,                       # WORLD rays for dir encoding
                near_plane=self.near_world,         # not used in this branch
                pixels_xy=None,
            )

            # -------- Marching rays (NDC if requested, else reuse WORLD) --------
            if self.use_ndc:
                (
                    _r_o_w_IGN, _r_d_w_u_IGN, _r_d_w_n_IGN,
                    rays_o_marching,           # (H*W, 3)
                    rays_d_marching_unit,      # (H*W, 3)
                    rays_d_marching_norm,      # (H*W, 1)
                ) = get_camera_rays(
                    image_h=H, image_w=W,
                    intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                    device=self.device, dtype=torch.float32,
                    convention=self.convention,
                    pixel_center=True,
                    as_ndc=True,
                    near_plane=float(self.ndc_near_plane_world),
                    pixels_xy=None,
                )
                near_render, far_render = 0.0, 1.0
            else:
                rays_o_marching      = rays_o_world
                rays_d_marching_unit = rays_d_world_unit
                rays_d_marching_norm = rays_d_world_norm
                near_render, far_render = self.near_world, self.far_world

            # -------- Render --------
            res = render_image_chunked(
                rays_o=rays_o_marching,                 # (H*W, 3)
                rays_d_unit=rays_d_marching_unit,       # (H*W, 3) unit marching dirs
                ray_norms=rays_d_marching_norm,         # (H*W, 1) pre-norm ||d||
                H=H, W=W,
                near=near_render, far=far_render,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                nc_eval=self.nc_eval, nf_eval=self.nf_eval,
                white_bkgd=self.white_bkgd, device=self.device,
                eval_chunk=self.eval_chunk, perturb=False,
                sigma_activation=self.sigma_activation,
                # WORLD/CAMERA unit viewdirs go to the MLP (never NDC)
                viewdirs_world_unit=rays_d_world_unit,
                infinite_last_bin=self.infinite_last_bin,
            )

            # Renderer returns linear→sRGB already; keep clamp for safety.
            rgb_srgb = res["rgb"].clamp(0, 1)
            acc      = res["acc"].squeeze(-1)
            depth    = res["depth"].squeeze(-1)

            depth_norm = (depth.clamp(0, 1) if self.use_ndc
                        else ((depth - self.near_world) / (self.far_world - self.near_world + 1e-8)).clamp(0, 1))

            p_rgb   = self.out_dir / f"val_idx{fid:04d}.png"
            p_acc   = self.out_dir / f"val_idx{fid:04d}_opacity.png"
            p_depth = self.out_dir / f"val_idx{fid:04d}_depth.png"
            save_rgb_png(rgb_srgb, p_rgb)
            save_gray_png(acc, p_acc)
            save_gray_png(depth_norm, p_depth)
            out_paths += [p_rgb, p_acc, p_depth]

            if log_to_tb and self.tb is not None:
                step = int(tb_step if tb_step is not None else 0)
                try:
                    self.tb.add_image(f"val/{fid}/rgb",    rgb_srgb,  step)
                    self.tb.add_image(f"val/{fid}/opacity", acc,       step)
                    self.tb.add_image(f"val/{fid}/depth",   depth_norm, step)
                    self.tb.flush()
                except Exception:
                    pass

        return out_paths


    # -------------------------------------------------------------------------
    # Step-tagged validation frames (for per-index time-lapse videos)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def render_indices_at_step(
        self,
        step: int,
        frame_indices: Sequence[int],
        *,
        use_mask: bool | str = "auto",
        res_scale: float = 1.0,
        log_to_tb: bool = False,
    ) -> tuple[List[Path], dict[str, float]]:

        paths: List[Path] = []
        psnrs: List[float] = []
        idxs = self._resolve_frame_indices(frame_indices, None)

        for fid in tqdm(idxs, desc=f"[VAL step {step}] indices"):
            fr = self.scene.frames[fid]
            H0, W0 = fr.image.shape[:2]
            K = np.array(fr.K, dtype=np.float32)

            # Scale intrinsics + resolution together
            if res_scale != 1.0:
                s = float(res_scale)
                H = max(1, int(round(H0 * s)))
                W = max(1, int(round(W0 * s)))
                K[0, 0] *= s; K[1, 1] *= s
                K[0, 2] *= s; K[1, 2] *= s
            else:
                H, W = H0, W0

            # -------- WORLD/CAMERA rays for MLP viewdirs (always as_ndc=False) --------
            (
                rays_o_world,            # (H*W, 3)
                rays_d_world_unit,       # (H*W, 3)
                rays_d_world_norm,       # (H*W, 1)
                _r_o_m_IGN,              # unused here
                _r_d_m_u_IGN,            # unused here
                _r_d_m_n_IGN,            # unused here
            ) = get_camera_rays(
                image_h=H, image_w=W,
                intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                device=self.device, dtype=torch.float32,
                convention=self.convention,
                pixel_center=True,                 # keep consistent with how K is defined
                as_ndc=False,                       # WORLD rays for dir encoding
                near_plane=self.near_world,         # not used in this branch
                pixels_xy=None,
            )

            # -------- Marching rays (NDC if requested, else reuse WORLD) --------
            if self.use_ndc:
                (
                    _r_o_w_IGN, _r_d_w_u_IGN, _r_d_w_n_IGN,
                    rays_o_marching,           # (H*W, 3)
                    rays_d_marching_unit,      # (H*W, 3)
                    rays_d_marching_norm,      # (H*W, 1)
                ) = get_camera_rays(
                    image_h=H, image_w=W,
                    intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                    device=self.device, dtype=torch.float32,
                    convention=self.convention,
                    pixel_center=True,
                    as_ndc=True,
                    near_plane=float(self.ndc_near_plane_world),
                    pixels_xy=None,
                )
                near_render, far_render = 0.0, 1.0
            else:
                rays_o_marching      = rays_o_world
                rays_d_marching_unit = rays_d_world_unit
                rays_d_marching_norm = rays_d_world_norm
                near_render, far_render = self.near_world, self.far_world

            # -------- Render --------
            res = render_image_chunked(
                rays_o=rays_o_marching,                 # (H*W,3)
                rays_d_unit=rays_d_marching_unit,       # (H*W,3) unit marching dirs
                ray_norms=rays_d_marching_norm,         # (H*W,1) pre-norms (scale Δ)
                H=H, W=W,
                near=near_render, far=far_render,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                nc_eval=self.nc_eval, nf_eval=self.nf_eval,
                white_bkgd=self.white_bkgd, device=self.device,
                eval_chunk=self.eval_chunk, perturb=False,
                sigma_activation=self.sigma_activation,
                # WORLD/CAMERA unit viewdirs go to the MLP (never NDC)
                viewdirs_world_unit=rays_d_world_unit,
                infinite_last_bin=self.infinite_last_bin,
            )

            # Renderer returns sRGB in [0,1]; keep clamp for safety
            rgb_srgb = res["rgb"].clamp(0, 1)
            acc      = res["acc"].squeeze(-1)
            depth    = res["depth"].squeeze(-1)
            depth_norm = (depth.clamp(0, 1) if self.use_ndc
                        else ((depth - self.near_world) / (self.far_world - self.near_world + 1e-8)).clamp(0, 1))

            # Save per-step images
            droot = self.out_dir / f"val_idx{fid:04d}"
            d_rgb, d_op, d_dp = droot / "rgb", droot / "opacity", droot / "depth"
            for d in (d_rgb, d_op, d_dp):
                d.mkdir(parents=True, exist_ok=True)
            p_rgb = d_rgb / f"step_{int(step):07d}.png"
            p_op  = d_op  / f"step_{int(step):07d}.png"
            p_dp  = d_dp  / f"step_{int(step):07d}.png"
            save_rgb_png(rgb_srgb, p_rgb)
            save_gray_png(acc, p_op)
            save_gray_png(depth_norm, p_dp)
            paths += [p_rgb, p_op, p_dp]

            # -------- PSNR against GT (masking auto/override) --------
            pred = rgb_srgb  # (H,W,3), torch, sRGB in [0,1]
            use_mask_bool = use_mask if isinstance(use_mask, bool) else (not self.composite_on_load)

            gt, mask = self._get_frame_gt(
                scene=self.scene,
                frame_idx=fid,
                target_hw=(pred.shape[0], pred.shape[1]),
                use_mask=use_mask_bool
            )

            # Device/dtype guards (avoids cuda/cpu mismatch)
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            gt   = gt.to(device=pred.device, dtype=pred.dtype)
            mask = (mask.to(device=pred.device, dtype=pred.dtype) if mask is not None else None)

            psnr_val = self._compute_psnr(pred, gt, mask)
            psnrs.append(psnr_val)

            if log_to_tb and self.tb is not None:
                try:
                    self.tb.add_image(f"val/{fid}/rgb",     rgb_srgb,  int(step))
                    self.tb.add_image(f"val/{fid}/opacity", acc,        int(step))
                    self.tb.add_image(f"val/{fid}/depth",   depth_norm, int(step))
                    self.tb.add_scalar(f"val/psnr_frame_{fid}", psnr_val, int(step))
                    self.tb.flush()
                except Exception:
                    pass

        val_psnr_metrics = {
            "psnr_per_frame": psnrs,
            "psnr_mean": (sum(psnrs) / len(psnrs)) if psnrs else None,
        }
        if log_to_tb and self.tb is not None and val_psnr_metrics["psnr_mean"] is not None:
            self.tb.add_scalar("val/psnr_mean", val_psnr_metrics["psnr_mean"], step)

        return paths, val_psnr_metrics


    def setup_progress_plan(
        self,
        *,
        val_steps,
        frames_subdir: str = "training_progress",
    ) -> None:

        # --- validate schedule ---
        val_steps = list(dict.fromkeys(int(s) for s in val_steps))
        assert len(val_steps) >= 1
        cfg = self.tr.cfg
        scene = self.tr.scene_val

        # --- pull config (safe defaults) ---
        n_frames        = int(getattr(cfg, "progress_frames", 120))
        path_type       = str(getattr(cfg, "path_type", "llff_spiral")) # "blender" | "llff_spiral" | "llff_zflat"
        res_scale       = float(getattr(cfg, "path_res_scale", 1.0))
        fps             = int(getattr(cfg, "path_fps", 24))

        # Blender-only knobs (ignored for LLFF)
        bl_phi_deg          = float(getattr(cfg, "bl_phi_deg", -30.0))
        bl_radius_val       = getattr(cfg, "bl_radius", None)
        bl_radius           = None if bl_radius_val is None else float(bl_radius_val)
        bl_theta_start_deg  = float(getattr(cfg, "bl_theta_start_deg", -180.0))
        bl_rots             = float(getattr(cfg, "bl_rots", 1.0))

        # LLFF-only knobs (ignored for Blender)
        data_root   = getattr(cfg, "data_root", None) # where to find poses_bounds.npy for LLFF paths
        rots        = float(getattr(cfg, "rots", 2.0))
        zrate       = float(getattr(cfg, "zrate", 0.5))
        path_zflat  = bool(getattr(cfg, "path_zflat", False))
        bd_factor   = float(getattr(cfg, "bd_factor", 0.75))


        poses, H, W, K = self.path_generator.generate(
            scene_val=scene,
            n_frames=n_frames,
            path_type=path_type,
            res_scale=res_scale,

            # Blender-only knobs
            bl_phi_deg=bl_phi_deg,
            bl_radius=bl_radius,
            bl_theta_start_deg=bl_theta_start_deg,
            bl_rots=bl_rots,

            # LLFF-only knobs
            data_root=data_root,
            rots=rots,
            zrate=zrate,
            path_zflat=path_zflat,
            bd_factor=bd_factor,
        )

        # --- quick ring sanity print (so you can verify non-degenerate radii) ---
        P = np.stack([p[:3, 3] for p in poses], 0)
        d = np.linalg.norm(P - P.mean(0, keepdims=True), axis=1)
        print(f"[progress-plan] frames={n_frames} type={path_type} res_scale={res_scale} "
            f"| ring_radius_med≈{np.median(d):.4f} ring_radius_p90≈{np.percentile(d,90):.4f} "
            f"| meanΔpos/frame={np.mean(np.linalg.norm(np.diff(P,axis=0),axis=1)):.6f} ")

        # --- H/W/K from VALIDATION scene at res_scale (for rendering) ---
        H0, W0 = self.tr.scene_val.frames[0].image.shape[:2]
        K0 = np.array(self.tr.scene_val.frames[0].K, dtype=np.float32)
        if res_scale != 1.0:
            H = max(1, int(round(H0 * res_scale)))
            W = max(1, int(round(W0 * res_scale)))
            K = K0.copy()
            K[0,0] *= res_scale; K[1,1] *= res_scale
            K[0,2] *= res_scale; K[1,2] *= res_scale
        else:
            H, W, K = int(H0), int(W0), K0

        # --- split frames evenly across validation events ---
        E = len(val_steps)
        base = n_frames // E
        rem  = n_frames - base * E
        block_sizes = [base + (1 if i < rem else 0) for i in range(E)]

        # --- persist plan ---
        self._prog_frames_dir = Path(self.out_dir) / str(frames_subdir)
        (self._prog_frames_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self._prog_frames_dir / "depth").mkdir(parents=True, exist_ok=True)
        (self._prog_frames_dir / "opacity").mkdir(parents=True, exist_ok=True)

        self._prog_poses = poses
        self._prog_H, self._prog_W = int(H), int(W)
        self._prog_K = K.astype(np.float32)
        self._prog_fps = int(fps)

        self._prog_block_sizes = block_sizes
        self._prog_next_block_idx = 0
        self._prog_total_frames = n_frames
        self._prog_val_steps = list(val_steps)
        self._prog_active = True

        self._prog_H, self._prog_W, self._prog_K = self._snap_hwk(self._prog_H, self._prog_W, self._prog_K)


    @torch.no_grad()
    def render_progress_block(self) -> Tuple[int, int]:
        if not self._prog_active or self._prog_next_block_idx >= len(self._prog_block_sizes):
            return (0, 0)

        block_idx = self._prog_next_block_idx
        count = int(self._prog_block_sizes[block_idx])
        start_idx = sum(self._prog_block_sizes[:block_idx])
        end_idx = start_idx + count

        # choose the NDC ray-warp plane (world units) and sampling bounds
        near_plane_world = self.ndc_near_plane_world if self.use_ndc else self.near_world
        samp_near, samp_far = (0.0, 1.0) if self.use_ndc else (self.near_world, self.far_world)

        for i in tqdm(range(start_idx, end_idx),
                      desc=f"[PROGRESS] block {block_idx+1}/{len(self._prog_block_sizes)}"):
            p = self._prog_frames_dir / f"frame_{i:05d}.png"
            if p.exists():
                # already rendered in a previous run/resume; skip work
                continue
            c2w = self._prog_poses[i]
            self.tr.nerf_c.eval(); self.tr.nerf_f.eval()
            res = render_pose(
                c2w=c2w, H=self._prog_H, W=self._prog_W, K=self._prog_K,
                # world-space near/far (kept for backward-compat)
                near=self.near_world, far=self.far_world,
                # model bits
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                device=self.device, white_bkgd=self.white_bkgd,
                nc_eval=self.nc_eval, nf_eval=self.nf_eval, eval_chunk=self.eval_chunk,
                perturb=False, sigma_activation=self.sigma_activation,
                # explicit ray and sampling controls
                use_ndc=self.use_ndc,
                convention=self.convention,
                near_plane=near_plane_world,     # world-space plane for NDC warp
                samp_near=samp_near, samp_far=samp_far,  # sampling interval (0..1 for NDC)
                infinite_last_bin=self.infinite_last_bin
            )

            # Renderer returns sRGB in [0,1]
            rgb_srgb = res["rgb"].clamp(0, 1)
            acc      = res["acc"].squeeze(-1)
            depth    = res["depth"].squeeze(-1)
            depth_norm = (depth.clamp(0, 1) if self.use_ndc
                        else ((depth - self.near_world) / (self.far_world - self.near_world + 1e-8)).clamp(0, 1))

            # Save per-step images
            d_rgb, d_op, d_dp = self._prog_frames_dir / "rgb", self._prog_frames_dir / "opacity", self._prog_frames_dir / "depth"
            for d in (d_rgb, d_op, d_dp):
                d.mkdir(parents=True, exist_ok=True)
            p_rgb = d_rgb / f"rgb_frame_{i:05d}.png"
            p_op  = d_op  / f"opacity_frame_{i:05d}.png"
            p_dp  = d_dp  / f"depth_frame_{i:05d}.png"
            save_rgb_png(rgb_srgb, p_rgb)
            save_gray_png(acc, p_op)
            save_gray_png(depth_norm, p_dp)

        self._prog_next_block_idx += 1

        self._debug_ndc_setup()
        return (start_idx, count)


    def export_triplet_videos(
        self,
        base_dir: Path,
        *,
        fps: int = 24,
        out_dir: Optional[Path] = None,
        stem: str = "",                         # e.g., "val_idx0003" or "training_progress"
        suffix: str = "",                       # e.g., "_v2"
        name_template: str = "{stem}_{type}{suffix}",  # or "{type}_{stem}{suffix}"
        frame_glob: str = "*.png",              # e.g., "step_*.png" or "*.png"
        types: Sequence[str] = ("rgb", "depth", "opacity"),
    ) -> dict[str, dict[str, Path]]:
        """
        Look for <base_dir>/{rgb,depth,opacity} subdirs, read PNG frames, and write MP4+GIF.

        Outputs (per present type):
            <out_dir>/<name_template>.mp4
            <out_dir>/<name_template>.gif
        where name_template can reference {type}, {stem}, {suffix}.

        Returns:
            { "<type>": {"mp4": Path(...), "gif": Path(...)} , ... }
        """
        base_dir = Path(base_dir)
        out_dir = Path(out_dir) if out_dir is not None else base_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, dict[str, Path]] = {}
        fps_i = int(fps)

        for t in types:
            frame_dir = base_dir / t
            if not frame_dir.is_dir():
                continue

            frames = sorted(frame_dir.glob(frame_glob))
            if not frames:
                continue

            name_stem = name_template.format(type=t, stem=stem, suffix=suffix)
            mp4_path = out_dir / f"{name_stem}.mp4"
            gif_path = out_dir / f"{name_stem}.gif"

            with imageio.get_writer(
                mp4_path, fps=fps_i, codec="libx264", quality=8, pixelformat="yuv420p"
                # ffmpeg_params=["-pix_fmt", "yuv420p"]
            ) as w_mp4, imageio.get_writer(
                gif_path, mode="I", duration=1.0 / float(fps_i), loop=0
            ) as w_gif:
                for p in tqdm(frames, desc=f"[EXPORT] {t} -> {name_stem}", leave=False):
                    img = imageio.imread(p)
                    w_mp4.append_data(img)
                    w_gif.append_data(img)

            results[t] = {"mp4": mp4_path, "gif": gif_path}

        return results

    def export_val_videos_for_indices(
        self,
        frame_indices: Sequence[int],
        *,
        fps: int = 24,
        out_suffix: str = "",
    ) -> List[Path]:
        """
        Assemble per-index MP4s + GIFs (RGB, depth, opacity) from the step-tagged PNGs.

        validation/val_idx0003/rgb/step_*.png     -> validation/val_idx0003_rgb{suffix}.mp4 and .gif
        validation/val_idx0003/depth/step_*.png   -> validation/val_idx0003_depth{suffix}.mp4 and .gif
        validation/val_idx0003/opacity/step_*.png -> validation/val_idx0003_opacity{suffix}.mp4 and .gif
        """
        idxs = self._resolve_frame_indices(frame_indices, None)

        for fid in tqdm(idxs, desc="[EXPORT] val videos (indices)"):
            root = self.out_dir / f"val_idx{fid:04d}"

            results = self.export_triplet_videos(
                base_dir=root,
                fps=fps,
                out_dir=root,
                stem=f"val_idx{fid:04d}",
                suffix=out_suffix,
                name_template="{stem}_{type}{suffix}",      # val_idx0003_rgb{suffix}.ext
                frame_glob="step_*.png",
            )
            for _, paths in results.items():
                print(f"[VAL-VIDEO] wrote → {paths['mp4']}")
                print(f"[VAL-VIDEO] wrote → {paths['gif']}")

    def export_progress_video(self, video_name: str = "training_progress") -> None:
        """
        Build progress videos for each frame type ("rgb", "opacity", "depth") from PNG
        frames. Saves both MP4 and GIF for each type.
        """
        if not self._prog_active or self._prog_frames_dir is None:
            return None

        results = self.export_triplet_videos(
            base_dir=self._prog_frames_dir,             # contains rgb/, depth/, opacity/
            fps=int(self._prog_fps),
            out_dir=self._prog_frames_dir,
            stem=video_name,
            name_template="{stem}_{type}{suffix}",
            frame_glob="*.png",
        )
        for _, paths in results.items():
            print(f"[PROGRESS] wrote → {paths['mp4']}")
            print(f"[PROGRESS] wrote → {paths['gif']}")

    @torch.no_grad()
    def render_camera_path_video(
        self,
        *,
        video_name: str = "camera_path",
        frames_subdir: str = "camera_path",
        overwrite: bool = True,
    ) -> Optional[Path]:
        """
        Render the *entire* precomputed camera path with the current (final) model state
        into a single MP4. Uses the exact same poses built in setup_progress_plan so
        the progress-video and final video are consistent.

        If the plan wasn't initialized (e.g., progress video disabled), this will call
        setup_progress_plan once using cfg path_* knobs to create the poses.
        """
        # Ensure we have a plan/poses
        if not getattr(self, "_prog_poses", None):
            # Build when there is no plan: derive a validation schedule, then plan
            max_steps = int(getattr(self.tr.cfg, "max_steps", 200_000))
            # prefer an explicit schedule if present; else uniform-ish:
            val_every = getattr(self.tr.cfg, "val_every", None)
            if val_every is not None and int(val_every) > 0:
                val_steps = build_validation_steps(max_steps, base_every=int(val_every))
            else:
                num_val_steps = int(getattr(self.tr.cfg, "num_val_steps", 100))
                power = float(getattr(self.tr.cfg, "val_power", 2.0))
                val_steps = build_validation_steps(max_steps, num_val_steps=num_val_steps,
                                                schedule="power", power=power)

            self.setup_progress_plan(
                val_steps=val_steps,
                n_frames=int(getattr(self.tr.cfg, "path_frames", 120)),
                path_type=str(getattr(self.tr.cfg, "path_type", "circle")),
                radius=getattr(self.tr.cfg, "path_radius", None),
                elevation_deg=float(getattr(self.tr.cfg, "path_elev_deg", 10.0)),
                yaw_range_deg=float(getattr(self.tr.cfg, "path_yaw_deg", 360.0)),
                res_scale=float(getattr(self.tr.cfg, "path_res_scale", 1.0)),
                fps=int(getattr(self.tr.cfg, "path_fps", 24)),
                world_up=None,
                frames_subdir="training_progress",
            )

        # Where to put final frames
        frames_dir = self.out_dir / frames_subdir
        if frames_dir.exists() and overwrite:
            for p in frames_dir.glob("frame_*.png"):
                try: p.unlink()
                except Exception: pass
        frames_dir.mkdir(parents=True, exist_ok=True)

        near_plane_world = self.ndc_near_plane_world if self.use_ndc else self.near_world
        samp_near, samp_far = (0.0, 1.0) if self.use_ndc else (self.near_world, self.far_world)

        # Render all frames (RGB) with the *final* model
        for i in tqdm(range(len(self._prog_poses)), desc="[FINAL PATH] frames"):
            c2w = self._prog_poses[i]
            self.tr.nerf_c.eval(); self.tr.nerf_f.eval()
            res = render_pose(
                c2w=c2w, H=self._prog_H, W=self._prog_W, K=self._prog_K,
                near=self.near_world, far=self.far_world,  # world-space, kept for compat
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                device=self.device, white_bkgd=self.white_bkgd,
                nc_eval=self.nc_eval, nf_eval=self.nf_eval, eval_chunk=self.eval_chunk,
                perturb=False, sigma_activation=self.sigma_activation,
                # Explicit ray + sampling controls
                use_ndc=self.use_ndc,
                convention=self.convention,
                near_plane=near_plane_world,
                samp_near=samp_near, samp_far=samp_far,
                infinite_last_bin=self.infinite_last_bin
            )

            # Renderer returns sRGB in [0,1]
            rgb_srgb = res["rgb"].clamp(0, 1)
            acc      = res["acc"].squeeze(-1)
            depth    = res["depth"].squeeze(-1)
            depth_norm = (depth.clamp(0, 1) if self.use_ndc
                        else ((depth - self.near_world) / (self.far_world - self.near_world + 1e-8)).clamp(0, 1))

            # Save per-step images
            d_rgb, d_op, d_dp = frames_dir / "rgb", frames_dir / "opacity", frames_dir / "depth"
            for d in (d_rgb, d_op, d_dp):
                d.mkdir(parents=True, exist_ok=True)
            p_rgb = d_rgb / f"rgb_frame_{i:05d}.png"
            p_op  = d_op  / f"opacity_frame_{i:05d}.png"
            p_dp  = d_dp  / f"depth_frame_{i:05d}.png"
            save_rgb_png(rgb_srgb, p_rgb)
            save_gray_png(acc, p_op)
            save_gray_png(depth_norm, p_dp)

        # Assemble vide

        results = self.export_triplet_videos(
            base_dir=frames_dir,             # contains rgb/, depth/, opacity/
            fps=int(self._prog_fps),
            out_dir=frames_dir,
            stem=video_name,
            name_template="{stem}_{type}{suffix}",
            frame_glob="*.png",
        )
        for _, paths in results.items():
            print(f"[CAMERA PATH] wrote → {paths['mp4']}")
            print(f"[CAMERA PATH] wrote → {paths['gif']}")

    def resume_to_step(self, current_step: int) -> None:
        """
        Advance internal progress state to reflect that training has already
        reached `current_step`. We also scan the frames directory to pick up
        partially written progress frames, so reruns don't duplicate work.
        """
        # If no plan is active, nothing to do
        if not getattr(self, "_prog_active", False):
            return
        if not hasattr(self, "_prog_block_sizes") or not hasattr(self, "_prog_val_steps"):
            return

        # 1) How many validation events should have fired already?
        passed_events = 0
        for s in self._prog_val_steps:
            if s <= int(current_step):
                passed_events += 1
            else:
                break

        # 2) How many frames are already on disk?
        n_existing = len(sorted((self._prog_frames_dir or self.out_dir).glob("frame_*.png")))
        cum = 0
        idx_from_disk = 0
        for i, b in enumerate(self._prog_block_sizes):
            if cum + b <= n_existing:
                cum += b
                idx_from_disk = i + 1
            else:
                break

        # Use the max of both signals (disk is the ground truth for idempotency)
        self._prog_next_block_idx = max(passed_events, idx_from_disk)


    def _debug_ndc_setup(self) -> None:
        """Print the essential NDC-related knobs once per block."""
        if self.use_ndc:
            print(
                "[NDC] enabled=True | near_plane_world="
                f"{float(self.ndc_near_plane_world):.6f} | samp_range=[0.0, 1.0] | "
                f"convention={self.convention} | Kfx={float(self._prog_K[0,0]):.6f} Kfy={float(self._prog_K[1,1]):.6f}"
            )
        else:
            print(
                "[NDC] enabled=False | world sampling range="
                f"[{float(self.near_world):.6f}, {float(self.far_world):.6f}] | "
                f"convention={self.convention}"
            )