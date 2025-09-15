from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm
import numpy as np
import torch
import imageio.v2 as imageio

from nerf_sandbox.source.utils.render_utils import (
    render_image_chunked,
    render_pose,
    linear_to_srgb,
    save_rgb_png,
    save_gray_png,
    get_camera_rays,
    generate_camera_path_poses
)

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
      - device, near, far, val_scene, pos_enc, dir_enc, nerf_c, nerf_f, out_dir
      - cfg (with white_bkgd, eval_* knobs, etc.)
      - optional TensorBoard helper `tb_logger` with add_image(...) and flush()
    """

    def __init__(self, trainer, out_dir: str | Path | None = None, tb_logger: Any | None = None):
        self.tr = trainer
        self.device = trainer.device
        self.near = float(getattr(trainer, "near", 2.0))
        self.far = float(getattr(trainer, "far", 6.0))
        self.scene = trainer.val_scene
        self.out_dir = Path(out_dir) if out_dir else (Path(trainer.out_dir) / "validation")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tb = tb_logger

        # Eval defaults
        self.eval_nc = int(getattr(trainer, "eval_nc", getattr(trainer.cfg, "eval_nc", 0)) or getattr(trainer, "nc", 64))
        self.eval_nf = int(getattr(trainer, "eval_nf", getattr(trainer.cfg, "eval_nf", 0)) or getattr(trainer, "nf", 128))
        self.eval_chunk = int(getattr(trainer, "eval_chunk", getattr(trainer.cfg, "eval_chunk", 2048)))
        self.white_bkgd = bool(getattr(trainer.cfg, "white_bkgd", True))
        self.sigma_activation = "relu" if bool(getattr(trainer, "vanilla", False)) \
                               else str(getattr(trainer, "sigma_activation", "softplus"))
        self.raw_noise_std = 0.0 if bool(getattr(trainer, "vanilla", False)) \
                               else float(getattr(trainer, "raw_noise_std", 0.0))

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

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

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
        """One-off renders (no step suffix). Useful outside the step loop."""
        idxs = self._resolve_frame_indices(frame_indices, filenames)
        out_paths: List[Path] = []

        for fid in tqdm(idxs, desc="[VAL] selected frames"):
            fr = self.scene.frames[fid]
            H0, W0 = fr.image.shape[:2]
            K = np.array(fr.K, dtype=np.float32)

            if res_scale != 1.0:
                H = max(1, int(round(H0 * res_scale)))
                W = max(1, int(round(W0 * res_scale)))
                s = float(res_scale)
                K[0, 0] *= s; K[1, 1] *= s
                K[0, 2] *= s; K[1, 2] *= s
            else:
                H, W = H0, W0

            rays_o, rays_d = get_camera_rays(
                image_h=H, image_w=W, intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                device=self.device, dtype=torch.float32, normalize_dirs=True, as_ndc=False,
                near_plane=self.near, pixels_xy=None,
            )

            res = render_image_chunked(
                rays_o=rays_o, rays_d=rays_d, H=H, W=W,
                near=self.near, far=self.far,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                nc_eval=self.eval_nc, nf_eval=self.eval_nf,
                white_bkgd=self.white_bkgd, device=self.device,
                eval_chunk=self.eval_chunk, perturb=False,
                raw_noise_std=self.raw_noise_std, sigma_activation=self.sigma_activation,
            )

            rgb_lin = res["rgb"]
            acc = res["acc"].squeeze(-1)
            depth = res["depth"].squeeze(-1)
            depth_norm = ((depth - self.near) / (self.far - self.near + 1e-8)).clamp(0, 1)
            rgb_srgb = linear_to_srgb(rgb_lin)

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
                    self.tb.add_image(f"val/{fid}/rgb", rgb_srgb, step)
                    self.tb.add_image(f"val/{fid}/opacity", acc, step)
                    self.tb.add_image(f"val/{fid}/depth", depth_norm, step)
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
        res_scale: float = 1.0,
        log_to_tb: bool = False,
    ) -> List[Path]:
        """
        Render RGB/opacity/depth for each index, saving with a step tag:
          validation/val_idx0003/{rgb|opacity|depth}/step_0000500.png

        Returns the list of written paths.
        """
        paths: List[Path] = []
        idxs = self._resolve_frame_indices(frame_indices, None)
        for fid in tqdm(idxs, desc=f"[VAL step {step}] indices"):
            fr = self.scene.frames[fid]
            H0, W0 = fr.image.shape[:2]
            K = np.array(fr.K, dtype=np.float32)

            if res_scale != 1.0:
                H = max(1, int(round(H0 * res_scale)))
                W = max(1, int(round(W0 * res_scale)))
                s = float(res_scale)
                K[0, 0] *= s; K[1, 1] *= s
                K[0, 2] *= s; K[1, 2] *= s
            else:
                H, W = H0, W0

            rays_o, rays_d = get_camera_rays(
                image_h=H, image_w=W, intrinsic_matrix=K, transform_camera_to_world=fr.c2w,
                device=self.device, dtype=torch.float32, normalize_dirs=True, as_ndc=False,
                near_plane=self.near, pixels_xy=None,
            )

            res = render_image_chunked(
                rays_o=rays_o, rays_d=rays_d, H=H, W=W,
                near=self.near, far=self.far,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                nc_eval=self.eval_nc, nf_eval=self.eval_nf,
                white_bkgd=self.white_bkgd, device=self.device,
                eval_chunk=self.eval_chunk, perturb=False,
                raw_noise_std=self.raw_noise_std, sigma_activation=self.sigma_activation,
            )

            rgb_lin = res["rgb"]
            acc = res["acc"].squeeze(-1)               # accumulated opacity (density projection)
            depth = res["depth"].squeeze(-1)
            depth_norm = ((depth - self.near) / (self.far - self.near + 1e-8)).clamp(0, 1)
            rgb_srgb = linear_to_srgb(rgb_lin)

            # Per-index subdirs
            droot = self.out_dir / f"val_idx{fid:04d}"
            d_rgb = droot / "rgb"
            d_op  = droot / "opacity"
            d_dp  = droot / "depth"
            for d in (d_rgb, d_op, d_dp):
                d.mkdir(parents=True, exist_ok=True)

            p_rgb = d_rgb / f"step_{int(step):07d}.png"
            p_op  = d_op  / f"step_{int(step):07d}.png"
            p_dp  = d_dp  / f"step_{int(step):07d}.png"
            save_rgb_png(rgb_srgb, p_rgb)
            save_gray_png(acc, p_op)
            save_gray_png(depth_norm, p_dp)
            paths += [p_rgb, p_op, p_dp]

            if log_to_tb and self.tb is not None:
                try:
                    self.tb.add_image(f"val/{fid}/rgb", rgb_srgb, int(step))
                    self.tb.add_image(f"val/{fid}/opacity", acc, int(step))
                    self.tb.add_image(f"val/{fid}/depth", depth_norm, int(step))
                    self.tb.flush()
                except Exception:
                    pass

        return paths

    def export_val_videos_for_indices(
        self,
        frame_indices: Sequence[int],
        *,
        fps: int = 24,
        out_suffix: str = "",
    ) -> List[Path]:
        """
        Assemble per-index MP4s (RGB, depth, opacity) from the step-tagged PNGs.

          validation/val_idx0003/rgb/step_*.png   -> validation/val_idx0003_rgb.mp4
          validation/val_idx0003/depth/step_*.png -> validation/val_idx0003_depth.mp4
          validation/val_idx0003/opacity/step_*.png -> validation/val_idx0003_opacity{suffix}.mp4
        """
        vids: List[Path] = []
        idxs = self._resolve_frame_indices(frame_indices, None)
        for fid in tqdm(idxs, desc="[EXPORT] val videos (indices)"):
            root = self.out_dir / f"val_idx{fid:04d}"
            triples = [
                ("rgb",     root / "rgb",     root.parent / f"val_idx{fid:04d}_rgb{out_suffix}.mp4"),
                ("depth",   root / "depth",   root.parent / f"val_idx{fid:04d}_depth{out_suffix}.mp4"),
                ("opacity", root / "opacity", root.parent / f"val_idx{fid:04d}_opacity{out_suffix}.mp4"),
            ]
            for label, d, out in triples:
                frames = sorted(d.glob("step_*.png"))
                if not frames:
                    continue
                imgs = [imageio.imread(p) for p in tqdm(frames, desc=f"  [{label}] idx={fid}", leave=False)]
                imageio.mimwrite(out, imgs, fps=int(fps), codec="libx264", quality=8)
                vids.append(out)
        return vids

    def setup_progress_plan(
        self,
        *,
        val_steps: Sequence[int],          # REQUIRED: list of training iterations when validation runs
        n_frames: int,                     # total frames in the progress video
        path_type: str = "circle",         # "circle" | "spiral"
        radius: Optional[float] = None,    # if None, derived from median camera distance
        elevation_deg: float = 0.0,
        yaw_range_deg: float = 360.0,
        res_scale: float = 1.0,
        fps: int = 24,
        world_up: Optional[np.ndarray] = None,  # if None, auto-detect from scene
        frames_subdir: str = "progress_by_val",
    ) -> None:
        """
        Plan a progress-video that is rendered incrementally across *this* validation schedule.

        - `val_steps` defines WHEN validation happens (length E).
        - We generate ONE smooth camera path of `n_frames` poses.
        - We split those poses into E contiguous blocks so the camera moves at a
        constant angular speed in the final video regardless of early-dense validation.

        Stored state:
        _prog_pose_list      : List[np.ndarray(4,4)] length n_frames
        _prog_H, _prog_W, _prog_K
        _prog_block_sizes    : list[int] of length E (sum = n_frames)
        _prog_next_block_idx : which block to render next
        _prog_frames_dir     : output directory
        _prog_val_steps      : copy of the schedule (for reference/diagnostics)
        _prog_fps            : fps for final video muxing
        """
        # ----------------------------
        # Validate inputs
        # ----------------------------
        val_steps = list(dict.fromkeys(int(s) for s in val_steps))  # de-dup, preserve order
        assert len(val_steps) >= 1, "val_steps must contain at least one iteration"
        assert n_frames > 0, "n_frames must be > 0"

        scene = self.tr.val_scene

        # ----------------------------
        # Determine radius if not provided
        # ----------------------------
        if radius is None:
            cam_positions = np.stack([np.array(f.c2w, dtype=np.float32)[:3, 3] for f in scene.frames], axis=0)
            dists = np.linalg.norm(cam_positions, axis=1)
            med = np.median(dists)
            if not np.isfinite(med) or med < 1e-3:
                med = 4.0  # sensible default for Blender synthetic
            radius = float(med)

        # ----------------------------
        # Auto-detect world-up if needed
        # ----------------------------
        if world_up is None:
            y_axes = []
            for f in scene.frames:
                C = np.array(f.c2w, dtype=np.float32)
                y_axes.append(C[:3, 1] / (np.linalg.norm(C[:3, 1]) + 1e-8))
            up_vec = np.mean(np.stack(y_axes, axis=0), axis=0)
            if np.linalg.norm(up_vec) < 1e-6:
                up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            world_up = (up_vec / (np.linalg.norm(up_vec) + 1e-8)).astype(np.float32)
        else:
            world_up = np.asarray(world_up, dtype=np.float32)
            world_up /= (np.linalg.norm(world_up) + 1e-8)

        # ----------------------------
        # Build the single, smooth camera path (constant angular spacing)
        # (Uses your convention-respecting generator from render_utils)
        # ----------------------------
        poses, H, W, K = generate_camera_path_poses(
            val_scene=scene,
            n_frames=int(n_frames),
            path_type=str(path_type),
            radius=radius,
            elevation_deg=float(elevation_deg),
            yaw_range_deg=float(yaw_range_deg),
            res_scale=float(res_scale),
            world_up=world_up,
        )

        # ----------------------------
        # Evenly split n_frames across the *actual* number of validation events
        # ----------------------------
        E = len(val_steps)
        base = int(n_frames) // E
        rem  = int(n_frames) - base * E
        block_sizes = [base + (1 if i < rem else 0) for i in range(E)]
        assert sum(block_sizes) == int(n_frames)

        # ----------------------------
        # Output dirs & persistent plan state
        # ----------------------------
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
        self._prog_total_frames = int(n_frames)
        self._prog_val_steps = list(val_steps)
        self._prog_active = True

    @torch.no_grad()
    def render_progress_block(self) -> Tuple[int, int]:
        if not self._prog_active or self._prog_next_block_idx >= len(self._prog_block_sizes):
            return (0, 0)

        block_idx = self._prog_next_block_idx
        count = int(self._prog_block_sizes[block_idx])
        start_idx = sum(self._prog_block_sizes[:block_idx])
        end_idx = start_idx + count

        for i in tqdm(range(start_idx, end_idx),
                      desc=f"[PROGRESS] block {block_idx+1}/{len(self._prog_block_sizes)}"):
            p = self._prog_frames_dir / f"frame_{i:05d}.png"
            if p.exists():
                # already rendered in a previous run/resume; skip work
                continue
            c2w = self._prog_poses[i]
            self.tr.nerf_c.eval(); self.tr.nerf_f.eval()
            rgb_lin = render_pose(
                c2w=c2w, H=self._prog_H, W=self._prog_W, K=self._prog_K,
                near=self.near, far=self.far,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                device=self.device, white_bkgd=self.white_bkgd,
                eval_nc=self.eval_nc, eval_nf=self.eval_nf,
                eval_chunk=self.eval_chunk, perturb=False,
                raw_noise_std=self.raw_noise_std, sigma_activation=self.sigma_activation,
            )
            rgb_srgb = linear_to_srgb(rgb_lin)
            p = self._prog_frames_dir / f"frame_{i:05d}.png"
            save_rgb_png(rgb_srgb, p)

        self._prog_next_block_idx += 1
        return (start_idx, count)

    def finalize_progress_video(self, video_name: str = "training_progress.mp4") -> Optional[Path]:
        if not self._prog_active or self._prog_frames_dir is None:
            return None
        frames = sorted(self._prog_frames_dir.glob("frame_*.png"))
        if not frames:
            return None
        mp4 = self.out_dir / video_name
        imgs = [imageio.imread(p) for p in tqdm(frames, desc="[EXPORT] progress video (rgb)")]
        imageio.mimwrite(mp4, imgs, fps=self._prog_fps, codec="libx264", quality=8)
        return mp4

    @torch.no_grad()
    def render_final_path_video(
        self,
        *,
        video_name: str = "camera_path_final.mp4",
        frames_subdir: str = "final_path",
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
            self.setup_progress_plan(
                max_steps=int(getattr(self.tr.cfg, "max_steps", 200_000)),
                val_every=int(getattr(self.tr.cfg, "val_every", 1000)),
                n_frames=int(getattr(self.tr.cfg, "path_frames", 120)),
                num_events=1,  # not used here, but required by signature
                path_type=str(getattr(self.tr.cfg, "path_type", "circle")),
                radius=getattr(self.tr.cfg, "path_radius", None),
                elevation_deg=float(getattr(self.tr.cfg, "path_elev_deg", 10.0)),
                yaw_range_deg=float(getattr(self.tr.cfg, "path_yaw_deg", 360.0)),
                res_scale=float(getattr(self.tr.cfg, "path_res_scale", 1.0)),
                fps=int(getattr(self.tr.cfg, "path_fps", 24)),
                world_up=None,
                frames_subdir="progress_by_val",  # internal; we won't use it for final frames
            )

        # Where to put final frames
        frames_dir = self.out_dir / frames_subdir
        if frames_dir.exists() and overwrite:
            for p in frames_dir.glob("frame_*.png"):
                try: p.unlink()
                except Exception: pass
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Render all frames (RGB) with the *final* model
        for i in tqdm(range(len(self._prog_poses)), desc="[FINAL PATH] frames"):
            c2w = self._prog_poses[i]
            self.tr.nerf_c.eval(); self.tr.nerf_f.eval()
            rgb_lin = render_pose(
                c2w=c2w, H=self._prog_H, W=self._prog_W, K=self._prog_K,
                near=self.near, far=self.far,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                device=self.device, white_bkgd=self.white_bkgd,
                eval_nc=self.eval_nc, eval_nf=self.eval_nf,
                eval_chunk=self.eval_chunk, perturb=False,
                raw_noise_std=self.raw_noise_std, sigma_activation=self.sigma_activation,
            )
            rgb_srgb = linear_to_srgb(rgb_lin)
            save_rgb_png(rgb_srgb, frames_dir / f"frame_{i:05d}.png")

        # Assemble MP4
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return None
        mp4 = self.out_dir / video_name
        imgs = [imageio.imread(p) for p in tqdm(frames, desc="[FINAL PATH] encode")]
        imageio.mimwrite(mp4, imgs, fps=self._prog_fps, codec="libx264", quality=8)
        return mp4


    @torch.no_grad()
    def render_camera_path(
        self,
        *,
        n_frames: int = 120,
        fps: int = 30,
        path_type: str = "circle",      # "circle" | "spiral"
        radius: Optional[float] = None,
        elevation_deg: float = 0.0,
        yaw_range_deg: float = 360.0,
        res_scale: float = 1.0,
        world_up: Optional[np.ndarray] = None,
        convention: str = "opengl",
        out_subdir: str = "final_path",
        basename: str = "camera_path",
    ) -> Path:
        """
        Render a smooth camera path with the *current* model state and assemble an MP4.
        Uses the same convention-aware pose generator as training/validation.
        """
        out_dir = (self.out_dir / out_subdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build poses + intrinsics once
        poses, H, W, K = generate_camera_path_poses(
            val_scene=self.scene,
            n_frames=int(n_frames),
            path_type=str(path_type),
            radius=radius,
            elevation_deg=float(elevation_deg),
            yaw_range_deg=float(yaw_range_deg),
            res_scale=float(res_scale),
            world_up=world_up if world_up is None else np.asarray(world_up, dtype=np.float32),
            convention=str(convention),
        )

        # Render frames
        frame_paths: List[Path] = []
        for i, c2w in enumerate(tqdm(poses, desc="Final camera path")):
            rgb_lin = render_pose(
                c2w=c2w, H=H, W=W, K=K,
                near=self.near, far=self.far,
                pos_enc=self.tr.pos_enc, dir_enc=self.tr.dir_enc,
                nerf_c=self.tr.nerf_c, nerf_f=self.tr.nerf_f,
                device=self.device, white_bkgd=self.white_bkgd,
                eval_nc=self.eval_nc, eval_nf=self.eval_nf,
                eval_chunk=self.eval_chunk, perturb=False,
                raw_noise_std=self.raw_noise_std, sigma_activation=self.sigma_activation,
            )
            p = out_dir / f"frame_{i:05d}.png"
            save_rgb_png(linear_to_srgb(rgb_lin), p)
            frame_paths.append(p)

        # Encode MP4
        mp4 = out_dir / f"{basename}.mp4"
        imgs = [imageio.imread(p) for p in frame_paths]
        imageio.mimwrite(mp4, imgs, fps=int(fps), codec="libx264", quality=8)
        return mp4

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