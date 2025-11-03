"""
Trainer class

- Clean separation of concerns:
  * GPU thermal & throttling: gpu_thermal.GpuThermalManager
  * Signal handling: signal_handlers.SignalController (+ installers)
  * Rendering (stateless): render_utils.volume_render_rays
  * Validation : Utilizes the ValidationRenderer class to handle all validation
  * Logging : Uses the TensorBoardLogger class to handle all logging to tensorboard

- Vanilla NeRF compatibility:
  * cfg.vanilla switches behavior to mirror bmild/nerf defaults:
      8x256 MLP with skip after layer 3, encoders Lx=10/Ld=4, ReLU sigma with raw noise,
      N_rand=1024 (single image + precrop), N_samples=64, N_importance=128, white background.
  * Extended features remain available when vanilla mode is off.
"""

from __future__ import annotations

import os as _os
_val = _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _val:
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _val + ("," if _val else "") + "expandable_segments:True"
# Optional: also set a max split size (uncomment if you still see fragmentation)
# if "max_split_size_mb" not in _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
#     _os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",max_split_size_mb:128"

import time
import random
import re
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Local imports
from nerf_sandbox.source.data.loaders.blender_loader import BlenderSceneLoader
from nerf_sandbox.source.data.loaders.llff_loader import LLFFSceneLoader
from nerf_sandbox.source.data.samplers import RandomPixelRaySampler
from nerf_sandbox.source.models.encoders import PositionalEncoder, get_vanilla_nerf_encoders
from nerf_sandbox.source.models.mlps import log_nerf_arch, NeRF
from nerf_sandbox.source.utils.render_utils import (
    volume_render_rays,
    nerf_forward_pass,
    save_rgb_png, save_gray_png
)
from nerf_sandbox.source.utils.ray_utils import get_camera_rays
from nerf_sandbox.source.utils.sampling_utils import sample_pdf
from nerf_sandbox.source.utils.gpu_thermal import GpuThermalManager
from nerf_sandbox.source.utils.signal_handlers import SignalController, install_signal_handlers
from nerf_sandbox.source.utils.tensorboard_utils import TensorBoardLogger
from nerf_sandbox.source.utils.validation_renderer import ValidationRenderer
from nerf_sandbox.source.utils.validation_schedule import build_validation_steps
from nerf_sandbox.source.utils.debug_utils import dump_run_debug, debug_topk_fine_hit


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Good perf defaults with PyTorch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def mse2psnr(x: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(x.clamp_min(1e-10))


def make_scheduler(opt: optim.Optimizer, name: str, params: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
    name = (name or "none").lower()
    if name in ("none", "constant"):
        return None
    if name == "cosine":
        T_max = int(params.get("T_max"))
        eta_min = float(params.get("eta_min", 0.0))
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    # bmild uses TF decay; you can wire a PyTorch equivalent here as needed.
    return None

def log_scheduler_once(optimizer, scheduler, logger=print, tag="[LR]"):
    name = type(scheduler).__name__
    base_lrs = getattr(scheduler, "base_lrs", [pg["lr"] for pg in optimizer.param_groups])
    attrs = {}
    for a in ("T_max", "eta_min", "T_0", "T_i", "T_mult", "gamma", "milestones", "last_epoch"):
        if hasattr(scheduler, a):
            v = getattr(scheduler, a)
            # pretty-print milestones as sorted list
            if a == "milestones":
                v = sorted(list(v))
            attrs[a] = v
    logger(f"{tag} scheduler={name} base_lrs={base_lrs} current_lrs={[pg['lr'] for pg in optimizer.param_groups]} attrs={attrs}")

# --- Device resolution (robust to None / strings / torch.device) ---
def resolve_device(cfg_dev):
    """Resolve device from None / 'auto' / 'cpu' / 'cuda[:idx]' / 'mps' / torch.device."""
    def _has_mps() -> bool:
        try:
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            return False

    # Already a torch.device
    if isinstance(cfg_dev, torch.device):
        return cfg_dev

    # String cases
    if isinstance(cfg_dev, str) and cfg_dev.strip():
        s = cfg_dev.strip().lower()

        if s == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if _has_mps():
                return torch.device("mps")
            return torch.device("cpu")

        if s.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(s)  # respects 'cuda' or 'cuda:0'
            return torch.device("cpu")

        if s == "mps":
            return torch.device("mps" if _has_mps() else "cpu")

        # 'cpu' or other valid backends
        return torch.device(s)

    # None / empty → best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _has_mps():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg) -> None:
        """
        Minimal set expected on `cfg` (others have sane defaults below):
            data_root, device, out_dir
            white_bkgd (for Blender set), (optional) near, far
            rays_per_batch, lr, max_steps, log_every, val_every, ckpt_every

        Extended (optional) knobs used when provided:
            downscale, scene_scale, center_origin,
            pos_num_freqs, dir_num_freqs, pos_include_input, dir_include_input
            n_layers, hidden_dim, skip_pos, init_acc
            nc, nf, det_fine, raw_noise_std, sigma_activation
            amp, scheduler, scheduler_params, micro_chunks, grad_clip_norm
            thermal_throttle, gpu_temp_threshold, gpu_temp_check_every, gpu_cooldown_seconds,
            thermal_throttle_max_micro, thermal_throttle_sleep
            resume_path, auto_resume, resume_no_optim
            path_frames, path_fps, path_type, path_res_scale
            use_tb, tb_logdir
            vanilla  (bmild mode toggle)
        """
        self.cfg = cfg
        self.device = resolve_device(getattr(cfg, "device", None))
        self.vanilla: bool = bool(getattr(cfg, "vanilla", False))
        self.white_bkgd = bool(getattr(cfg, "white_bkgd", False))
        self.infinite_last_bin = bool(getattr(cfg, "infinite_last_bin", False))

        self.cache_images_on_device = bool(getattr(cfg, "cache_images_on_device", False))

        print(f"[DEBUG]: Trainer.white_bkgd: {self.white_bkgd}, Trainer.infinite_last_bin: {self.infinite_last_bin}")

        set_global_seed(int(getattr(cfg, "seed", 42)))

        # ---- Data ----

        data_kind = str(getattr(cfg, "data_kind", "auto")).lower()


        if data_kind == "auto":
            is_llff = (Path(cfg.data_root) / "poses_bounds.npy").exists()
        elif data_kind == "llff":
            is_llff = True
        else:
            is_llff = False  # blender or other

        if is_llff:
            self.loader = LLFFSceneLoader(
                root=cfg.data_root,
                downscale=int(getattr(cfg, "downscale", 1)),
                white_bkgd=self.white_bkgd,
                bd_factor=float(getattr(cfg, "bd_factor", 0.75)),
                use_llff_holdout=bool(getattr(cfg, "use_llff_holdout", True)),
                holdout_every=int(getattr(cfg, "holdout_every", 0)),
                holdout_offset=int(getattr(cfg, "holdout_offset", 0)),
            )
        else:
            self.loader = BlenderSceneLoader(
                root=cfg.data_root,
                downscale=int(getattr(cfg, "downscale", 1)),
                white_bkgd=self.white_bkgd,
                centering=str(cfg.centering).lower(),
                scene_scale=float(cfg.scene_scale)
            )

        # --- Camera convention (from loader) ---
        self.camera_convention = getattr(self.loader, "camera_convention", "opengl")

        # --- Load scenes ---
        self.scene_train = self.loader.load(getattr(cfg, "split", "train"))
        try:
            self.scene_val = self.loader.load("val")
        except FileNotFoundError:
            try:
                self.scene_val = self.loader.load("test")
            except FileNotFoundError:
                self.scene_val = self.scene_train

        # --- Marching space toggle ---
        self.use_ndc = bool(getattr(cfg, "use_ndc", False))

        # --- World-space bounds (single source of truth) ---
        if is_llff:
            user_near = getattr(cfg, "near_world", None)
            user_far  = getattr(cfg, "far_world",  None)
            if (user_near is not None) and (user_far is not None):
                self.near_world, self.far_world = float(user_near), float(user_far)
            else:
                # Use loader’s bounds with percentiles; these already respect any loader normalization.
                n, f = self.loader.get_global_near_far(
                    percentile=(
                        float(getattr(cfg, "llff_near_percentile", 5.0)),
                        float(getattr(cfg, "llff_far_percentile", 95.0)),
                    )
                )
                self.near_world, self.far_world = float(n), float(f)
        else:
            # Blender defaults; allow user override via CLI if provided
            nw = getattr(cfg, "near_world", None)
            fw = getattr(cfg, "far_world",  None)
            self.near_world = float(nw) if nw is not None else 2.0
            self.far_world  = float(fw) if fw is not None else 6.0

        # --- NDC near-plane in world units ---
        # Priority: explicit CLI/CFG > (else) fall back to the resolved near_world
        ndc_np = getattr(cfg, "ndc_near_plane_world", None)
        self.ndc_near_plane_world = float(ndc_np) if ndc_np is not None else float(self.near_world)

        # --- Sampling interval actually used for marching ---
        if self.use_ndc:
            self.samp_near, self.samp_far = 0.0, 1.0
        else:
            self.samp_near, self.samp_far = self.near_world, self.far_world

        print(
            f"[rays] use_ndc={self.use_ndc} convention={self.camera_convention} "
            f"ndc_near_plane_world={self.ndc_near_plane_world:.3f} "
            f"samp=[{self.samp_near:.3f},{self.samp_far:.3f}] "
            f"world_bounds=[{self.near_world:.3f},{self.far_world:.3f}]"
        )

        # FIXME
        print()
        self._debug_check_center_ray(self.scene_train, self.camera_convention, as_ndc=True,  near_plane_world=self.ndc_near_plane_world)
        self._debug_check_center_ray(self.scene_train, self.camera_convention, as_ndc=False, near_plane_world=self.ndc_near_plane_world)

        # ---- Sampler ----
        if self.vanilla:
            # Vanilla NeRF: single-frame batches + center precrop
            self.sampler = RandomPixelRaySampler(
                self.scene_train,
                rays_per_batch=1024,  # N_rand
                device=self.device,
                white_bg_composite=self.white_bkgd,
                cache_images_on_device=self.cache_images_on_device,
                sample_from_single_frame=True,
                precrop_iters=int(getattr(cfg, "precrop_iters", 500)),
                precrop_frac=float(getattr(cfg, "precrop_frac", 0.5)),
                convention=self.camera_convention,
                as_ndc=self.use_ndc,
                near_plane=self.ndc_near_plane_world
            )
        else:
            # Mixed-frames batches (default)
            self.sampler = RandomPixelRaySampler(
                self.scene_train,
                rays_per_batch=int(getattr(cfg, "rays_per_batch", 2048)),
                device=self.device,
                white_bg_composite=self.white_bkgd,
                cache_images_on_device=self.cache_images_on_device,
                sample_from_single_frame=bool(getattr(cfg, "sample_from_single_frame", False)),
                precrop_iters=int(getattr(cfg, "precrop_iters", 0)),
                precrop_frac=float(getattr(cfg, "precrop_frac", 1.0)),
                convention=self.camera_convention,
                as_ndc=self.use_ndc,
                near_plane=self.ndc_near_plane_world
            )

        # ---- Encoders ----
        if self.vanilla:
            self.pos_enc, self.dir_enc = get_vanilla_nerf_encoders()
            self.pos_enc = self.pos_enc.to(self.device)
            self.dir_enc = self.dir_enc.to(self.device)
        else:
            self.pos_enc = PositionalEncoder(
                input_dims=3,
                num_freqs=int(getattr(cfg, "pos_num_freqs", 10)),
                include_input=bool(getattr(cfg, "pos_include_input", True)),
            ).to(self.device)
            self.dir_enc = PositionalEncoder(
                input_dims=3,
                num_freqs=int(getattr(cfg, "dir_num_freqs", 4)),
                include_input=bool(getattr(cfg, "dir_include_input", True)),
            ).to(self.device)

        # ---- Models ----
        if self.vanilla:
            # Vanilla: 8x256, skip after layer 5 (matching your class default), ReLU sigma (noise handled in trainer)
            self.nerf_c = NeRF(
                enc_pos_dim=self.pos_enc.out_dim, enc_dir_dim=self.dir_enc.out_dim,
                n_layers=8, hidden_dim=256, skip_pos=4,
                near=self.near_world, far=self.far_world,
                initial_acc_opacity=None,    # vanilla doesn't set sigma bias init
                sigma_activation="relu"      # only relevant if initial_acc_opacity is used
            ).to(self.device)
            self.nerf_f = NeRF(
                enc_pos_dim=self.pos_enc.out_dim, enc_dir_dim=self.dir_enc.out_dim,
                n_layers=8, hidden_dim=256, skip_pos=4,
                near=self.near_world, far=self.far_world,
                initial_acc_opacity=None,
                sigma_activation="relu"
            ).to(self.device)
        else:
            self.nerf_c = NeRF(
                enc_pos_dim=self.pos_enc.out_dim,
                enc_dir_dim=self.dir_enc.out_dim,
                n_layers=int(getattr(cfg, "n_layers", 8)),
                hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
                skip_pos=int(getattr(cfg, "skip_pos", 4)),
                near=self.near_world,
                far=self.far_world,
                initial_acc_opacity=float(getattr(cfg, "initial_acc_opacity", 0.0)),
                sigma_activation=str(getattr(cfg, "sigma_activation", "softplus"))
            ).to(self.device)
            self.nerf_f = NeRF(
                enc_pos_dim=self.pos_enc.out_dim,
                enc_dir_dim=self.dir_enc.out_dim,
                n_layers=int(getattr(cfg, "n_layers", 8)),
                hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
                skip_pos=int(getattr(cfg, "skip_pos", 4)),
                near=self.near_world,
                far=self.far_world,
                initial_acc_opacity=float(getattr(cfg, "initial_acc_opacity", 0.0)),
                sigma_activation=str(getattr(cfg, "sigma_activation", "softplus"))
            ).to(self.device)

        # Print log statements to validate the MLP architecture
        log_nerf_arch(self.nerf_c, self.pos_enc, self.dir_enc, logger=print, name="NeRF-Coarse")
        if self.nerf_f is not None:
            log_nerf_arch(self.nerf_f, self.pos_enc, self.dir_enc, logger=print, name="NeRF-Fine")

        # Dump the static architecture table once (optional but nice)
        self.nerf_c._debug_dump_arch_once()
        if self.nerf_f is not None:
            self.nerf_f._debug_dump_arch_once()

        # Enable runtime checks for first K mini-batches
        K = 3  # or make this a CLI like --debug_mlp_steps 3
        self.nerf_c.enable_debug(steps=K, logger=print)
        if self.nerf_f is not None:
            self.nerf_f.enable_debug(steps=K, logger=print)

        # ---- Optimizer + Scheduler ----
        self.opt = optim.Adam(
            list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
            lr=float(getattr(cfg, "lr", 5e-4))
        )
        self.sched = make_scheduler(
            opt=self.opt,
            name=getattr(cfg, "lr_scheduler", "none"),
            params=getattr(cfg, "lr_scheduler_params", {})
        )
        log_scheduler_once(optimizer=self.opt, scheduler=self.sched, logger=print)


        # ---- Runtime toggles ----
        self.amp = bool(getattr(cfg, "amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

        # micro-chunks: per-phase overrides
        self.train_micro_chunks = int(getattr(cfg, "train_micro_chunks", getattr(cfg, "micro_chunks", 0)))
        self.eval_micro_chunks  = int(getattr(cfg, "eval_micro_chunks",  getattr(cfg, "micro_chunks", 0)))
        self.micro_chunks       = self.train_micro_chunks  # back-compat with existing code

        # MLP-level chunking (cap encoder+MLP query size)
        self.train_mlp_chunk = int(getattr(cfg, "train_chunk", 0))   # 0 = no cap
        self.eval_mlp_chunk  = int(getattr(cfg, "eval_chunk", 0))    # you already expose --eval_chunk

        self.ckpt_mlp = bool(getattr(cfg, "ckpt_mlp", False))

        # ---- Samples per ray / density activation ----
        if self.vanilla:
            self.nc = 64
            self.nf = 128
            self.det_fine = False
            self.raw_noise_std = 1.0   # add Gaussian noise to raw sigma during training
            self.sigma_activation = "relu"
        else:
            self.nc = int(getattr(cfg, "nc", 64))
            self.nf = int(getattr(cfg, "nf", 128))
            self.det_fine = bool(getattr(cfg, "det_fine", False))
            self.raw_noise_std = float(getattr(cfg, "raw_noise_std", 0.0))
            self.sigma_activation = str(getattr(cfg, "sigma_activation", "softplus"))

        # ---- IO ----
        self.out_dir = Path(getattr(cfg, "out_dir", "./exp"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Thermal manager ----
        self.thermal = GpuThermalManager(
            enable_throttle=bool(getattr(cfg, "thermal_throttle", False)),
            temp_threshold=int(getattr(cfg, "gpu_temp_threshold", 85)),
            check_every=int(getattr(cfg, "gpu_temp_check_every", 20)),
            cooldown_seconds=int(getattr(cfg, "gpu_cooldown_seconds", 45)),
            max_micro=int(getattr(cfg, "thermal_throttle_max_micro", 16)),
            throttle_sleep=float(getattr(cfg, "thermal_throttle_sleep", 5.0)),
        )

        # ---- Signals ----
        self.signals = SignalController()
        install_signal_handlers(self.signals)

        # ---- Resume / TB ----
        self.auto_resume: bool = bool(getattr(cfg, "auto_resume", False))
        self.resume_path: Optional[str] = getattr(cfg, "resume_path", None)
        self.resume_no_optim: bool = bool(getattr(cfg, "resume_no_optim", False))
        self.on_interrupt: str = str(getattr(cfg, "on_interrupt", "render_and_exit"))
        self.on_pause_signal: str = str(getattr(cfg, "on_pause_signal", "render"))

        self.use_tb: bool = bool(getattr(cfg, "use_tb", False))
        self.tb_logdir: Optional[str] = getattr(cfg, "tb_logdir", None)
        # New TB helper (lazy init, image helpers
        self.tb_logger = TensorBoardLogger(
            enabled=self.use_tb,
            logdir=self.tb_logdir or str(self.out_dir / "tb"),
            image_max_side=int(getattr(cfg, "tb_image_max_side", 512)),
        )

        # --- ETA bookkeeping ---
        self._eta_window = deque(maxlen=20)
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._val_event_durations = []               # seconds per validation event (rolling history)
        self._val_avg_seconds = 0.0                  # running average

        print(f"[DEBUG] Trainer: white_bkgd={self.white_bkgd}  vanilla={self.vanilla}")
        print(f"[DEBUG] rays: convention={self.camera_convention} use_ndc={self.use_ndc} "
                f"near={self.near_world:.4g} far={self.far_world:.4g} ndc_near_plane={self.ndc_near_plane_world:.4g} "
                f"samp_range={[self.samp_near, self.samp_far]}")

        # ---- Validation Renderer helper ----

        # --- Validation indices: default to [0] if not provided ---
        raw_idxs = getattr(self.cfg, "val_indices", None)
        if isinstance(raw_idxs, str) and raw_idxs.strip():
            self.val_frame_indices = [int(s) for s in raw_idxs.split(",")]
        elif isinstance(raw_idxs, (list, tuple)):
            self.val_frame_indices = [int(x) for x in raw_idxs]
        else:
            self.val_frame_indices = [0]  # <— default single view

        # --- Validation renderer (uses current model state; no ckpt reloads) ---
        self.valr = ValidationRenderer(
            trainer=self,
            out_dir=self.out_dir / "validation",
            tb_logger=(self.tb_logger if getattr(self, "tb_logger", None) else None),
        )

        # --- Build validation step schedule (dense early, taper later) ---
        self.val_steps = build_validation_steps(
            max_steps=int(getattr(self.cfg, "max_steps", 200_000)),
            base_every=getattr(self.cfg, "val_every", None),
            num_val_steps=getattr(self.cfg, "num_val_steps", None),
            schedule=str(getattr(self.cfg, "val_schedule", "power")),
            power=float(getattr(self.cfg, "val_power", 2.0))
        )
        self._val_step_set = set(self.val_steps)
        self._val_next_idx = 0
        self._val_next_step = self.val_steps[0] if self.val_steps else None
        self._val_remaining_steps = len(self.val_steps)
        if self._val_next_step is not None:
            print(f"[VAL] first validation at step {self._val_next_step} (1/{len(self.val_steps)})")

        # --- Progress video plan uses the number of validation events to keep speed constant ---

        self.valr.setup_progress_plan(
            val_steps=self.val_steps,
            frames_subdir="training_progress",
        )

        # --- Run debug utilities ---
        dump_run_debug(
            self, self.scene_train, self.out_dir,
            get_camera_rays=get_camera_rays,
            nerf_forward_pass=nerf_forward_pass,
            sample_pdf=sample_pdf,
            probe_grid=8,
            probe_Nc=64,
        )

    def _debug_check_center_ray(
        self, scene, convention: str, as_ndc: bool, near_plane_world: float, device="cpu"
    ):
        """
        Prints the angle (degrees) between the center pixel's WORLD unit view direction and the
        camera-forward axis implied by `convention`. Works for both NDC and non-NDC:

        • Marching rays may be in NDC or world depending on `as_ndc`
        • Comparison always uses the WORLD unit viewdir returned by get_camera_rays
        • Also prints marching-space unit-dir norm (~1.0) and pre-norm ||d|| (scale used for Δ)
        """

        fr = scene.frames[0]
        H, W = fr.image.shape[:2]
        K    = fr.K
        c2w  = fr.c2w

        (
            rays_o_world,            # (N,3)
            rays_d_world_unit,       # (N,3)
            rays_d_world_norm,       # (N,1)
            rays_o_marching,         # (N,3)
            rays_d_marching_unit,    # (N,3)
            rays_d_marching_norm,    # (N,1)
        ) = get_camera_rays(
            image_h=H, image_w=W,
            intrinsic_matrix=K, transform_camera_to_world=c2w,
            device=device, dtype=torch.float32,
            convention=convention,
            pixel_center=True,                         # keep consistent with how K was defined
            as_ndc=as_ndc, near_plane=float(near_plane_world),
            pixels_xy=None,
        )

        # Center pixel index (map principal point to nearest pixel center index)
        cx = float(K[0, 2]); cy = float(K[1, 2])
        ix = int(np.clip(round(cx - 0.5), 0, W - 1))
        iy = int(np.clip(round(cy - 0.5), 0, H - 1))
        idx = iy * W + ix

        # WORLD forward axis implied by convention
        R = np.asarray(c2w[:3, :3])
        fwd_world = (R[:, 2] if convention in ("colmap", "opencv") else -R[:, 2])  # (3,)

        # WORLD unit viewdir at center pixel
        v_world = rays_d_world_unit[idx].detach().cpu().numpy()  # unit
        # Marching-space unit dir and pre-norm ||d|| (scale for Δ in compositor)
        v_march = rays_d_marching_unit[idx].detach().cpu().numpy()
        s_march = float(rays_d_marching_norm[idx].item())

        # Angle(center ray, forward)
        dot = float(np.clip(np.dot(v_world, fwd_world) / (np.linalg.norm(v_world) * (np.linalg.norm(fwd_world) + 1e-9)), -1.0, 1.0))
        ang_deg = float(np.degrees(np.arccos(dot)))

        print(
            f"[ray sanity] convention={convention} as_ndc={as_ndc} "
            f"center=(x={ix}, y={iy}) angle(center,fwd_world)={ang_deg:.3f}° | "
            f"||march_unit||={np.linalg.norm(v_march):.3f} ray_norm(march)={s_march:.6f}"
        )


    # ---------------- TensorBoard ----------------
    def maybe_init_tensorboard(self):
        # Backwards-compat wrapper: ensure the helper's writer exists.
        self.tb_logger._ensure_writer()

    # ---------------- Checkpoints ----------------
    @property
    def ckpt_dir(self) -> Path:
        d = self.out_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"ckpt_{step:07d}.pt"

    def save_checkpoint(self, step: int, tag: str | None = None, latest: bool = True, include_optim: bool = True) -> Path:
        path = (self.ckpt_dir / f"{tag}.pt") if tag else self._ckpt_path(step)
        obj = {
            "step": step,
            "nerf_c": self.nerf_c.state_dict(),
            "nerf_f": self.nerf_f.state_dict(),
            "opt": (self.opt.state_dict() if include_optim and hasattr(self, "opt") else None),
            "scaler": (self.scaler.state_dict() if getattr(self, "scaler", None) is not None else None),
            "sched": (self.sched.state_dict() if getattr(self, "sched", None) is not None else None),
            "cfg": getattr(self, "cfg", None).__dict__ if getattr(self, "cfg", None) is not None else {},
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        torch.save(obj, path)
        print(f"[CKPT] saved → {path}")
        if latest:
            latest_path = self.ckpt_dir / "ckpt_latest.pt"
            try:
                if latest_path.exists() or latest_path.is_symlink():
                    latest_path.unlink()
                latest_path.symlink_to(path.name)
            except Exception:
                try:
                    torch.save(obj, latest_path)
                except Exception:
                    pass
        return path

    def find_latest_checkpoint(self) -> Path | None:
        candidates = []
        candidates += list(self.ckpt_dir.glob("ckpt_*.pt"))
        candidates += list(self.out_dir.glob("ckpt_*.pt"))
        if not candidates:
            return None
        step_re = re.compile(r"(?:ckpt|step)[_-]?(\d+)", re.IGNORECASE)
        def step_of(p: Path) -> int:
            m = step_re.search(p.name); return int(m.group(1)) if m else -1
        candidates.sort(key=lambda p: (step_of(p), p.stat().st_mtime))
        return candidates[-1]

    def load_checkpoint(self, path: Path, load_optim: bool = True) -> int:
        print(f"[CKPT] loading ← {path}")
        obj = torch.load(path, map_location=self.device)
        self.nerf_c.load_state_dict(obj["nerf_c"]); self.nerf_f.load_state_dict(obj["nerf_f"])
        if load_optim and obj.get("opt") is not None and hasattr(self, "opt"):
            self.opt.load_state_dict(obj["opt"])
        if obj.get("scaler") is not None and getattr(self, "scaler", None) is not None:
            self.scaler.load_state_dict(obj["scaler"])
        if obj.get("sched") is not None and getattr(self, "sched", None) is not None:
            self.sched.load_state_dict(obj["sched"])
        return int(obj.get("step", 0))

    def save_preview_bundle(self, step: int, val: dict[str, torch.Tensor], prefix: str = "step") -> None:
        name = f"{prefix}_{step:07d}"
        pdir = (self.out_dir / "preview"); pdir.mkdir(parents=True, exist_ok=True)
        rgb_path   = pdir / f"{name}.png"
        acc_path   = pdir / f"acc_{step:07d}.png"
        depth_path = pdir / f"depth_{step:07d}.png"
        rgb_srgb = val["rgb"].clamp(0, 1)
        acc_img = val["acc"].squeeze(-1).clamp(0, 1)
        depth_lin = val["depth"]
        if self.use_ndc:
            depth_norm = depth_lin.squeeze(-1).clamp(0, 1)
        else:
            depth_norm = ((depth_lin - self.near_world) / (self.far_world - self.near_world + 1e-8)).squeeze(-1).clamp(0, 1)
        save_rgb_png(rgb_srgb, rgb_path)
        save_gray_png(acc_img, acc_path)
        save_gray_png(depth_norm, depth_path)

    # ---------------- Train loop ----------------
    def train(self):
        interrupted = False
        it = iter(self.sampler)
        self.nerf_c.train(); self.nerf_f.train()

        # ---- Resume logic ----
        start_step = 1
        resume_from = Path(self.resume_path) if getattr(self, "resume_path", None) else None
        if resume_from is None and self.auto_resume:
            resume_from = self.find_latest_checkpoint()
        if resume_from and resume_from.exists():
            start_step = self.load_checkpoint(resume_from, load_optim=(not self.resume_no_optim)) + 1
            print(f"[CKPT] Resuming from step {start_step} ({resume_from.name})")
            if getattr(self.cfg, "progress_video_during_training", False):
                # We set up the progress plan in __init__; now align it to the resumed step.
                self.valr.resume_to_step(start_step - 1)
                done = sum(self.valr._prog_block_sizes[: self.valr._prog_next_block_idx])
                print(f"[RESUME] progress frames already on disk: {done}/{self.valr._prog_total_frames} "
                    f"(next block idx = {self.valr._prog_next_block_idx})")

        # --- Fast-forward validation schedule to match the resume step ---
        # We want the first _val_next_step that is >= start_step
        i = 0
        while i < len(self.val_steps) and self.val_steps[i] < start_step:
            i += 1
        self._val_next_idx = i
        self._val_next_step = self.val_steps[i] if i < len(self.val_steps) else None

        if self._val_next_step is not None:
            print(f"[VAL] next validation at step {self._val_next_step} ({i+1}/{len(self.val_steps)})")
        else:
            print("[VAL] validation schedule already completed at resume.")

        max_steps  = int(getattr(self.cfg, "max_steps", 200_000))
        log_every  = int(getattr(self.cfg, "log_every", 100))
        ckpt_every = int(getattr(self.cfg, "ckpt_every", 5000))

        for step in range(start_step, max_steps + 1):
            self.global_step = step
            batch = next(it)  # {"rays_o","rays_d","rgb","viewdirs"}
            self.opt.zero_grad(set_to_none=True)

            # Forward + loss
            if self.train_micro_chunks > 0:
                loss_val, psnr_val = self._train_step_chunked(batch, self.train_micro_chunks)
            else:
                out = self._train_step(batch)
                loss = out["loss"]
                if not torch.isfinite(loss):
                    print(f"[WARN] Non-finite loss at step {step}: {float(loss.detach().cpu())}. Skipping.")
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update(); continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                clip = float(getattr(self.cfg, "grad_clip_norm", 0.0) or 0.0)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()), max_norm=clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                if self.sched is not None:
                    self.sched.step()
                if self.global_step in {0, 1, 1000, (self.cfg.max_steps//2), self.cfg.max_steps-1}:
                    print(f"[LR] step={self.global_step} lr={[pg['lr'] for pg in self.opt.param_groups]}")
                loss_val = float(loss.detach().cpu())
                psnr_val = float(out["psnr"].item())

            # TB + Thermal
            self.tb_logger.add_scalar("train/loss", loss_val, step)
            self.tb_logger.add_scalar("train/psnr", float(psnr_val), step)
            self.tb_logger.add_scalar("train/lr", float(self.opt.param_groups[0]["lr"]), step)
            # Thermal guard: pass a real SummaryWriter from the helper
            if self.tb_logger._ensure_writer():
                self.thermal.log_to_tb(self.tb_logger.writer, step)

            # Signals
            if self.signals.sigusr1:
                try: self.save_checkpoint(step, tag=f"pause_step_{step}")
                except Exception as e: print(f"[PAUSE] checkpoint save failed: {e}")
                self.signals.sigusr1 = False

            if self.signals.sigint:
                try:
                    self.save_checkpoint(step, tag=f"interrupt_step_{step}")
                except Exception as e:
                    print(f"[INT] checkpoint save failed: {e}")
                print("[INT] Exiting training loop.")
                interrupted = True
                break

            # Logs
            if step % log_every == 0:
                now = time.time()
                secs = now - self._last_log_time
                steps_delta = step - (self._last_log_step or 0) or 1
                sec_per_step = secs / steps_delta
                self._eta_window.append(sec_per_step)
                avg_sec_per_step = sum(self._eta_window) / len(self._eta_window)

                # TRAIN-ONLY ETA
                max_steps = int(getattr(self.cfg, "max_steps", 0) or 0)
                steps_left = max(0, max_steps - step)
                eta_train_sec = steps_left * avg_sec_per_step

                # VALIDATION ETA (remaining events × avg seconds per event)
                rem_val_events = self._val_remaining_steps
                avg_val_sec = float(self._val_avg_seconds or 0.0)
                eta_val_sec = rem_val_events * avg_val_sec

                # TOTAL ETA = training + future validation
                eta_total_sec = eta_train_sec + eta_val_sec

                def _fmt_eta(seconds: float) -> str:
                    h = int(seconds // 3600)
                    m = int((seconds % 3600) // 60)
                    s = int(seconds % 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"

                print(
                    f"[{step:7d}] loss={loss_val:.6f} psnr={psnr_val:.2f} "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} "
                    f"({secs:.2f}s/{steps_delta} steps, {sec_per_step:.3f}s/step avg {avg_sec_per_step:.3f}s) "
                    f"ETA(train) {_fmt_eta(eta_train_sec)} | ETA(total) {_fmt_eta(eta_total_sec)}"
                )

            # Validation (dense-early schedule)
            if self._val_next_step is not None and step == self._val_next_step:
                val_t0 = time.perf_counter()
                self.nerf_c.eval(); self.nerf_f.eval()
                with torch.no_grad():
                    self.valr.tb = (self.tb_logger if getattr(self, "tb_logger", None) else None)
                    # Let the renderer know our eval chunking preferences (no-op if it ignores them)
                    setattr(self.valr, "eval_micro_chunks", self.eval_micro_chunks)
                    setattr(self.valr, "eval_chunk",       self.eval_mlp_chunk)
                    paths, val_psnr_metrics = self.valr.render_indices_at_step(
                        step=step,
                        frame_indices=self.val_frame_indices,   # already defaulted to [0]
                        use_mask="auto",
                        res_scale=float(getattr(self.cfg, "val_res_scale", 1.0)),
                        log_to_tb=bool(getattr(self.cfg, "use_tb", False)),
                    )
                    if val_psnr_metrics.get("psnr_mean") is not None:
                        mean_psnr = val_psnr_metrics["psnr_mean"]
                        print(f"[VAL] step={self.global_step} mean PSNR={mean_psnr:.2f} dB over {len(self.val_frame_indices)} frames")
                    if bool(getattr(self.cfg, "progress_video_during_training", False)):
                        start_idx, count = self.valr.render_progress_block()
                        if count > 0:
                            print(f"[PROGRESS] wrote progress frames [{start_idx}..{start_idx+count-1}]")
                self.nerf_c.train(); self.nerf_f.train()

                # Measure and update rolling validation-time average
                val_secs = max(0.0, time.perf_counter() - val_t0)
                self._val_event_durations.append(val_secs)
                # Use last K to smooth; K=10 is a nice default
                recent_val_event_durations = self._val_event_durations[-10:]
                self._val_avg_seconds = float(sum(recent_val_event_durations) / max(1, len(recent_val_event_durations)))
                # advance to the next scheduled validation step
                self._val_next_idx += 1
                self._val_next_step = self.val_steps[self._val_next_idx] if self._val_next_idx < len(self.val_steps) else None
                self._val_remaining_steps = len(self.val_steps[self._val_next_idx:])
                if self._val_next_step is not None:
                    print(f"[VAL] next validation at step {self._val_next_step} "
                        f"({self._val_next_idx+1}/{len(self.val_steps)})")
                else:
                    print("[VAL] schedule complete.")


                self.nerf_c.train(); self.nerf_f.train()

            # Checkpoints
            if step % ckpt_every == 0:
                self.save_checkpoint(step)

        # If we were interrupted, close TB and exit immediately (no rendering).
        if interrupted:
            self.tb_logger.close()
            return

        # ---------- Post-training exports (ValidationRenderer owns everything) ----------
        # A) Per-index time-lapse videos (RGB / depth / opacity)
        try:
            self.valr.export_val_videos_for_indices(
                frame_indices=self.val_frame_indices,
                fps=int(getattr(self.cfg, "path_fps", 24)),
                out_suffix="",
            )
        except Exception as e:
            print(f"[VAL-VIDEO] export failed: {e}")

        # B) Final camera-path video with the finished model (reuse the SAME poses)
        if bool(getattr(self.cfg, "render_path_after", False)):
            try:
                print("[CAMERA PATH] Rendering final camera-path video...")
                self.nerf_c.eval(); self.nerf_f.eval()
                self.valr.render_camera_path_video(
                    video_name="camera_path", overwrite=True
                )
            except Exception as e:
                print(f"[CAMERA PATH] Failed: {e}")

        # C) Finalize the training-progress video (if we were writing it during training)
        if bool(getattr(self.cfg, "progress_video_during_training", False)):
            try:
                self.valr.export_progress_video(video_name="training_progress")
            except Exception as e:
                print(f"[PROGRESS] Video assembly failed: {e}")

        # Close helper-managed writer
        self.tb_logger.close()

    # ---------------- Internal training steps ----------------

    def _train_step(self, batch) -> dict[str, torch.Tensor]:
        # ----------------------------
        # Unpack the batch
        # ----------------------------
        rays_o_marching         = batch["rays_o_marching"]        # (B, 3)
        rays_d_marching_unit    = batch["rays_d_marching_unit"]   # (B, 3)  unit dirs in marching space
        rays_d_marching_norm    = batch["rays_d_marching_norm"]   # (B, 1)  ||d_raw|| in marching space
        rays_d_world_unit       = batch["rays_d_world_unit"]      # (B, 3)  unit WORLD/CAMERA dirs (for MLP)
        target                  = batch["rgb"]                    # (B, 3)  sRGB/linear per your loader

        if self.use_ndc:
            rn = batch["rays_d_marching_norm"]
            if self.global_step == 100:
                print(f"[sanity] NDC march: ray_norms min/max {rn.min().item():.6f} / {rn.max().item():.6f}")
        else:
            rn = batch["rays_d_marching_norm"]
            if self.global_step == 100:
                print(f"[sanity] WORLD march: ray_norms min/max {rn.min().item():.6f} / {rn.max().item():.6f}")


        B     = rays_o_marching.shape[0]
        dtype = rays_o_marching.dtype

        with torch.amp.autocast(self.device.type, enabled=self.amp):
            # --- Coarse stratified samples with jitter (training) ---
            t  = torch.linspace(0.0, 1.0, steps=self.nc, device=self.device, dtype=dtype)
            zc = (self.samp_near * (1.0 - t) + self.samp_far * t).expand(B, self.nc).contiguous()

            mids  = 0.5 * (zc[:, 1:] + zc[:, :-1])
            lower = torch.cat([zc[:, :1], mids], dim=-1)
            upper = torch.cat([mids, zc[:, -1:]], dim=-1)
            zc    = lower + (upper - lower) * torch.rand_like(zc)  # perturb=True
            zc    = torch.sort(zc, dim=-1).values

            # ----- Coarse forward (train-only σ-noise) -----
            comp_c, w_c, _, _ = nerf_forward_pass(
                rays_o=rays_o_marching,
                rays_d_unit=rays_d_marching_unit,                   # unit marching dirs
                z_vals=zc,
                pos_enc=self.pos_enc, dir_enc=self.dir_enc, nerf=self.nerf_c,
                white_bkgd=self.white_bkgd,
                ray_norms=rays_d_marching_norm,                     # scale Δz in marching space
                viewdirs_world_unit=rays_d_world_unit,           # WORLD unit dirs for MLP
                sigma_activation=self.sigma_activation,
                raw_noise_std=float(self.raw_noise_std), training=True,
                mlp_chunk=int(getattr(self, "train_mlp_chunk", 0)),
                infinite_last_bin=self.infinite_last_bin,
            )

            # --- Fine sampling via PDF using bin midpoints & interval weights ---
            bins_mid     = 0.5 * (zc[:, 1:] + zc[:, :-1])             # (B, Nc-1)
            weights_bins = 0.5 * (w_c[:, 1:] + w_c[:, :-1])           # (B, Nc-1)
            weights_bins = (weights_bins.detach() + 1e-5)

            zf = sample_pdf(
                bins_mid, weights_bins,
                n_samples=self.nf,
                deterministic=self.det_fine
            )  # (B, Nf)

            # (optional) diagnostic: % fine samples that land in top-k intervals
            with torch.no_grad():
                if (self.global_step % 500) == 0:
                    # Re-derive bins_mid from the exact zc we used
                    bins_mid = 0.5 * (zc[:, 1:] + zc[:, :-1])             # (B, Nc-1)

                    # Make sure coarse weights align with zc along the last axis.
                    # We expect w_c.shape[-1] == zc.shape[-1] (Nc). If off-by-one, pad/trim once.
                    if w_c.shape[-1] == zc.shape[-1] - 1:
                        # Missing the very last weight — pad a zero once
                        w_c = torch.nn.functional.pad(w_c, (0, 1))
                    elif w_c.shape[-1] == zc.shape[-1] + 1:
                        # Has one extra trailing weight — drop once
                        w_c = w_c[..., :-1]

                    # Average neighboring weights to get one weight per interval (match bins_mid)
                    weights_bins = 0.5 * (w_c[:, 1:] + w_c[:, :-1]).detach()

                    # If anything is still off, hard-align by slicing to the common M
                    M = min(bins_mid.shape[-1], weights_bins.shape[-1])
                    if bins_mid.shape[-1] != weights_bins.shape[-1]:
                        print(f"[diag] aligning bins: bins_mid={bins_mid.shape[-1]} weights_bins={weights_bins.shape[-1]} → M={M}")
                        bins_mid     = bins_mid[:, :M].contiguous()
                        weights_bins = weights_bins[:, :M].contiguous()

                    # Normalize for top-k inspection
                    wb    = weights_bins / (weights_bins.sum(-1, keepdim=True) + 1e-9)
                    top_i = wb.topk(4, dim=-1).indices

                    # Keep z_fine contiguous to avoid the searchsorted perf warning
                    zf = zf.contiguous()
                    idx   = torch.clamp(torch.searchsorted(bins_mid, zf) - 1, 0, bins_mid.shape[-1] - 1)
                    hit   = (idx.unsqueeze(-1) == top_i.unsqueeze(1)).any(-1).float()
                    print(f"[diag] fine samples landing in top-4 intervals: {hit.mean().item()*100:.1f}%")

                    debug_topk_fine_hit(
                        step=self.global_step,
                        bins_mid=bins_mid,
                        weights_bins=weights_bins,
                        z_fine=zf,
                        topk=4,
                        logger=print,
                        tag="coarse→fine",
                    )

            z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

            # ----- Fine forward -----
            comp_f, _, _, _ = nerf_forward_pass(
                rays_o=rays_o_marching,
                rays_d_unit=rays_d_marching_unit,                   # unit marching dirs
                z_vals=z_all,
                pos_enc=self.pos_enc, dir_enc=self.dir_enc, nerf=self.nerf_f,
                white_bkgd=self.white_bkgd,
                ray_norms=rays_d_marching_norm,                     # (B,1)
                viewdirs_world_unit=rays_d_world_unit,
                sigma_activation=self.sigma_activation,
                raw_noise_std=float(self.raw_noise_std), training=True,
                mlp_chunk=int(getattr(self, "train_mlp_chunk", 0)),
                infinite_last_bin=self.infinite_last_bin,
            )

            # Numeric guards + clamp
            comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            loss_c = F.mse_loss(comp_c, target)
            loss_f = F.mse_loss(comp_f, target)
            loss   = loss_c + loss_f
            psnr   = mse2psnr(loss_f.detach())

        return {
            "loss": loss,
            "psnr": psnr,
            "comp_f": comp_f.detach(),
            "comp_c": comp_c.detach(),
        }

    def _train_step_chunked(self, batch, micro_chunks: int) -> tuple[float, float]:
        # ----------------------------
        # Unpack your verbose batch API
        # ----------------------------
        rays_o_marching         = batch["rays_o_marching"]         # (B,3)
        rays_d_marching_unit    = batch["rays_d_marching_unit"]    # (B,3)  unit dirs in marching space
        rays_d_marching_norm    = batch["rays_d_marching_norm"]    # (B,1)  ||d_raw|| in marching space
        rays_d_world_unit       = batch["rays_d_world_unit"]       # (B,3)  unit WORLD/CAMERA dirs (for MLP)
        target                  = batch["rgb"]                     # (B,3)

        if self.use_ndc:
            rn = batch["rays_d_marching_norm"]
            if self.global_step == 100:
                print(f"[sanity] NDC march: ray_norms min/max {rn.min().item():.6f} / {rn.max().item():.6f}")
        else:
            rn = batch["rays_d_marching_norm"]
            if self.global_step == 100:
                print(f"[sanity] WORLD march: ray_norms min/max {rn.min().item():.6f} / {rn.max().item():.6f}")

        B = rays_o_marching.shape[0]
        m = max(1, int(micro_chunks))
        chunk = (B + m - 1) // m

        total_loss = 0.0
        last_psnr = 0.0

        for i in range(0, B, chunk):
            ro  = rays_o_marching[i:i+chunk]
            rd  = rays_d_marching_unit[i:i+chunk]            # unit marching dirs
            rn  = rays_d_marching_norm[i:i+chunk]            # (Bi,1) metric scale in marching space
            vd  = rays_d_world_unit[i:i+chunk]            # unit WORLD/CAMERA dirs for MLP
            tgt_slice = target[i:i+chunk]

            Bi    = ro.shape[0]
            dtype = ro.dtype

            with torch.amp.autocast(self.device.type, enabled=self.amp):
                # --- Coarse stratified samples with jitter (training) ---
                t  = torch.linspace(0.0, 1.0, steps=self.nc, device=self.device, dtype=dtype)
                zc = (self.samp_near * (1.0 - t) + self.samp_far * t).expand(Bi, self.nc).contiguous()

                mids  = 0.5 * (zc[:, 1:] + zc[:, :-1])
                lower = torch.cat([zc[:, :1], mids], dim=-1)
                upper = torch.cat([mids, zc[:, -1:]], dim=-1)
                zc    = lower + (upper - lower) * torch.rand_like(zc)  # jitter
                zc    = torch.sort(zc, dim=-1).values

                # ----- Coarse forward (uses unit dirs + ray_norms) -----
                comp_c, w_c, _, _ = nerf_forward_pass(
                    rays_o=ro,
                    rays_d_unit=rd,                       # unit marching dirs
                    z_vals=zc,
                    pos_enc=self.pos_enc, dir_enc=self.dir_enc, nerf=self.nerf_c,
                    white_bkgd=self.white_bkgd,
                    ray_norms=rn,                         # (Bi,1) scale Δz in marching space
                    viewdirs_world_unit=vd,               # WORLD unit dirs for MLP
                    sigma_activation=self.sigma_activation,
                    raw_noise_std=float(self.raw_noise_std), training=True,
                    mlp_chunk=int(getattr(self, "train_mlp_chunk", 0)),
                    infinite_last_bin=self.infinite_last_bin,
                )

                # --- Fine sampling via PDF using bin midpoints & interval weights ---
                bins_mid     = 0.5 * (zc[:, 1:] + zc[:, :-1])         # (Bi, Nc-1)
                weights_bins = 0.5 * (w_c[:, 1:] + w_c[:, :-1])       # (Bi, Nc-1)
                weights_bins = (weights_bins.detach() + 1e-5)

                zf = sample_pdf(
                    bins_mid, weights_bins,
                    n_samples=self.nf,
                    deterministic=self.det_fine
                )  # (Bi, Nf)

                # Optional diag: % fine samples that land in top-k intervals
                with torch.no_grad():
                    if (self.global_step % 500) == 0:
                        # Re-derive bins_mid from the exact zc we used
                        bins_mid = 0.5 * (zc[:, 1:] + zc[:, :-1])             # (B, Nc-1)

                        # Make sure coarse weights align with zc along the last axis.
                        # We expect w_c.shape[-1] == zc.shape[-1] (Nc). If off-by-one, pad/trim once.
                        if w_c.shape[-1] == zc.shape[-1] - 1:
                            # Missing the very last weight — pad a zero once
                            w_c = torch.nn.functional.pad(w_c, (0, 1))
                        elif w_c.shape[-1] == zc.shape[-1] + 1:
                            # Has one extra trailing weight — drop once
                            w_c = w_c[..., :-1]

                        # Average neighboring weights to get one weight per interval (match bins_mid)
                        weights_bins = 0.5 * (w_c[:, 1:] + w_c[:, :-1]).detach()

                        # If anything is still off, hard-align by slicing to the common M
                        M = min(bins_mid.shape[-1], weights_bins.shape[-1])
                        if bins_mid.shape[-1] != weights_bins.shape[-1]:
                            print(f"[diag] aligning bins: bins_mid={bins_mid.shape[-1]} weights_bins={weights_bins.shape[-1]} → M={M}")
                            bins_mid     = bins_mid[:, :M].contiguous()
                            weights_bins = weights_bins[:, :M].contiguous()

                        # Normalize for top-k inspection
                        wb    = weights_bins / (weights_bins.sum(-1, keepdim=True) + 1e-9)
                        top_i = wb.topk(4, dim=-1).indices

                        # Keep z_fine contiguous to avoid the searchsorted perf warning
                        zf = zf.contiguous()
                        idx   = torch.clamp(torch.searchsorted(bins_mid, zf) - 1, 0, bins_mid.shape[-1] - 1)
                        hit   = (idx.unsqueeze(-1) == top_i.unsqueeze(1)).any(-1).float()
                        print(f"[diag] fine samples landing in top-4 intervals: {hit.mean().item()*100:.1f}%")

                        debug_topk_fine_hit(
                            step=self.global_step,
                            bins_mid=bins_mid,
                            weights_bins=weights_bins,
                            z_fine=zf,
                            topk=4,
                            logger=print,
                            tag="coarse→fine",
                        )

                z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

                # ----- Fine forward -----
                comp_f, _, _, _ = nerf_forward_pass(
                    rays_o=ro,
                    rays_d_unit=rd,                       # unit marching dirs
                    z_vals=z_all,
                    pos_enc=self.pos_enc, dir_enc=self.dir_enc, nerf=self.nerf_f,
                    white_bkgd=self.white_bkgd,
                    ray_norms=rn,                         # (Bi,1)
                    viewdirs_world_unit=vd,
                    sigma_activation=self.sigma_activation,
                    raw_noise_std=float(self.raw_noise_std), training=True,
                    mlp_chunk=int(getattr(self, "train_mlp_chunk", 0)),
                    infinite_last_bin=self.infinite_last_bin,
                )

                # Targets + numeric guards
                tgt = tgt_slice.to(device=self.device, dtype=comp_f.dtype, non_blocking=True)
                comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                tgt    = torch.nan_to_num(tgt,    nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                # Grad accumulation (scale by 1/m)
                loss = (F.mse_loss(comp_c, tgt) + F.mse_loss(comp_f, tgt)) / m
                last_psnr = float(mse2psnr(F.mse_loss(comp_f, tgt).detach()).item())

            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())

        # Step optimizer & scheduler
        self.scaler.unscale_(self.opt)
        clip = float(getattr(self.cfg, "grad_clip_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
                max_norm=clip
            )
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.sched is not None:
            self.sched.step()
        if self.global_step in {0, 1, 1000, (self.cfg.max_steps//2), self.cfg.max_steps-1}:
            print(f"[LR] step={self.global_step} lr={[pg['lr'] for pg in self.opt.param_groups]}")

        return total_loss, last_psnr
