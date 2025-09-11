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
from nerf_experiments.source.data.loaders.blender_loader import BlenderSceneLoader
from nerf_experiments.source.data.samplers import RandomPixelRaySampler
from nerf_experiments.source.models.encoders import PositionalEncoder, get_vanilla_nerf_encoders
from nerf_experiments.source.models.mlps import NeRF
from nerf_experiments.source.utils.render_utils import (
    volume_render_rays,
    srgb_to_linear, linear_to_srgb,
    save_rgb_png, save_gray_png,
    render_image_chunked, render_pose
)
from nerf_experiments.source.utils.ray_utils import get_camera_rays
from nerf_experiments.source.utils.sampling_utils import sample_pdf
from nerf_experiments.source.utils.gpu_thermal import GpuThermalManager
from nerf_experiments.source.utils.signal_handlers import SignalController, install_signal_handlers
from nerf_experiments.source.utils.tensorboard_utils import TensorBoardLogger
from nerf_experiments.source.utils.validation_renderer import ValidationRenderer
from nerf_experiments.source.utils.validation_schedule import build_validation_steps


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
            downscale, scene_scale, center_origin, composite_on_load
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
        self.device = torch.device(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.vanilla: bool = bool(getattr(cfg, "vanilla", False))

        set_global_seed(int(getattr(cfg, "seed", 42)))

        # ---- Data ----
        self.loader = BlenderSceneLoader(
            root=cfg.data_root,
            downscale=int(getattr(cfg, "downscale", 1)),
            white_bkgd=bool(getattr(cfg, "white_bkgd", True)),
            scene_scale=float(getattr(cfg, "scene_scale", 1.0)),
            center_origin=bool(getattr(cfg, "center_origin", False)),
            composite_on_load=bool(getattr(cfg, "composite_on_load", True)),
        )
        self.train_scene = self.loader.load(getattr(cfg, "split", "train"))
        try:
            self.val_scene = self.loader.load("val")
        except FileNotFoundError:
            try:
                self.val_scene = self.loader.load("test")
            except FileNotFoundError:
                self.val_scene = self.train_scene

        # Near/Far (Blender synthetic commonly ~[2.0, 6.0])
        self.near = float(getattr(cfg, "near", 2.0))
        self.far  = float(getattr(cfg, "far",  6.0))

        # ---- Sampler ----
        if self.vanilla:
            # Vanilla NeRF: single-frame batches + center precrop
            self.sampler = RandomPixelRaySampler(
                self.train_scene,
                rays_per_batch=1024,  # N_rand
                device=self.device,
                white_bg_composite=bool(getattr(cfg, "white_bkgd", True)),
                sample_from_single_frame=True,
                precrop_iters=int(getattr(cfg, "precrop_iters", 500)),
                precrop_frac=float(getattr(cfg, "precrop_frac", 0.5)),
            )
        else:
            # Mixed-frames batches (default)
            self.sampler = RandomPixelRaySampler(
                self.train_scene,
                rays_per_batch=int(getattr(cfg, "rays_per_batch", 2048)),
                device=self.device,
                white_bg_composite=bool(getattr(cfg, "white_bkgd", True)),
                sample_from_single_frame=bool(getattr(cfg, "sample_from_single_frame", False)),
                precrop_iters=int(getattr(cfg, "precrop_iters", 0)),
                precrop_frac=float(getattr(cfg, "precrop_frac", 0.5)),
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
                n_layers=8, hidden_dim=256, skip_pos=5,
                near=self.near, far=self.far,
                initial_acc_opacity=None,    # vanilla doesn't set sigma bias init
                sigma_activation="relu"      # only relevant if initial_acc_opacity is used
            ).to(self.device)
            self.nerf_f = NeRF(
                enc_pos_dim=self.pos_enc.out_dim, enc_dir_dim=self.dir_enc.out_dim,
                n_layers=8, hidden_dim=256, skip_pos=5,
                near=self.near, far=self.far,
                initial_acc_opacity=None,
                sigma_activation="relu"
            ).to(self.device)
        else:
            self.nerf_c = NeRF(
                enc_pos_dim=self.pos_enc.out_dim,
                enc_dir_dim=self.dir_enc.out_dim,
                n_layers=int(getattr(cfg, "n_layers", 8)),
                hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
                skip_pos=int(getattr(cfg, "skip_pos", 5)),
                near=self.near,
                far=self.far,
                initial_acc_opacity=float(getattr(cfg, "initial_acc_opacity", 0.0)),
                sigma_activation=str(getattr(cfg, "sigma_activation", "softplus"))
            ).to(self.device)
            self.nerf_f = NeRF(
                enc_pos_dim=self.pos_enc.out_dim,
                enc_dir_dim=self.dir_enc.out_dim,
                n_layers=int(getattr(cfg, "n_layers", 8)),
                hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
                skip_pos=int(getattr(cfg, "skip_pos", 5)),
                near=self.near,
                far=self.far,
                initial_acc_opacity=float(getattr(cfg, "initial_acc_opacity", 0.0)),
                sigma_activation=str(getattr(cfg, "sigma_activation", "softplus"))
            ).to(self.device)

        # ---- Optimizer + Scheduler ----
        self.opt = optim.Adam(
            list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
            lr=float(getattr(cfg, "lr", 5e-4))
        )
        self.sched = make_scheduler(self.opt, getattr(cfg, "scheduler", "none"),
                                    getattr(cfg, "scheduler_params", {}))

        # ---- Runtime toggles ----
        self.amp = bool(getattr(cfg, "amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.micro_chunks = int(getattr(cfg, "micro_chunks", 0))
        self.ckpt_mlp = bool(getattr(cfg, "ckpt_mlp", False))
        self.targets_are_srgb: bool = bool(getattr(cfg, "targets_are_srgb", True))
        # For strict vanilla parity, training targets should be used as-is (no linearization).
        if self.vanilla and not hasattr(cfg, "targets_are_srgb"):
            self.targets_are_srgb = False

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

        self._eta_window = deque(maxlen=20)
        self._last_log_time = time.time()
        self._last_log_step = 0

        print(f"[DEBUG] Trainer: white_bkgd={bool(getattr(cfg,'white_bkgd',True))}  vanilla={self.vanilla}")

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
            base_every=int(getattr(self.cfg, "val_every", 1000)),
            events=getattr(self.cfg, "val_events", None),
            strategy=str(getattr(self.cfg, "val_schedule", "power")),
            power=float(getattr(self.cfg, "val_power", 2.0)),
            first_step=getattr(self.cfg, "val_first_step", None),
            last_step=int(getattr(self.cfg, "max_steps", 200_000)),
        )
        self._val_next_idx = 0
        self._val_next_step = self.val_steps[0] if self.val_steps else None

        # --- Progress video plan uses the number of validation events to keep speed constant ---
        self.valr.setup_progress_plan(
            max_steps=int(getattr(self.cfg, "max_steps", 200_000)),
            val_every=int(getattr(self.cfg, "val_every", 1000)),
            n_frames=int(getattr(self.cfg, "progress_frames", 240)),
            num_events=len(self.val_steps),  # <— use event count from schedule
            path_type=str(getattr(self.cfg, "path_type", "circle")),
            radius=getattr(self.cfg, "path_radius", None),
            elevation_deg=float(getattr(self.cfg, "path_elev_deg", 10.0)),
            yaw_range_deg=float(getattr(self.cfg, "path_yaw_deg", 360.0)),
            res_scale=float(getattr(self.cfg, "path_res_scale", 1.0)),
            fps=int(getattr(self.cfg, "path_fps", 24)),
            world_up=None,
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
        rgb_srgb = linear_to_srgb(val["rgb"])
        acc_img = val["acc"].squeeze(-1).clamp(0, 1)
        depth_lin = val["depth"]
        depth_norm = ((depth_lin - self.near) / (self.far - self.near + 1e-8)).squeeze(-1).clamp(0, 1)
        save_rgb_png(rgb_srgb, rgb_path)
        save_gray_png(acc_img, acc_path)
        save_gray_png(depth_norm, depth_path)

    # ---------------- Train loop ----------------
    def train(self):
        it = iter(self.sampler)
        self.nerf_c.train(); self.nerf_f.train()

        # Resume
        start_step = 1
        resume_from = Path(self.resume_path) if getattr(self, "resume_path", None) else None
        if resume_from is None and self.auto_resume:
            resume_from = self.find_latest_checkpoint()
        if resume_from and resume_from.exists():
            start_step = self.load_checkpoint(resume_from, load_optim=(not self.resume_no_optim)) + 1
            print(f"[CKPT] Resuming from step {start_step} ({resume_from.name})")

        max_steps  = int(getattr(self.cfg, "max_steps", 200_000))
        log_every  = int(getattr(self.cfg, "log_every", 100))
        val_every  = int(getattr(self.cfg, "val_every", 1000))
        ckpt_every = int(getattr(self.cfg, "ckpt_every", 5000))

        for step in range(start_step, max_steps + 1):
            batch = next(it)  # {"rays_o","rays_d","rgb"}
            self.opt.zero_grad(set_to_none=True)

            # Forward + loss
            if self.micro_chunks > 0:
                loss_val, psnr_val = self._train_step_chunked(batch, self.micro_chunks)
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
                if self.sched is not None: self.sched.step()
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
                try:
                    # Optional convenience while paused:
                    # 1) assemble per-index time-lapse videos from what we have so far
                    self.valr.export_val_videos_for_indices(
                        frame_indices=self.val_frame_indices,
                        fps=int(getattr(self.cfg, "path_fps", 24)),
                        out_suffix=f"_upto_{step:07d}",
                    )
                    # 2) (optional) assemble a partial progress video from whatever frames exist
                    #    comment out if you prefer to only finalize at the very end.
                    # self.valr.finalize_progress_video(video_name=f"training_progress_upto_{step:07d}.mp4")
                except Exception as e:
                    print(f"[PAUSE] export failed: {e}")
                self.signals.sigusr1 = False

            if self.signals.sigint:
                try:
                    self.save_checkpoint(step, tag=f"interrupt_step_{step}")
                except Exception as e:
                    print(f"[INT] checkpoint save failed: {e}")
                print("[INT] Exiting training loop.")
                break

            # Logs
            if step % log_every == 0:
                now = time.time(); secs = now - self._last_log_time
                steps_delta = step - (self._last_log_step or 0) or 1
                sec_per_step = secs / steps_delta
                self._eta_window.append(sec_per_step)
                avg_sec_per_step = sum(self._eta_window) / len(self._eta_window)
                steps_left = max(0, max_steps - step)
                eta_sec = steps_left * avg_sec_per_step
                eta_h = int(eta_sec // 3600); eta_m = int((eta_sec % 3600) // 60); eta_s = int(eta_sec % 60)
                eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                print(f"[{step:>7d}] loss={loss_val:.6f} psnr={psnr_val:.2f} lr={self.opt.param_groups[0]['lr']:.2e} "
                      f"({secs:.2f}s/{steps_delta} steps, {sec_per_step:.3f}s/step avg {avg_sec_per_step:.3f}s) ETA {eta_str}")

            # Validation (dense-early schedule)
            if self._val_next_step is not None and step == self._val_next_step:
                self.nerf_c.eval(); self.nerf_f.eval()
                with torch.no_grad():
                    self.valr.tb = (self.tb_logger if getattr(self, "tb_logger", None) else None)
                    self.valr.render_indices_at_step(
                        step=step,
                        frame_indices=self.val_frame_indices,   # already defaulted to [0]
                        res_scale=float(getattr(self.cfg, "val_res_scale", 1.0)),
                        log_to_tb=bool(getattr(self.cfg, "use_tb", False)),
                    )
                    if bool(getattr(self.cfg, "progress_video_during_training", False)):
                        start_idx, count = self.valr.render_progress_block()
                        if count > 0:
                            print(f"[PROGRESS] wrote progress frames [{start_idx}..{start_idx+count-1}]")
                self.nerf_c.train(); self.nerf_f.train()

                # advance to the next scheduled validation step
                self._val_next_idx += 1
                self._val_next_step = self.val_steps[self._val_next_idx] \
                    if self._val_next_idx < len(self.val_steps) else None


                self.nerf_c.train(); self.nerf_f.train()

            # Checkpoints
            if step % ckpt_every == 0:
                self.save_checkpoint(step)

        # ---------- Post-training exports (ValidationRenderer owns everything) ----------
        # A) Per-index time-lapse videos (RGB / depth / opacity)
        try:
            vids = self.valr.export_val_videos_for_indices(
                frame_indices=self.val_frame_indices,
                fps=int(getattr(self.cfg, "path_fps", 24)),
                out_suffix="",  # customize if desired
            )
            for v in vids:
                print(f"[VAL-VIDEO] wrote → {v}")
        except Exception as e:
            print(f"[VAL-VIDEO] export failed: {e}")

        # B) Final camera-path video with the finished model (reuse the SAME poses)
        if bool(getattr(self.cfg, "render_path_after", False)):
            try:
                print("[RENDER PATH] Rendering final camera-path video...")
                self.nerf_c.eval(); self.nerf_f.eval()
                mp4 = self.valr.render_final_path_video(
                    video_name="camera_path.mp4", overwrite=True
                )
                if mp4:
                    print(f"[RENDER PATH] wrote → {mp4}")
            except Exception as e:
                print(f"[RENDER PATH] Failed: {e}")

        # C) Finalize the training-progress video (if we were writing it during training)
        if bool(getattr(self.cfg, "progress_video_during_training", False)):
            try:
                mp4 = self.valr.finalize_progress_video(video_name="training_progress.mp4")
                if mp4:
                    print(f"[PROGRESS] assembled → {mp4}")
            except Exception as e:
                print(f"[PROGRESS] MP4 assembly failed: {e}")

        # Close helper-managed writer
        self.tb_logger.close()

    # ---------------- Internal training steps ----------------

    def _train_step(self, batch) -> dict[str, torch.Tensor]:
        rays_o, rays_d, target = batch["rays_o"], batch["rays_d"], batch["rgb"]
        target_lin = srgb_to_linear(target) if self.targets_are_srgb else target
        B = rays_o.shape[0]; dtype = rays_o.dtype

        with torch.amp.autocast(self.device.type, enabled=self.amp):
            # Coarse stratified samples
            t = torch.linspace(0., 1., steps=self.nc, device=self.device, dtype=dtype)
            zc = self.near * (1. - t) + self.far * t
            zc = zc.expand(B, self.nc).contiguous()
            # Training perturbation
            mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
            lower = torch.cat([zc[:, :1], mids], dim=-1)
            upper = torch.cat([mids, zc[:, -1:]], dim=-1)
            zc = lower + (upper - lower) * torch.rand_like(zc)  # perturb=True
            zc = torch.sort(zc, dim=-1).values

            comp_c, w_c, _, _ = self._forward_once(rays_o, rays_d, zc, self.nerf_c)

            # Fine sampling via PDF (skip first/last)
            mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
            w_mid = w_c[:, 1:-1].detach()
            zf = sample_pdf(mids, w_mid, n_samples=self.nf, deterministic=self.det_fine)
            z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

            comp_f, _, _, _ = self._forward_once(rays_o, rays_d, z_all, self.nerf_f)

            comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            target_lin = torch.nan_to_num(target_lin, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            import torch.nn.functional as F
            loss_c = F.mse_loss(comp_c, target_lin)
            loss_f = F.mse_loss(comp_f, target_lin)
            loss = loss_c + loss_f
            psnr = mse2psnr(loss_f.detach())

        return {"loss": loss, "psnr": psnr, "comp_f": comp_f.detach(), "comp_c": comp_c.detach()}

    def _train_step_chunked(self, batch, micro_chunks: int) -> tuple[float, float]:
        rays_o, rays_d, target = batch["rays_o"], batch["rays_d"], batch["rgb"]
        B = rays_o.shape[0]
        m = max(1, int(micro_chunks))
        chunk = (B + m - 1) // m

        total_loss = 0.0; last_psnr = 0.0
        for i in range(0, B, chunk):
            ro = rays_o[i:i+chunk]; rd = rays_d[i:i+chunk]; tgt_slice = target[i:i+chunk]
            Bi = ro.shape[0]; dtype = ro.dtype

            with torch.amp.autocast(self.device.type, enabled=self.amp):
                t = torch.linspace(0., 1., steps=self.nc, device=self.device, dtype=dtype)
                zc = self.near * (1. - t) + self.far * t
                zc = zc.expand(Bi, self.nc).contiguous()
                mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
                lower = torch.cat([zc[:, :1], mids], dim=-1)
                upper = torch.cat([mids, zc[:, -1:]], dim=-1)
                zc = lower + (upper - lower) * torch.rand_like(zc)
                zc = torch.sort(zc, dim=-1).values

                comp_c, w_c, _, _ = self._forward_once(ro, rd, zc, self.nerf_c)

                mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
                zf = sample_pdf(mids, w_c[:, 1:-1].detach(), n_samples=self.nf, deterministic=self.det_fine)
                z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

                comp_f, _, _, _ = self._forward_once(ro, rd, z_all, self.nerf_f)

                tgt_lin = srgb_to_linear(tgt_slice) if self.targets_are_srgb else tgt_slice
                tgt = tgt_lin.to(device=self.device, dtype=comp_f.dtype, non_blocking=True)

                import torch.nn.functional as F
                comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                loss = (F.mse_loss(comp_c, tgt) + F.mse_loss(comp_f, tgt)) / m
                last_psnr = float(mse2psnr(F.mse_loss(comp_f, tgt).detach()).item())

            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())

        self.scaler.unscale_(self.opt)
        clip = float(getattr(self.cfg, "grad_clip_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()), max_norm=clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.sched is not None: self.sched.step()

        return total_loss, last_psnr

    # ---------------- Utilities ----------------

    def _forward_once(self, rays_o, rays_d, z, model):
        B, N = z.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]
        viewdirs = F.normalize(rays_d, dim=-1)
        vdirs = viewdirs[:, None, :].expand(B, N, 3)

        enc_pos = self.pos_enc(pts.reshape(-1, 3))
        enc_dir = self.dir_enc(vdirs.reshape(-1, 3))

        pred = model(enc_pos, enc_dir)
        rgb_raw, sigma_raw = pred[..., :3], pred[..., 3]
        rgb = torch.sigmoid(rgb_raw).reshape(B, N, 3)

        # Vanilla NeRF: add Gaussian noise to *raw* density during training and use ReLU
        sigma_raw_use = sigma_raw
        if self.vanilla and model.training:
            sigma_raw_use = sigma_raw_use + torch.randn_like(sigma_raw_use) * 1.0
        elif (not self.vanilla) and float(getattr(self, "raw_noise_std", 0.0)) > 0.0 and model.training:
            sigma_raw_use = sigma_raw_use + torch.randn_like(sigma_raw_use) * float(self.raw_noise_std)

        if (self.vanilla) or (str(getattr(self, "sigma_activation", "softplus")).lower() == "relu"):
            sigma = torch.relu(sigma_raw_use).reshape(B, N)
        else:
            sigma = F.softplus(sigma_raw_use, beta=1.0).reshape(B, N)

        rn = rays_d.norm(dim=-1, keepdim=True)
        comp_rgb, weights, acc, depth = volume_render_rays(
            rgb=rgb, sigma=sigma, z_depths=z, ray_norm=rn, white_bkgd=bool(getattr(self.cfg, "white_bkgd", True))
        )
        return comp_rgb, weights, acc, depth
