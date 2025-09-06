"""Contains code for training an MLP"""


from __future__ import annotations
import random
from collections import deque
import math, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import signal
from datetime import datetime
from contextlib import contextmanager

import re
import imageio.v2 as imageio  # uses imageio-ffmpeg under the hood
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

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

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow((x + 0.055) / 1.055, 2.4),
    )

def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(x, 1.0 / 2.4) - 0.055,
    )

def save_rgb_png(t: torch.Tensor, path: Path):
    arr = (t.clamp(0,1).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(arr).save(path)

def save_gray_png(img, path):
    """
    Save a single-channel image in [0,1] as a PNG by expanding to 3 channels.
    Accepts torch.Tensor [H,W] or [H,W,1] or numpy with same shapes.
    """
    if isinstance(img, torch.Tensor):
        img = img.clamp(0, 1)
        if img.ndim == 2:
            img = img.unsqueeze(-1)
        img = img.repeat(1, 1, 3)   # [H,W,3]
    else:
        # numpy path
        import numpy as np
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = img.clip(0, 1)
    save_rgb_png(img, path)

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
            acc0=self.cfg.init_acc, near=self.near, far=self.far,
        ).to(self.device)
        self.nerf_f = NeRF(
            enc_pos_dim=self.pos_enc.out_dim,
            enc_dir_dim=self.dir_enc.out_dim,
            acc0=self.cfg.init_acc, near=self.near, far=self.far,
        ).to(self.device)
        print("[DEBUG] sigma_out.bias[0] =", self.nerf_f.sigma_out.bias[0].item())
        with torch.no_grad():
            self._w0_norm = float(sum(p.norm().detach() for p in self.nerf_f.parameters()))
            self._w0_cksm = float(torch.stack([p.float().detach().abs().mean() for p in self.nerf_f.parameters()]).mean())
            self._probe_param = next(self.nerf_f.parameters())
            self._last_probe = float(self._probe_param.flatten()[0].detach())

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

        # Are dataset images sRGB (usual for PNG/JPEG)? If so, convert → linear for loss.
        self.targets_are_srgb: bool = getattr(self.cfg, "targets_are_srgb", True)

        # optional runtime toggles injected from driver (train_nerf.py)
        self.micro_chunks: int = getattr(self, "micro_chunks", 0)  # 0 = disabled
        self.ckpt_mlp: bool = getattr(self, "ckpt_mlp", False)

        # thermal safety toggles (can be absent; provide sane defaults)
        self.gpu_temp_threshold: int = getattr(self, "gpu_temp_threshold", 85)
        self.gpu_temp_check_every: int = getattr(self, "gpu_temp_check_every", 20)
        self.gpu_cooldown_seconds: int = getattr(self, "gpu_cooldown_seconds", 45)
        self.thermal_throttle: bool = getattr(self, "thermal_throttle", False)
        self.thermal_throttle_max_micro: int = getattr(self, "thermal_throttle_max_micro", 16)
        self.thermal_throttle_sleep: float = getattr(self, "thermal_throttle_sleep", 5.0)
        self._last_temp_check_step: int = 0
        self._pynvml_ready: bool = self._maybe_init_pynvml()

        # ---- Run control / resume flags (injected by script via setattr) ----
        self.auto_resume: bool = getattr(self, "auto_resume", False)
        self.resume_path: str | None = getattr(self, "resume_path", None)
        self.resume_no_optim: bool = getattr(self, "resume_no_optim", False)
        self.on_interrupt: str = getattr(self, "on_interrupt", "render_and_exit")  # none|save|render|render_and_exit
        self.on_pause_signal: str = getattr(self, "on_pause_signal", "render")     # render|save_and_render

        # ---- Internal control flags set by signals ----
        self._sigint = False
        self._sigusr1 = False
        self._cancel_render = False
        self._install_signal_handlers()

        # ---- TensorBoard flags (set later via extras/CLI), writer created lazily
        self.use_tb: bool = getattr(self, "use_tb", False)
        self.tb_logdir: str | None = getattr(self, "tb_logdir", None)
        self.tb_group_root: str | None = getattr(self, "tb_group_root", None)
        self._tb = None

        # ETA smoothing over recent log intervals
        self._eta_window = deque(maxlen=20)  # store seconds-per-step values
        self._last_log_time = time.time()
        self._last_log_step = 0

    def maybe_init_tensorboard(self):
        """Create SummaryWriter once, after tb_* attrs are set from YAML/CLI."""
        if self._tb is not None:
            return
        if not getattr(self, "use_tb", False):
            return
        exp_dir = self.out_dir
        run_name = exp_dir.name
        logdir = self.tb_logdir or str(exp_dir / "tb")
        try:
            self._tb = SummaryWriter(logdir)
            print(f"[TB] Logging to: {logdir} (run: {run_name})")
            # (optional) add grouping/symlink logic here if you want
        except Exception as e:
            print(f"[TB] Failed to create SummaryWriter: {e}")
            self._tb = None

    # ------------------------ Signal handling ------------------------
    def _install_signal_handlers(self):
        def _h_int(signum, frame):
            print("\n[CTRL-C] SIGINT received → scheduling graceful interrupt after current step.")
            self._sigint = True
        def _h_usr1(signum, frame):
            print("\n[PAUSE] SIGUSR1 received → scheduling pause-render after current step.")
            self._sigusr1 = True
        def _h_usr2(signum, frame):
            print("\n[CANCEL-RENDER] SIGUSR2 received → will abort current render at next frame.")
            self._cancel_render = True
        try:
            signal.signal(signal.SIGINT, _h_int)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGUSR1, _h_usr1)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGUSR2, _h_usr2)
        except Exception:
            pass

    # ------------------------ Checkpoints ------------------------
    @property
    def ckpt_dir(self) -> Path:
        d = self.out_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def preview_dir(self) -> Path:
        d = self.out_dir / "preview"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"ckpt_{step:07d}.pt"

    def save_checkpoint(
        self,
        step: int,
        tag: str | None = None,
        latest: bool = True,
        include_optim: bool = True,
    ) -> Path:
        """
        Save a checkpoint to <exp_dir>/checkpoints.
        If `tag` is provided, uses <tag>.pt; otherwise uses step_XXXXXXX.pt.
        If `latest` is True, also update ckpt_latest.pt in the same folder.
        """
        # Decide filename
        path = (self.ckpt_dir / f"{tag}.pt") if tag else self._ckpt_path(step)

        # Build object
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

        # Update latest pointer if requested
        if latest:
            latest_path = self.ckpt_dir / "ckpt_latest.pt"
            try:
                if latest_path.exists() or latest_path.is_symlink():
                    latest_path.unlink()
                # Relative symlink inside checkpoints/ if filesystem allows
                latest_path.symlink_to(path.name)
            except Exception:
                # Fallback: copy the checkpoint object
                try:
                    torch.save(obj, latest_path)
                except Exception:
                    pass

        return path

    def find_latest_checkpoint(self) -> Path | None:
        # 1) Prefer an explicit "latest" pointer if present
        for p in [self.ckpt_dir / "ckpt_latest.pt", self.out_dir / "ckpt_latest.pt"]:
            try:
                if p.exists():
                    return p.resolve() if p.is_symlink() else p
            except Exception:
                pass

        # 2) Gather all candidates (new + legacy)
        candidates: list[Path] = []
        candidates += list(self.ckpt_dir.glob("step_*.pt"))
        candidates += list(self.ckpt_dir.glob("ckpt_*.pt"))
        # legacy root-level checkpoints from older runs:
        candidates += list(self.out_dir.glob("ckpt_*.pt"))

        if not candidates:
            return None

        step_re = re.compile(r"(?:step|ckpt)[_-]?(\d+)", re.IGNORECASE)

        def step_of(p: Path) -> int:
            m = step_re.search(p.name)
            return int(m.group(1)) if m else -1  # -1 = unknown

        # 3) Sort by (parsed_step, mtime) and take the max
        candidates.sort(key=lambda p: (step_of(p), p.stat().st_mtime))
        return candidates[-1]

    def load_checkpoint(self, path: Path, load_optim: bool = True) -> int:
        print(f"[CKPT] loading ← {path}")
        obj = torch.load(path, map_location=self.device)
        self.nerf_c.load_state_dict(obj["nerf_c"])
        self.nerf_f.load_state_dict(obj["nerf_f"])
        if load_optim and obj.get("opt") is not None and hasattr(self, "opt"):
            self.opt.load_state_dict(obj["opt"])
        if obj.get("scaler") is not None and getattr(self, "scaler", None) is not None:
            self.scaler.load_state_dict(obj["scaler"])
        if obj.get("sched") is not None and getattr(self, "sched", None) is not None:
            self.sched.load_state_dict(obj["sched"])
        return int(obj.get("step", 0))

    def prune_checkpoints(self, keep: int = 5):
        """Keep only the newest `keep` step_* checkpoints (preserves ckpt_latest.pt)."""
        ckpts = sorted(self.ckpt_dir.glob("step_*.pt"))
        to_delete = ckpts[:-keep] if len(ckpts) > keep else []
        for p in to_delete:
            try:
                p.unlink()
            except Exception:
                pass

    # ------------------------ Mid-run rendering ------------------------
    def cancel_render(self):
        """You can also call this from code to abort a render early."""
        self._cancel_render = True

    @contextmanager
    def _render_cancellable(self):
        """
        Use around any long render loop. Resets cancel flag on enter,
        guarantees writer/cleanup blocks will still run.
        """
        self._cancel_render = False
        try:
            yield
        finally:
            # don't reset here; leave it True if it was requested, just in case caller wants to inspect
            pass

    def _render_should_abort(self) -> bool:
        """Lightweight, inlined check to sprinkle inside per-frame loops."""
        return bool(self._cancel_render)

    def render_artifacts(self):
        """
        Run validation video export and fly-around path render.
        Assumes you already have the underlying routines in this class.
        """
        try:
            # (A) Validation → video (implement with your existing val render loop)
            if hasattr(self, "export_validation_video"):
                self.export_validation_video()
            elif hasattr(self, "validate_full_image"):
                # Fallback: run a single full-res validation frame as a sanity artifact
                print("[RENDER] export_validation_video() not found; rendering single validation frame instead.")
                with torch.no_grad():
                    _ = self.validate_full_image(idx=getattr(self, "val_frame_idx", 0))
            # (B) Fly-around / path render with current model
            # Output under: <exp_dir>/path/
            path_dir = self.out_dir / "render_path"
            self.render_camera_path(
                out_dir=path_dir,
                n_frames=getattr(self.cfg, "path_frames", 120),
                fps=getattr(self.cfg, "path_fps", 30),
                path_type=getattr(self.cfg, "path_type", "circle"),
                res_scale=getattr(self.cfg, "path_res_scale", 1.0),
                save_video=True,
            )
        except Exception as e:
            print(f"[RENDER] Skipped mid-run rendering due to error: {e}")

    # ------------------------ Validation video stitching ------------------------
    def export_validation_video(self,
                                src_glob: str | None = None,
                                out_path: str | Path | None = None,
                                fps: int | None = None,
                                pad_to_mod: int = 16) -> Path | None:
        """
        Stitch saved validation frames into a single MP4.
        - src_glob: glob pattern relative to experiment dir (self.out_dir). If None, tries common defaults.
        - out_path: output filename (relative or absolute). If None, uses <exp>/val_video.mp4.
        - fps: frames per second. If None, uses self.val_video_fps or 24.
        - pad_to_mod: pad width/height up to next multiple (16 recommended for codec compatibility).
        Returns: Path to the written video or None if nothing was written.
        """
        exp_dir = self.out_dir
        fps = int(fps if fps is not None else getattr(self, "val_video_fps", 24))
        # Resolve glob for frames
        if src_glob is None:
            # Try user-provided CLI, then sensible defaults
            src_glob = getattr(self, "val_video_glob", None)
        if src_glob is None:
            # Common layouts: "<exp>/val/step_xxxxxx.png" or nested
            # Feel free to tailor this if your val saver uses a specific path
            candidates = ["preview/step_*.png", "preview/*.png"]
        else:
            candidates = [src_glob]

        def _list_frames():
            frames: list[Path] = []
            for pat in candidates:
                frames.extend(sorted(exp_dir.glob(pat), key=lambda p: p.as_posix()))
            # Deduplicate while preserving order
            seen: set[Path] = set()
            uniq: list[Path] = []
            for p in frames:
                if p not in seen:
                    uniq.append(p)
                    seen.add(p)
            return uniq

        frames = _list_frames()
        if not frames:
            print(f"[VAL-VIDEO] No validation frames found under {exp_dir} with patterns: {candidates}")
            return None

        # Filenames are zero-padded (step_0001000.png), so simple string sort is fine
        frames.sort(key=lambda p: p.name)


        # Output path
        if out_path is None:
            out_path = exp_dir / "val_video.mp4"
        else:
            out_path = Path(out_path)
            if not out_path.is_absolute():
                out_path = exp_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read first frame to establish size
        first = imageio.imread(frames[0])
        if first.ndim == 2:
            first = np.stack([first]*3, axis=-1)
        H, W = first.shape[:2]
        newW = W + ((pad_to_mod - (W % pad_to_mod)) % pad_to_mod)
        newH = H + ((pad_to_mod - (H % pad_to_mod)) % pad_to_mod)
        pad_w = newW - W
        pad_h = newH - H

        def _pad(img: np.ndarray) -> np.ndarray:
            if pad_w == 0 and pad_h == 0:
                return img
            # pad on right/bottom to preserve top-left alignment
            if img.ndim == 2:
                return np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        # Write video via imageio-ffmpeg; we pad ourselves so no macro_block warning
        writer = imageio.get_writer(str(out_path),
                                    fps=fps,
                                    codec="libx264",
                                    quality=8,  # adjust if desired
                                    ffmpeg_params=["-pix_fmt", "yuv420p"])  # wide compatibility
        try:
            with self._render_cancellable():
                writer.append_data(_pad(first))
                for idx, fp in enumerate(frames[1:], start=1):
                    if self._render_should_abort():
                        print(f"[VAL-VIDEO] Cancel requested → stopping at frame {idx}/{len(frames)}")
                        break
                    img = imageio.imread(fp)
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=-1)
                    if img.shape[0] != H or img.shape[1] != W:
                        # Resize to match first frame if shapes vary (rare). Simple nearest to avoid heavy deps.
                        # Replace with cv2.resize if OpenCV is already in your stack.
                        y = np.linspace(0, img.shape[0]-1, H).astype(np.int32)
                        x = np.linspace(0, img.shape[1]-1, W).astype(np.int32)
                        img = img[y][:, x]
                    writer.append_data(_pad(img))
        finally:
            writer.close()
        print(f"[VAL-VIDEO] Wrote {out_path}  ({len(frames)} frames @ {fps} FPS; size {newW}x{newH})")
        return out_path

    # ---------- Thermal helpers ----------
    def _maybe_init_pynvml(self) -> bool:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception:
            return False

    def _read_gpu_stats(self) -> dict:
        """
        Returns a dict with keys:
          temp_c, util_pct, mem_used_mb, mem_total_mb
        Missing fields may be None if not available.
        """
        stats = {"temp_c": None, "util_pct": None, "mem_used_mb": None, "mem_total_mb": None}
        if getattr(self, "_pynvml_ready", False):
            try:
                t = self._nvml.nvmlDeviceGetTemperature(self._nvml_handle, self._nvml.NVML_TEMPERATURE_GPU)
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                stats["temp_c"] = int(t)
                stats["util_pct"] = int(util.gpu)
                stats["mem_used_mb"] = int(mem.used // (1024 * 1024))
                stats["mem_total_mb"] = int(mem.total // (1024 * 1024))
                return stats
            except Exception:
                pass
        # Fallback to nvidia-smi (slower)
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL)
            line = out.decode("utf-8").strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                stats["temp_c"] = int(parts[0])
                stats["util_pct"] = int(parts[1])
                stats["mem_used_mb"] = int(parts[2])
                stats["mem_total_mb"] = int(parts[3])
        except Exception:
            pass
        return stats

    def _thermal_guard(self, step: int):
        """Check temperature every N steps; throttle or sleep if too hot."""
        if self.gpu_temp_check_every <= 0:
            return
        if step - self._last_temp_check_step < self.gpu_temp_check_every:
            return
        self._last_temp_check_step = step

        # read and log GPU vitals
        gpu = self._read_gpu_stats()
        t = gpu.get("temp_c")
        if self._tb is not None:
            # push vitals to TensorBoard
            if gpu["temp_c"] is not None:
                self._tb.add_scalar("gpu/temp_c", gpu["temp_c"], step)
            if gpu["util_pct"] is not None:
                self._tb.add_scalar("gpu/util_pct", gpu["util_pct"], step)
            if gpu["mem_used_mb"] is not None:
                self._tb.add_scalar("gpu/mem_used_mb", gpu["mem_used_mb"], step)
            if gpu["mem_total_mb"] is not None:
                self._tb.add_scalar("gpu/mem_total_mb", gpu["mem_total_mb"], step)
            self._tb.add_scalar("train/micro_chunks", getattr(self, "micro_chunks", 0) or 0, step)

        if t is None or t < self.gpu_temp_threshold:
            return

        # Too hot: take action
        if self.thermal_throttle:
            # Increase micro-chunks (more splitting → less peak load / heat)
            prev = int(getattr(self, "micro_chunks", 0) or 1)
            new = min(max(prev * 2, prev + 1), self.thermal_throttle_max_micro)
            if new > prev:
                self.micro_chunks = new
                print(f"[THERMAL] GPU {t}°C ≥ {self.gpu_temp_threshold}°C → increasing micro-chunks {prev} → {new} "
                      f"(max {self.thermal_throttle_max_micro}). Sleeping {self.thermal_throttle_sleep:.1f}s.")
                time.sleep(self.thermal_throttle_sleep)
            else:
                # Already at max, brief cooloff anyway
                print(f"[THERMAL] GPU {t}°C hot and micro-chunks at cap ({self.micro_chunks}). "
                      f"Cooling for {self.gpu_cooldown_seconds}s.")
                time.sleep(self.gpu_cooldown_seconds)
        else:
            # Simple cooldown sleep
            print(f"[THERMAL] GPU {t}°C ≥ {self.gpu_temp_threshold}°C → cooling for {self.gpu_cooldown_seconds}s.")
            time.sleep(self.gpu_cooldown_seconds)


    def _forward_nerf(self, rays_o: torch.Tensor, rays_d: torch.Tensor, z: torch.Tensor, model: nn.Module):
        B, N = z.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]     # [B,N,3]
        viewdirs = F.normalize(rays_d, dim=-1)
        vdirs = viewdirs[:, None, :].expand(B, N, 3)                      # [B,N,3]

        enc_pos = self.pos_enc(pts.reshape(-1, 3))
        enc_dir = self.dir_enc(vdirs.reshape(-1, 3))

        # pred = model(enc_pos, enc_dir)                                    # [B*N, 4]

        # Optional gradient checkpoint around MLP to reduce activation memory
        # Use the model's .training flag (True in train mode, False in eval)
        if self.ckpt_mlp and model.training:
            # Non-reentrant checkpoint doesn't require input tensors to have requires_grad=True
            try:
                pred = checkpoint(lambda ep, ed: model(ep, ed), enc_pos, enc_dir, use_reentrant=False)
            except TypeError:
                # Back-compat for older torch that doesn't support use_reentrant kwarg
                pred = checkpoint(lambda ep, ed: model(ep, ed), enc_pos, enc_dir)
        else:
            pred = model(enc_pos, enc_dir)                                # [B*N, 4]

        # rgb = pred[..., :3].reshape(B, N, 3)
        # sigma = pred[..., 3].reshape(B, N)
        rgb_raw  = pred[..., :3]
        sigma_raw = pred[..., 3]

        rgb   = torch.sigmoid(rgb_raw).reshape(B, N, 3)
        sigma = torch.nn.functional.softplus(sigma_raw, beta=1.0).reshape(B, N)  # safe density

        # if not model.training:  # eval only
        #     print("[VAL DEBUG] sigma stats:",
        #         float(sigma.min()), float(sigma.mean()), float(sigma.max()))

        # Encourage crisper opacity instead of “transparent smears”
        sigma_noise_std = float(getattr(self.cfg, "sigma_noise_std", 1e-3))
        if model.training and sigma_noise_std > 0:
            sigma = sigma + sigma_noise_std * torch.randn_like(sigma)

        comp_rgb, weights, acc, depth = volume_render_rays(
            rgb=rgb, sigma=sigma, z_depths=z, white_bkgd=self.cfg.white_bg
        )
        return comp_rgb, weights, acc, depth

    def train_step(self, batch) -> dict[str, torch.Tensor]:
        rays_o, rays_d, target = batch["rays_o"], batch["rays_d"], batch["rgb"]
        target_lin = srgb_to_linear(target) if self.targets_are_srgb else target
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

            # Sanitize
            comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            target_lin = torch.nan_to_num(target_lin, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            # Losses
            loss_c = F.mse_loss(comp_c, target_lin)
            loss_f = F.mse_loss(comp_f, target_lin)
            loss = loss_c + loss_f
            psnr = mse2psnr(loss_f.detach())

        return {"loss": loss, "psnr": psnr, "comp_f": comp_f.detach(), "comp_c": comp_c.detach()}

    def train_step_chunked_backward(self, batch, micro_chunks: int) -> dict[str, torch.Tensor]:
        rays_o, rays_d, target = batch["rays_o"], batch["rays_d"], batch["rgb"]
        B = rays_o.shape[0]
        m = max(1, int(micro_chunks))
        chunk = (B + m - 1) // m

        total_loss = 0.0
        last_psnr = None

        for i in range(0, B, chunk):
            ro = rays_o[i:i+chunk]
            rd = rays_d[i:i+chunk]
            tgt_slice = target[i:i+chunk]  # slice first; convert to linear later
            Bi = ro.shape[0]

            with torch.amp.autocast(self.device.type, enabled=self.cfg.amp):
                # --- Coarse ---
                zc = stratified_samples(self.near, self.far, self.nc, Bi,
                                        device=self.device, dtype=ro.dtype, perturb=True)
                comp_c, w_c, _, _ = self._forward_nerf(ro, rd, zc, self.nerf_c)

                # --- Fine sampling ---
                mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
                w_mid = w_c[:, 1:-1].detach()
                zf = sample_pdf(mids, w_mid, n_samples=self.nf, deterministic=self.det_fine)
                z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values

                # --- Fine ---
                comp_f, _, _, _ = self._forward_nerf(ro, rd, z_all, self.nerf_f)

                # Convert targets → linear to match training space
                tgt_lin = srgb_to_linear(tgt_slice) if self.targets_are_srgb else tgt_slice
                tgt = tgt_lin.to(device=self.device, dtype=comp_f.dtype, non_blocking=True)

                # Sanitize
                comp_c = torch.nan_to_num(comp_c, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                comp_f = torch.nan_to_num(comp_f, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

                # Losses
                loss_c = F.mse_loss(comp_c, tgt)
                loss_f = F.mse_loss(comp_f, tgt)
                loss = (loss_c + loss_f) / m  # scale for accumulation
                last_psnr = mse2psnr(loss_f.detach())

            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())

        return {"loss_value": total_loss, "psnr": last_psnr}

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

        nc_eval = int(getattr(self, "eval_nc", self.nc * 2))
        nf_eval = int(getattr(self, "eval_nf", self.nf * 2))
        return self._render_image_chunked(rays_o, rays_d, H, W, nc_eval, nf_eval, perturb=False)

    def save_preview_bundle(self, step: int, val: dict[str, torch.Tensor], prefix: str = "step") -> None:
        """
        Save RGB (sRGB), opacity (acc), and normalized depth previews for a given step.
        Expects `val` from validate_full_image(): {"rgb": [H,W,3], "acc": [H,W,1], "depth": [H,W,1]} in *linear* space.
        """
        # filenames with zero-padded step for stable sorting
        name = f"{prefix}_{step:07d}"
        rgb_path   = self.preview_dir / f"{name}.png"
        acc_path   = self.preview_dir / f"acc_{step:07d}.png"
        depth_path = self.preview_dir / f"depth_{step:07d}.png"

        # Convert RGB (linear → sRGB) for PNG viewing
        rgb_srgb = linear_to_srgb(val["rgb"])

        # Opacity in [0,1]
        acc_img = val["acc"].squeeze(-1).clamp(0, 1)

        # Depth normalized to [0,1] for viewing
        depth_lin = val["depth"]
        depth_norm = ((depth_lin - self.near) / (self.far - self.near + 1e-8)).squeeze(-1).clamp(0, 1)

        # Write files
        save_rgb_png(rgb_srgb, rgb_path)
        save_gray_png(acc_img, acc_path)
        save_gray_png(depth_norm, depth_path)

    def train(self):
        it = iter(self.sampler)
        self.nerf_c.train(); self.nerf_f.train()

        # --- optional resume ---
        start_step = 1
        resume_from = None
        if getattr(self, "resume_path", None):
            resume_from = Path(self.resume_path)
        elif getattr(self, "auto_resume", False):
            resume_from = self.find_latest_checkpoint()
        if resume_from and resume_from.exists():
            start_step = self.load_checkpoint(resume_from, load_optim=(not self.resume_no_optim)) + 1
            print(f"[CKPT] Resuming from step {start_step} (file: {resume_from.name})")
        else:
            if getattr(self, "auto_resume", False):
                print("[CKPT] auto-resume requested but no checkpoints found; starting fresh.")

        for step in range(start_step, self.cfg.max_steps + 1):
            batch = next(it)
            self.opt.zero_grad(set_to_none=True)

            if getattr(self, "micro_chunks", 0) and self.micro_chunks > 0:
                out = self.train_step_chunked_backward(batch, self.micro_chunks)
                # (Optional) guard using the averaged loss per step if you expose it; otherwise skip if psnr is NaN
                if out["psnr"] is not None and not torch.isfinite(out["psnr"]):
                    print(f"[WARN] Non-finite psnr at step {step}. Skipping optimizer step.")
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()
                    continue
                # Clip after unscale
                self.scaler.unscale_(self.opt)
                clip = float(getattr(self.cfg, "grad_clip_norm", 0.0) or 0.0)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
                        max_norm=clip
                    )
                # step once after all chunks
                self.scaler.step(self.opt)
                probe_now = float(self._probe_param.flatten()[0].detach())
                if step % self.cfg.log_every == 0:
                    print(f"[DEBUG] probe Δ = {probe_now - self._last_probe:+.3e}")
                self._last_probe = probe_now
                self.scaler.update()
                if self.sched is not None:
                    self.sched.step()
                loss_val = out["loss_value"]
                psnr_val = out["psnr"]
            else:
                out = self.train_step(batch)
                loss = out["loss"]
                if not torch.isfinite(loss):
                    print(f"[WARN] Non-finite loss at step {step}: {float(loss.detach().cpu())}. Skipping step.")
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.update()
                    continue
                self.scaler.scale(loss).backward()
                # Clip after unscale
                self.scaler.unscale_(self.opt)
                clip = float(getattr(self.cfg, "grad_clip_norm", 0.0) or 0.0)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.nerf_c.parameters()) + list(self.nerf_f.parameters()),
                        max_norm=clip
                    )
                self.scaler.step(self.opt)
                probe_now = float(self._probe_param.flatten()[0].detach())
                if step % self.cfg.log_every == 0:
                    print(f"[DEBUG] probe Δ = {probe_now - self._last_probe:+.3e}")
                self._last_probe = probe_now
                self.scaler.update()
                if self.sched is not None:
                    self.sched.step()
                loss_val = float(loss.detach().cpu())
                psnr_val = out["psnr"]



            # Thermal guard (periodic)
            self._thermal_guard(step)

            # ----- Handle pause/cancel signals after the step -----
            if self._sigusr1:
                # optional save before render
                if self.on_pause_signal == "save_and_render":
                    try:
                        self.save_checkpoint(step, tag=f"pause_step_{step}")
                    except Exception as e:
                        print(f"[PAUSE] checkpoint save failed: {e}")
                # render artifacts, then resume
                self.render_artifacts()
                self._sigusr1 = False

            if self._sigint:
                # graceful interrupt according to policy
                mode = self.on_interrupt
                try:
                    if mode in ("save", "render_and_exit"):
                        self.save_checkpoint(step, tag=f"interrupt_step_{step}")
                except Exception as e:
                    print(f"[INT] checkpoint save failed: {e}")
                try:
                    if mode in ("render", "render_and_exit"):
                        self.render_artifacts()
                except Exception as e:
                    print(f"[INT] render failed: {e}")
                print("[INT] Exiting training loop.")
                break

            if step % self.cfg.log_every == 0:

                # time since last LOG (not per single step)
                now = time.time()
                secs = now - self._last_log_time
                steps_delta = step - (self._last_log_step or 0)
                if steps_delta <= 0:
                    steps_delta = max(1, getattr(self.cfg, "log_every", 1))

                sec_per_step = secs / steps_delta
                self._eta_window.append(sec_per_step)
                avg_sec_per_step = sum(self._eta_window) / len(self._eta_window)

                steps_left = max(0, self.cfg.max_steps - step)
                eta_sec = steps_left * avg_sec_per_step
                eta_h = int(eta_sec // 3600)
                eta_m = int((eta_sec % 3600) // 60)
                eta_s = int(eta_sec % 60)
                eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

                # update log anchors
                self._last_log_time = now
                self._last_log_step = step

                # your existing scalars
                ps = psnr_val.item() if hasattr(psnr_val, "item") else float(psnr_val)
                lr_now = self.opt.param_groups[0]["lr"]

                print(
                    f"[{step:>7d}] loss={loss_val:.6f} psnr={ps:.2f} "
                    f"lr={lr_now:.2e} ({secs:.2f}s/{steps_delta} steps, {sec_per_step:.3f}s/step "
                    f"avg {avg_sec_per_step:.3f}s) ETA {eta_str}"
                )
                if self._tb is not None:
                    self._tb.add_scalar("train/loss", loss_val, step)
                    self._tb.add_scalar("train/psnr", ps, step)
                    self._tb.add_scalar("train/lr", lr_now, step)


            if step % self.cfg.val_every == 0:
                with torch.no_grad():
                    wnorm = float(sum(p.norm() for p in self.nerf_f.parameters()))
                    wcksm = float(torch.stack([p.float().abs().mean() for p in self.nerf_f.parameters()]).mean())
                if self._tb is not None:
                    self._tb.add_scalar("debug/nerf_f_weight_norm", wnorm, step)
                    self._tb.add_scalar("debug/nerf_f_weight_cksm", wcksm, step)
                print(f"[DEBUG] nerf_f | weight_norm={wnorm:.3f}  weight_cksm={wcksm:.6f}")
                self.nerf_c.eval(); self.nerf_f.eval()
                with torch.no_grad():
                    val = self.validate_full_image(frame_idx=self.val_frame_idx)
                    rgbl = val["rgb"]  # linear
                    acc  = val["acc"]
                    print(
                        f"[VAL] acc mean={acc.mean().item():.4f} max={acc.max().item():.4f}  "
                        f"rgb lin min={rgbl.min().item():.4f} max={rgbl.max().item():.4f} mean={rgbl.mean().item():.4f}"
                    )
                    print(f"[VAL] near={self.near} far={self.far} acc_mean={val['acc'].mean().item():.4f}")
                    self.save_preview_bundle(step, val)
                self.nerf_c.train(); self.nerf_f.train()

            if step % self.cfg.ckpt_every == 0:
                self.save_checkpoint(step)
                self.prune_checkpoints(keep=5)

        # close TB
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass

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
        # Use eval overrides + shared chunked renderer (with OOM fallback)
        nc_eval = int(getattr(self, "eval_nc", self.nc))
        nf_eval = int(getattr(self, "eval_nf", self.nf))
        res = self._render_image_chunked(rays_o, rays_d, H, W, nc_eval, nf_eval, perturb=False)
        return res["rgb"]  # keep render_pose()’s original return type

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

        # Render frames (cancelable with SIGUSR2)
        png_paths = []
        with self._render_cancellable():
            for idx, pose in enumerate(tqdm(c2ws, desc="Rendering path", unit="frame")):
                if self._render_should_abort():
                    print(f"[RENDER] Cancel requested → stopping at frame {idx}/{len(c2ws)}")
                    break
                img = self.render_pose(pose, H=H, W=W, K=K)
                path = out_dir / f"path_{idx:04d}.png"
                save_rgb_png(linear_to_srgb(img), path)
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

    def _render_image_chunked(self, rays_o, rays_d, H, W, nc_eval, nf_eval, perturb=False):
        chunk = int(getattr(self, "eval_chunk", 8192))
        while True:
            try:
                out_rgb, out_acc, out_depth = [], [], []
                for i in range(0, H * W, chunk):
                    ro = rays_o[i:i+chunk]; rd = rays_d[i:i+chunk]
                    Bc = ro.shape[0]
                    zc = stratified_samples(self.near, self.far, nc_eval, Bc, device=self.device, dtype=ro.dtype, perturb=perturb)
                    comp_c, w_c, _, _ = self._forward_nerf(ro, rd, zc, self.nerf_c)
                    mids = 0.5 * (zc[:, 1:] + zc[:, :-1])
                    zf = sample_pdf(mids, w_c[:, 1:-1], n_samples=nf_eval, deterministic=not perturb)
                    z_all = torch.sort(torch.cat([zc, zf], dim=-1), dim=-1).values
                    comp_f, w_f, _, _ = self._forward_nerf(ro, rd, z_all, self.nerf_f)

                    out_rgb.append(comp_f)                                  # [B,3]
                    out_acc.append(w_f.sum(-1, keepdim=True))               # [B,1]
                    out_depth.append((w_f * z_all).sum(-1, keepdim=True))   # [B,1]

                rgb   = torch.cat(out_rgb,   dim=0).reshape(H, W, 3)
                acc   = torch.cat(out_acc,   dim=0).reshape(H, W, 1)
                depth = torch.cat(out_depth, dim=0).reshape(H, W, 1)
                return {"rgb": rgb, "acc": acc, "depth": depth}
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and chunk > 512:
                    new_chunk = max(512, chunk // 2)
                    print(f"[VAL] OOM with chunk={chunk}. Retrying with chunk={new_chunk}.")
                    chunk = new_chunk
                    torch.cuda.empty_cache()
                    continue
                raise