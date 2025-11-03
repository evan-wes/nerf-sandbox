"""
Training / rendering entry script for NeRF sandbox.

Goals
-----
- Keep *all* config/default mutation here (NOT inside Trainer).
- Provide a dataset-aware `--vanilla` profile that mirrors bmild/nerf for
  both Blender (synthetic) and LLFF (forward-facing) datasets.
- Remove redundant / unused switches and parse complex flags up-front.

Usage (examples)
----------------
# Blender (synthetic)
python -m nerf_sandbox.source.scripts.train_nerf \
  --data_kind blender \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir ./exp/lego_vanilla \
  --vanilla --use_tb

# LLFF (fern)
python -m nerf_sandbox.source.scripts.train_nerf \
  --data_kind llff \
  --data_root /path/to/nerf_llff_data/fern \
  --downscale 1 \
  --llff_recenter \
  --llff_holdout_every 8 --llff_angle_sorted_holdout \
  --use_ndc --ndc_near_plane_world 1.0 \
  --white_bkgd True \
  --vanilla --use_tb \
  --out_dir ./exp/fern_vanilla

"""
from __future__ import annotations

import argparse
import ast
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

from nerf_sandbox.source.train.trainer import Trainer
from nerf_sandbox.source.utils.validation_renderer import ValidationRenderer


# ---------------------------
# Argument parsing
# ---------------------------

def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"true", "1", "yes", "y", "on"}: return True
    if s in {"false", "0", "no", "n", "off"}: return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("NeRF training / rendering entry script")

    # Dataset + IO
    p.add_argument("--data_kind", type=str, default="auto", choices=["auto", "blender", "llff"],
                   help="Dataset type. 'auto' infers LLFF if poses_bounds.npy exists")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--centering", choices=["auto", "none"], default=None,
                        help="Centering policy. LLFF default=auto (average-pose). Blender default=none.")
    p.add_argument("--scene_scale", type=float, default=1.0,
                        help="Uniform multiplier applied to camera translations after centering.")
    p.add_argument("--cache_images_on_device", type=bool, required=False, default=False,
                        help="Whether to cache the images on the same device. Leave off to save VRAM.")

    # LLFF specifics
    p.add_argument("--bd_factor", type=float, default=0.75,
                help="Scale factor applied to LLFF bounds before recentering (nerf-pytorch default 0.75).")
    p.add_argument("--use_llff_holdout", type=_str2bool, default=True,
                help="If True, choose a single test/val view via center distance (nerf-pytorch behavior).")
    p.add_argument("--holdout_every", type=int, default=0,
                help="If >0, override with periodic split cadence (every N frames).")
    p.add_argument("--holdout_offset", type=int, default=0,
                help="Offset for periodic holdout split.")

    # Ray / space conventions
    p.add_argument("--camera_convention", type=str, default=None,
                   help="If None, loader decides (Blender->opengl, LLFF->opengl)")
    p.add_argument("--use_ndc", action="store_true")
    p.add_argument("--ndc_near_plane_world", type=float, default=None,
                   help="near_world plane (world units) for forward-facing NDC warp. Default: loader near_world if None")
    p.add_argument("--white_bkgd", type=_str2bool, default=False)

    # Model + rendering core
    p.add_argument("--pos_num_freqs", type=int, default=10)
    p.add_argument("--dir_num_freqs", type=int, default=4)
    p.add_argument("--pos_include_input", type=_str2bool, default=True)
    p.add_argument("--dir_include_input", type=_str2bool, default=True)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--skip_pos", type=int, default=4)
    p.add_argument("--sigma_activation", type=str, default="relu", choices=["relu", "softplus"])

    # Sampling
    p.add_argument("--nc", type=int, default=64, help="Coarse samples per ray")
    p.add_argument("--nf", type=int, default=128, help="Fine samples per ray")
    p.add_argument("--det_fine", action="store_true", help="Disable fine stratified jitter")
    p.add_argument("--rays_per_batch", type=int, default=2048)
    p.add_argument("--raw_noise_std", type=float, default=1.0,
                   help="Stddev of gaussian noise added to sigma during training")
    p.add_argument("--precrop_iters", type=int, default=0,
                   help="Warm-start steps with center-crop (LLFF). Set by --vanilla for LLFF")
    p.add_argument("--precrop_frac", type=float, default=1.0)
    p.add_argument("--sample_from_single_frame", action="store_true",
                   help="Draw all rays in a step from one image (bmild-style); default is mixed")

    # Micro-batching / chunking
    p.add_argument("--micro_chunks", type=int, default=0,
                   help="If >0, split each training step into this many micro-batches (grad accumulation)")
    p.add_argument("--train_micro_chunks", type=int, default=None,
                   help="Override for training; defaults to --micro_chunks")
    p.add_argument("--eval_micro_chunks", type=int, default=None,
                   help="Override for eval/render; defaults to --micro_chunks")
    p.add_argument("--train_chunk", type=int, default=0,
                   help="If >0, cap training-time MLP queries per forward (0=auto); default is mixed")

    # Ranges
    p.add_argument("--near_world", type=float, default=None)
    p.add_argument("--far_world", type=float, default=None)

    # Optim / schedule
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])  # used by Trainer
    p.add_argument("--lr_scheduler_params", type=str, default={"eta_min": 5e-6, "T_max": 200_000},
                   help="JSON or Python dict for scheduler params, e.g. '{\"T_max\":200000,\"eta_min\":5e-6}'")

    # Runtime
    p.add_argument("--max_steps", type=int, default=200_000)
    p.add_argument("--ckpt_every", type=int, default=10_000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_tb", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # Validation rendering
    p.add_argument("--val_every", type=int, default=None,
                   help="If set, run validation every N steps; otherwise uses an auto schedule")
    p.add_argument("--val_indices", type=str, default=None,
                   help="Comma-separated indices to validate, e.g. '0,8,17' (defaults to [0])")
    p.add_argument("--num_val_steps", type=int, default=None,
                   help="If set, total number of validation step checkpoints to run")
    p.add_argument("--eval_chunk", type=int, default=16384, help="Eval micro-batch size")
    p.add_argument("--val_res_scale", type=float, default=1.0, help="Eval render resolution scale")
    p.add_argument("--progress_video_during_training", action="store_true",
                   help="Renders validation frames during training for use in creating a training progress video"
                   )
    p.add_argument("--val_schedule", type=str, default="power",
                    help="Validation cadence strategy; 'power' concentrates renders early.")
    p.add_argument("--val_power", type=float, default=2.0,
                    help="Exponent for 'power' schedule (>1 → denser early).")



    # Convenience profiles
    p.add_argument("--vanilla", action="store_true",
                   help="Apply dataset-aware defaults to mimic bmild/nerf for Blender and LLFF")

    # Render-only / resume
    p.add_argument("--render_only", action="store_true",
                   help="Skip training; just render a final camera path from a checkpoint")
    p.add_argument("--resume", type=str, default=None,
                   help="'latest' or a path to a checkpoint to resume from")

    # Path rendering
    p.add_argument("--render_path_after", action="store_true")
    p.add_argument("--progress_frames", type=int, default=120)
    p.add_argument("--path_fps", type=int, default=30)
    p.add_argument("--path_res_scale", type=float, default=1.0)



    return p


# ---------------------------
# Config helpers
# ---------------------------

def _parse_scheduler_params(s: str | dict) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    s = (s or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def make_cfg_from_args(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace(**vars(args))

    # Normalize some fields/types that Trainer expects
    cfg.lr_scheduler_params = _parse_scheduler_params(args.lr_scheduler_params)

    # Derive micro-chunk overrides
    if getattr(cfg, "train_micro_chunks", None) is None:
        cfg.train_micro_chunks = int(getattr(cfg, "micro_chunks", 0) or 0)
    if getattr(cfg, "eval_micro_chunks", None) is None:
        cfg.eval_micro_chunks = int(getattr(cfg, "micro_chunks", 0) or 0)

    if cfg.centering is None:
        if str(cfg.data_kind).lower() == "llff":
            cfg.centering = "auto"
        else:  # "blender" or anything else
            cfg.centering = "none"

    # Ensure integers
    cfg.train_chunk = int(getattr(cfg, "train_chunk", 0) or 0)

    # Ensure paths exist
    cfg.out_dir = str(args.out_dir)

    return cfg


def apply_vanilla_profile(cfg: SimpleNamespace) -> SimpleNamespace:
    """Dataset-aware profile that mirrors official NeRF defaults.

    Blender (synthetic):
      - world-space rays (no NDC), OpenGL convention
      - white background
      - mixed-frame sampling, no center-crop warm start
      - N_rand=1024, N_samples=64, N_importance=128

    LLFF (forward-facing):
      - NDC on (near_world=1.0), OpenGL convention, recenter
      - white background (pragmatic default; bmild used black — override if desired)
      - single-frame + center-crop for first ~1000 steps
      - N_rand=1024, N_samples=64, N_importance=128
    """
    data_kind = (getattr(cfg, "data_kind", "") or "").lower()

    # Common Nerf baseline knobs
    cfg.pos_num_freqs           = getattr(cfg, "pos_num_freqs", 10)
    cfg.dir_num_freqs           = getattr(cfg, "dir_num_freqs", 4)
    cfg.pos_include_input       = getattr(cfg, "pos_include_input", True)
    cfg.dir_include_input       = getattr(cfg, "dir_include_input", True)
    cfg.n_layers                = getattr(cfg, "n_layers", 8)
    cfg.hidden_dim              = getattr(cfg, "hidden_dim", 256)
    cfg.skip_pos                = getattr(cfg, "skip_pos", 4)
    cfg.sigma_activation        = getattr(cfg, "sigma_activation", "relu")
    cfg.nc                      = getattr(cfg, "nc", 64)
    cfg.nf                      = getattr(cfg, "nf", 128)
    cfg.det_fine                = getattr(cfg, "det_fine", False)
    cfg.rays_per_batch          = getattr(cfg, "rays_per_batch", 1024)
    cfg.raw_noise_std           = getattr(cfg, "raw_noise_std", 1.0)
    cfg.lr                      = getattr(cfg, "lr", 5e-4)
    cfg.infinite_last_bin       = getattr(cfg, "infinite_last_bin", True)
    cfg.lr_scheduler            = getattr(cfg, "lr_scheduler", "cosine")
    cfg.lr_scheduler_params     = getattr(cfg, "lr_scheduler_params", {"eta_min": 5e-6, "T_max": getattr(cfg, "max_steps", 200_000)})

    if data_kind in {"blender", "synthetic"}:
        cfg.use_ndc = False
        cfg.white_bkgd = True
        cfg.camera_convention = cfg.camera_convention or "opengl"
        cfg.sample_from_single_frame = False
        cfg.precrop_iters = 0
        cfg.precrop_frac  = 1.0

    elif data_kind == "llff":
        cfg.use_ndc = True
        cfg.ndc_near_plane_world = getattr(cfg, "ndc_near_plane_world", 1.0)
        cfg.camera_convention = cfg.camera_convention or "opengl"

        # NEW: exactly the loader’s knobs
        cfg.bd_factor = getattr(cfg, "bd_factor", 0.75)
        cfg.use_llff_holdout = getattr(cfg, "use_llff_holdout", True)
        cfg.holdout_every = getattr(cfg, "holdout_every", 0)
        cfg.holdout_offset = getattr(cfg, "holdout_offset", 0)

        # Vanilla sampling warm start
        cfg.sample_from_single_frame = True
        cfg.precrop_iters = getattr(cfg, "precrop_iters", 1000)
        cfg.precrop_frac  = getattr(cfg, "precrop_frac", 0.5)

        # Pragmatic defaults
        cfg.white_bkgd = True

    # else: leave user-provided settings intact

    return cfg

# --- Progress video path defaults --------------------------------------------

def _set_default(cfg, name, value, *, force: bool = False):
    """Set cfg.<name> = value only if missing (or if force=True)."""
    if force or not hasattr(cfg, name):
        setattr(cfg, name, value)

def apply_path_defaults_from_data_kind(cfg, data_kind: str, *, force: bool = False) -> SimpleNamespace:
    """
    Populate ONLY the fields used by the current PathPoseGenerator and ValidationRenderer.

    Fields set:
      - progress_frames        (int)
      - path_type              ("blender" | "llff_spiral" | "llff_zflat")
      - path_res_scale         (float)
      - path_fps               (int)

      Blender-only:
      - bl_phi_deg             (float)
      - bl_rots                (float)
      - bl_theta_start_deg     (float)
      - bl_radius              (float | None)

      LLFF-only:
      - rots                   (float)
      - zrate                  (float)
      - path_zflat             (bool)
    """
    kind = str(data_kind or "").lower()

    if kind in ("blender", "synthetic", "nerf_synthetic"):
        _set_default(cfg, "progress_frames",      120,       force=force)
        _set_default(cfg, "path_type",            "blender", force=force)
        _set_default(cfg, "path_res_scale",       1.0,       force=force)
        _set_default(cfg, "path_fps",             30,        force=force)

        # Blender spherical knobs
        _set_default(cfg, "bl_phi_deg",          -30.0,      force=force)
        _set_default(cfg, "bl_rots",              1.0,       force=force)
        _set_default(cfg, "bl_theta_start_deg", -180.0,      force=force)
        _set_default(cfg, "bl_radius",            None,      force=force)

    elif kind in ("llff", "llff_nerf", "llff_data"):
        _set_default(cfg, "progress_frames",      120,          force=force)
        _set_default(cfg, "path_type",            "llff_spiral",force=force)
        _set_default(cfg, "path_res_scale",       1.0,          force=force)
        _set_default(cfg, "path_fps",             24,           force=force)

        # LLFF spiral knobs (official)
        _set_default(cfg, "rots",                 2.0,          force=force)
        _set_default(cfg, "zrate",                0.5,          force=force)
        _set_default(cfg, "path_zflat",           False,        force=force)
        _set_default(cfg, "rads_scale",           3.0,          force=force)

    else:
        # Fallback: LLFF-style spiral with conservative defaults
        _set_default(cfg, "progress_frames",      120,          force=force)
        _set_default(cfg, "path_type",            "llff_spiral",force=force)
        _set_default(cfg, "path_res_scale",       1.0,          force=force)
        _set_default(cfg, "path_fps",             24,           force=force)

        _set_default(cfg, "rots",                 2.5,          force=force)
        _set_default(cfg, "zrate",                0.75,         force=force)
        _set_default(cfg, "path_zflat",           False,        force=force)

    return cfg



# ---------------------------
# Main
# ---------------------------

def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Build config and apply dataset-aware vanilla profile if requested
    cfg = make_cfg_from_args(args)
    if getattr(cfg, "vanilla", False):
        cfg = apply_vanilla_profile(cfg)

    cfg = apply_path_defaults_from_data_kind(cfg=cfg, data_kind=cfg.data_kind)

    # Ensure output directory exists
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construct trainer
    trainer = Trainer(cfg)

    # Optional: resume before training or rendering
    if getattr(cfg, "resume", None):
        which = str(cfg.resume)
        if which.lower() == "latest":
            trainer.load_latest_checkpoint()
        else:
            trainer.load_checkpoint(Path(which))

    # Render-only mode
    if getattr(cfg, "render_only", False):
        valr = ValidationRenderer(trainer=trainer, out_dir=out_dir / "render_only",
                                  tb_logger=(trainer.tb_logger if getattr(trainer, "tb_logger", None) else None))
        valr.render_camera_path_video(
            video_name="camera_path", overwrite=True
        )
        return

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
