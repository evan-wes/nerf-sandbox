"""
Minimal training / render script for the refactored Trainer.

Adds:
  --render_only            -> skip training; just load a checkpoint and render a camera path
  --resume {latest|PATH}   -> choose which checkpoint to load (works with/without --render_only)
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

from nerf_sandbox.source.train.trainer import Trainer
from nerf_sandbox.source.utils.validation_renderer import ValidationRenderer


def make_cfg_from_args(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace(**vars(args))

    # One-switch: bmild/vanilla parity
    if args.vanilla:
        cfg.gold_nerf = True
        cfg.white_bkgd = True
        cfg.rays_per_batch = 1024
        cfg.precrop_iters = 500
        cfg.precrop_frac = 0.5
        cfg.nc = 64
        cfg.nf = 128
        cfg.det_fine = False
        cfg.pos_num_freqs = 10
        cfg.dir_num_freqs = 4
        cfg.pos_include_input = True
        cfg.dir_include_input = True
        cfg.n_layers = 8
        cfg.hidden_dim = 256
        cfg.skip_pos = 3
        cfg.sigma_activation = "relu"
        cfg.raw_noise_std = 1.0
        cfg.near = 2.0
        cfg.far = 6.0
        cfg.lr = 5e-4
    else:
        cfg.gold_nerf = False

    # map --resume into existing fields
    if args.resume:
        if args.resume.lower() in ("latest", "auto"):
            cfg.auto_resume = True
            cfg.resume_path = None
        else:
            cfg.auto_resume = False
            cfg.resume_path = args.resume

    # Ensure eval defaults follow train unless explicitly set
    cfg.eval_nc = args.eval_nc
    cfg.eval_nf = args.eval_nf
    cfg.eval_chunk = int(args.eval_chunk)
    cfg.val_res_scale = float(args.val_res_scale)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Refactored NeRF Trainer / Renderer")

    # Data
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--white_bkgd", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--scene_scale", type=float, default=1.0)
    ap.add_argument("--center_origin", action="store_true")
    ap.add_argument("--composite_on_load", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=6.0)

    # Output / Runtime
    ap.add_argument("--out_dir", type=str, default="./exp")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)

    # Model / Encoders
    ap.add_argument("--pos_num_freqs", type=int, default=10)
    ap.add_argument("--dir_num_freqs", type=int, default=4)
    ap.add_argument("--pos_include_input", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--dir_include_input", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--skip_pos", type=int, default=5)
    ap.add_argument("--initial_acc_opacity", type=float, default=0.0)
    ap.add_argument("--sigma_activation", type=str, default="softplus", choices=["relu","softplus"])

    # Sampling / Density
    ap.add_argument("--rays_per_batch", type=int, default=2048)
    ap.add_argument("--precrop_iters", type=int, default=500)
    ap.add_argument("--precrop_frac", type=float, default=0.5)
    ap.add_argument("--nc", type=int, default=64)
    ap.add_argument("--nf", type=int, default=128)
    ap.add_argument("--det_fine", action="store_true")
    ap.add_argument("--raw_noise_std", type=float, default=0.0)
    ap.add_argument("--targets_are_srgb", type=lambda x: str(x).lower() in ("1","true","yes"), default=True)

    # Optim / Schedule
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--scheduler", type=str, default="none")
    ap.add_argument("--scheduler_params", type=str, default="{}")

    # Train loop
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--micro_chunks", type=int, default=0)
    ap.add_argument("--grad_clip_norm", type=float, default=0.0)

    # Thermal (optional)
    ap.add_argument("--thermal_throttle", action="store_true")
    ap.add_argument("--gpu_temp_threshold", type=int, default=85)
    ap.add_argument("--gpu_temp_check_every", type=int, default=20)
    ap.add_argument("--gpu_cooldown_seconds", type=int, default=45)
    ap.add_argument("--thermal_throttle_max_micro", type=int, default=16)
    ap.add_argument("--thermal_throttle_sleep", type=float, default=5.0)

    # Resume / Logging
    ap.add_argument("--auto_resume", action="store_true",
                    help="If true, resume from newest checkpoint in out_dir/checkpoints.")
    ap.add_argument("--resume_path", type=str, default=None,
                    help="Path to a specific checkpoint to resume from.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Convenience: 'latest' or a checkpoint path. Overrides --auto_resume/--resume_path.")
    ap.add_argument("--resume_no_optim", action="store_true")
    ap.add_argument("--on_interrupt", type=str, default="render_and_exit")
    ap.add_argument("--on_pause_signal", type=str, default="render")
    ap.add_argument("--use_tb", action="store_true")
    ap.add_argument("--tb_logdir", type=str, default=None)
    ap.add_argument("--tb_image_max_side", type=int, default=512)

    # Path rendering
    ap.add_argument("--render_path_after", action="store_true")
    ap.add_argument("--path_frames", type=int, default=120)
    ap.add_argument("--path_fps", type=int, default=30)
    ap.add_argument("--path_type", type=str, default="circle", choices=["circle", "spiral"])
    ap.add_argument("--path_radius", type=float, default=None)
    ap.add_argument("--path_elev_deg", type=float, default=0.0)
    ap.add_argument("--path_yaw_deg", type=float, default=360.0)
    ap.add_argument("--path_res_scale", type=float, default=1.0)

    ap.add_argument("--val_every", type=int, default=None)
    ap.add_argument("--val_indices", type=str, default=None,
                    help="Comma-separated validation frame indices, e.g. '0,8,17'")
    ap.add_argument("--val_files", type=str, default=None,
                    help="Comma-separated filenames/stems from transforms JSON, e.g. 'r_0,val/r_5'")
    ap.add_argument("--progress_video_during_training", action="store_true")
    ap.add_argument("--progress_frames", type=int, default=240)
    ap.add_argument("--val_schedule", type=str, default="power",
                    help="Validation cadence strategy; 'power' concentrates renders early.")
    ap.add_argument("--num_val_steps", type=int, default=None,
                    help="Total number of validation events; if None, derived from --val_every.")
    ap.add_argument("--val_power", type=float, default=2.0,
                    help="Exponent for 'power' schedule (>1 → denser early).")

    # --- Eval/validation memory controls ---
    ap.add_argument("--eval_chunk", type=int, default=2048, help="Rays per eval chunk.")
    ap.add_argument("--eval_nc", type=int, default=None, help="Eval coarse samples per ray (default nc).")
    ap.add_argument("--eval_nf", type=int, default=None, help="Eval fine samples per ray (default nf).")
    ap.add_argument("--val_res_scale", type=float, default=1.0, help="Scale validation resolution to save VRAM.")

    # One-switch: vanilla bmild parity
    ap.add_argument("--vanilla", action="store_true",
                    help="Enable bmild/nerf parity with standard Blender defaults.")

    # ----- Render-only convenience -----
    ap.add_argument("--render_only", action="store_true",
                    help="Skip training, load a checkpoint (or latest), and render.")
    ap.add_argument("--render_ckpt", type=str, default=None,
                    help="Path to checkpoint to load in render-only mode; if omitted, load latest.")
    ap.add_argument("--render_path", action="store_true",
                    help="Also render a final camera-path video in render-only mode.")

    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    cfg = make_cfg_from_args(args)

    trainer = Trainer(cfg=cfg)

    # -------------------------
    # Render-only mode
    # -------------------------
    if args.render_only:
        # Load requested checkpoint or latest available
        ckpt_path = None
        if args.render_ckpt:
            ckpt_path = Path(args.render_ckpt)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--render_ckpt not found: {ckpt_path}")
        else:
            ckpt_path = trainer.find_latest_checkpoint()
            if ckpt_path is None:
                raise FileNotFoundError("No checkpoint found for render-only mode.")
        trainer.load_checkpoint(ckpt_path, load_optim=False)

        # Spin up a ValidationRenderer bound to this trainer
        vr = ValidationRenderer(trainer, out_dir=Path(cfg.out_dir) / "validation", tb_logger=None)

        # Parse indices (defaults to [0])
        if args.val_indices:
            idxs = [int(tok) for tok in args.val_indices.replace(",", " ").split()]
        else:
            idxs = [0]

        # Render static validation frames (RGB/opacity/depth) with the final model
        vr.render_selected_frames(frame_indices=idxs, res_scale=float(cfg.val_res_scale),
                                  log_to_tb=False, tb_step=None)

        # Optional: final camera-path video (smooth circle/spiral)
        if args.render_path or args.render_path_after:
            vr.render_final_camera_path(
                n_frames=int(cfg.path_frames),
                fps=int(cfg.path_fps),
                path_type=str(cfg.path_type),
                radius=cfg.path_radius,
                elevation_deg=float(cfg.path_elev_deg),
                yaw_range_deg=float(cfg.path_yaw_deg),
                res_scale=float(cfg.path_res_scale),
                world_up=None,            # auto-detect from validation scene
                convention="opengl",      # must match get_camera_rays
                out_subdir="final_path",
                basename="camera_path",
            )
        return

    # -------------------------
    # Normal training
    # -------------------------
    trainer.train()

if __name__ == "__main__":
    main()
