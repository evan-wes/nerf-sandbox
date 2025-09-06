"""

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Any

from nerf_experiments.source.config.config_utils import load_yaml_config, parse_nerf_config
from nerf_experiments.source.config.runtime_config import load_configs_with_extras


def main():
    ap = argparse.ArgumentParser("NeRF trainer")
    ap.add_argument("--config", type=str, required=True,
                    help="Path to YAML config (NerfConfig schema).")
    ap.add_argument("--data_root", type=str, default=None,
                    help="Override data.root from YAML (e.g. /local_dir/nerf_synthetic/lego).")
    ap.add_argument("--exp_name", type=str, default="exp",
                    help="Experiment name (used to create output dir).")
    ap.add_argument("--out_root", type=str, default="runs",
                    help="Root folder for outputs; final out_dir = out_root/exp_name.")
    ap.add_argument("--device", type=str, default=None,
                    help="Optional device override, e.g. cuda, cuda:0, or cpu.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional random seed override.")
    ap.add_argument("--max_steps", type=str, default=None,
                    help="Override train.max_steps (accepts 200k, 5k, etc).")
    ap.add_argument("--rays_per_batch", type=str, default=None,
                    help="Override train.rays_per_batch (accepts 4k, etc).")

    # Validation controls
    ap.add_argument("--val_every_steps", type=str, default=None,
                    help="Override train.val_every (accepts 1k, 500, etc).")
    ap.add_argument("--val_frame_idx", type=int, default=0,
                    help="Which frame index to render during validation previews.")
    ap.add_argument("--eval-chunk", type=int, default=None,
                    help="Rays per forward pass during validation/path rendering (default 8192). "
                         "Lower if you hit CUDA OOM while validating.")

    # Render novel views
    ap.add_argument("--render_path", action="store_true", help="Render a novel camera path after training.")
    ap.add_argument("--path_type", type=str, default="circle", choices=["circle", "spiral"])
    ap.add_argument("--path_frames", type=int, default=120)
    ap.add_argument("--path_fps", type=int, default=30)
    ap.add_argument("--path_radius", type=float, default=None)
    ap.add_argument("--path_elev_deg", type=float, default=0.0)
    ap.add_argument("--path_yaw_deg", type=float, default=360.0)
    ap.add_argument("--path_res_scale", type=float, default=1.0)

    # Memory / allocator controls
    ap.add_argument("--cuda-expandable-segments", action="store_true",
                    help="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing torch.")
    ap.add_argument("--micro-chunks", type=int, default=0,
                    help="If >0, split each training ray batch into this many micro-chunks with per-chunk backward.")
    ap.add_argument("--ckpt-mlp", action="store_true",
                    help="Enable gradient checkpointing around NeRF MLP forward to save activation memory.")

    # TensorBoard logging
    ap.add_argument("--tensorboard", action="store_true",
                    help="Enable TensorBoard logging of metrics and GPU vitals.")
    ap.add_argument("--tb-logdir", type=str, default=None,
                    help="Optional TensorBoard log directory. Defaults to <save_dir>/tb.")
    ap.add_argument("--tb-group-root", type=str, default=None,
                    help="Optional directory to collect multiple runs for comparison. "
                         "Creates a symlink <tb-group-root>/<run_name> -> <save_dir>/tb.")

    # Thermal safety controls
    ap.add_argument("--gpu-temp-threshold", type=int, default=85,
                    help="If GPU temp (°C) exceeds this, trigger thermal response. Default: 85.")
    ap.add_argument("--gpu-temp-check-every", type=int, default=20,
                    help="How many training steps between temperature checks. Default: 20.")
    ap.add_argument("--gpu-cooldown-seconds", type=int, default=45,
                    help="How long to sleep when too hot (if not using auto-throttle). Default: 45.")
    ap.add_argument("--thermal-throttle", action="store_true",
                    help="When hot, automatically increase micro-chunks up to a cap instead of sleeping.")
    ap.add_argument("--thermal-throttle-max-micro", type=int, default=16,
                    help="Upper bound for micro-chunks when --thermal-throttle is enabled. Default: 16.")
    ap.add_argument("--thermal-throttle-sleep", type=float, default=5.0,
                    help="Brief sleep (seconds) after applying throttle, to let temps settle. Default: 5.0.")

    # Validation video export controls
    ap.add_argument("--val-video-glob", type=str, default=None,
                    help="Glob (relative to the experiment folder) for validation frames, e.g. 'val/**/*.png'. "
                         "If omitted, defaults to a sensible pattern.")
    ap.add_argument("--val-video-out", type=str, default=None,
                    help="Output filename for the validation video (relative to experiment folder). "
                         "Defaults to 'val_video.mp4'.")
    ap.add_argument("--val-video-fps", type=int, default=24,
                    help="FPS for validation video (default: 24).")

    # Run control / resume
    ap.add_argument("--auto-resume", action="store_true",
                    help="If set, automatically resume from latest checkpoint in the experiment folder.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint .pt to resume from (overrides --auto-resume if both are set).")
    ap.add_argument("--resume-no-optim", action="store_true",
                    help="When resuming, load model weights but NOT optimizer/scaler (fresh optimizer).")
    ap.add_argument("--on-interrupt", choices=["none","save","render","render_and_exit"], default="render_and_exit",
                    help="Behavior on Ctrl+C (SIGINT): just ignore (none), save checkpoint only (save), "
                         "render videos only (render), or render and then exit (render_and_exit).")
    ap.add_argument("--on-pause-signal", choices=["render","save_and_render"], default="render",
                    help="Behavior on SIGUSR1 during training: render videos, or save checkpoint then render.")

    args = ap.parse_args()

    # Set CUDA allocator before any torch import
    if args.cuda_expandable_segments and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Delay importing Trainer (torch) until after env is set
    from nerf_experiments.source.train.trainer import Trainer
    from nerf_experiments.source.config.runtime_config import load_configs_with_extras

    # 1) Compose output directory (keep as Path)
    out_dir = Path(args.out_root) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) EXTENDED overrides (core + new sections -> YAML-style namespaces)
    overrides = {
        # core
        "data.root": args.data_root,
        "train.out_dir": str(out_dir),
        "train.device": args.device,
        "train.max_steps": args.max_steps,
        "train.rays_per_batch": args.rays_per_batch,
        "train.val_every": args.val_every_steps,
        "eval.eval_chunk": args.eval_chunk,
        # memory / perf
        "memory.micro_chunks": args.micro_chunks,
        "memory.ckpt_mlp": args.ckpt_mlps if hasattr(args, "ckpt_mlps") else args.ckpt_mlp,  # keep args.ckpt_mlp
        "memory.cuda_expandable_segments": args.cuda_expandable_segments,
        # logging / TB
        "logging.tensorboard": args.tensorboard,
        "logging.tb_logdir": args.tb_logdir,
        "logging.tb_group_root": getattr(args, "tb_group_root", None),
        # safety / thermals
        "safety.thermal_throttle": args.thermal_throttle,
        "safety.thermal_throttle_max_micro": args.thermal_throttle_max_micro,
        "safety.thermal_throttle_sleep": args.thermal_throttle_sleep,
        "safety.gpu_temp_threshold": args.gpu_temp_threshold,
        "safety.gpu_temp_check_every": args.gpu_temp_check_every,
        # path render controls
        "path.render_path": args.render_path,
        "path.path_type": args.path_type,
        "path.path_frames": args.path_frames,
        "path.path_fps": args.path_fps,
        "path.path_res_scale": args.path_res_scale,
        # validation video (if you wired these into extras; otherwise leave as direct setattrs later)
        "logging.val_video_glob": args.val_video_glob,
        "logging.val_video_out": args.val_video_out,
        "logging.val_video_fps": args.val_video_fps,
    }
    # prune unset values (None)
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # 3) Load frozen runtime config + EXTRAS (TB/thermals/eval/path/etc.)
    rt_cfg, extras, _ = load_configs_with_extras(
        yaml_path=args.config,
        cli_overrides=overrides,
        save_dir=out_dir,
    )

    # 4) Build trainer (no TB args in constructor)
    trainer = Trainer(cfg=rt_cfg)

    # 5) Apply EXTRAS from YAML/CLI namespaces
    for k, v in (extras or {}).items():
        if v is not None:
            setattr(trainer, k, v)

    # 6) CLI-only flags that are not part of extras
    setattr(trainer, "auto_resume", bool(args.auto_resume))
    setattr(trainer, "resume_path", args.resume)
    setattr(trainer, "resume_no_optim", bool(getattr(args, "resume_no_optim", False)))
    setattr(trainer, "on_interrupt", getattr(args, "on_interrupt", "render_and_exit"))
    setattr(trainer, "on_pause_signal", getattr(args, "on_pause_signal", "render"))
    setattr(trainer, "val_frame_idx", int(args.val_frame_idx))
    # (If val-video settings weren’t mapped into extras, set them directly:)
    if getattr(trainer, "val_video_glob", None) is None:
        setattr(trainer, "val_video_glob", args.val_video_glob)
    if getattr(trainer, "val_video_out", None) is None:
        setattr(trainer, "val_video_out", args.val_video_out)
    if getattr(trainer, "val_video_fps", None) is None:
        setattr(trainer, "val_video_fps", int(args.val_video_fps))

    # 7) Initialize TensorBoard AFTER attributes are final
    trainer.maybe_init_tensorboard()

    # 8) Train
    trainer.train()

    # 9) Optional: render novel path after training
    if args.render_path:
        path_out = out_dir / "render_path"
        trainer.render_camera_path(
            out_dir=path_out,
            n_frames=args.path_frames,
            fps=args.path_fps,
            path_type=args.path_type,
            radius=args.path_radius,
            elevation_deg=args.path_elev_deg,
            yaw_range_deg=args.path_yaw_deg,
            res_scale=args.path_res_scale,
            save_video=True,
        )


if __name__ == "__main__":
    main()
