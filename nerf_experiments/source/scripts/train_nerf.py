"""

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any

from nerf_experiments.source.config.config_utils import load_yaml_config, parse_nerf_config
from nerf_experiments.source.config.runtime_config import to_runtime_train_config
from nerf_experiments.source.train.trainer import Trainer


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

    # validation controls
    ap.add_argument("--val_every_steps", type=str, default=None,
                    help="Override train.val_every (accepts 1k, 500, etc).")
    ap.add_argument("--val_frame_idx", type=int, default=0,
                    help="Which frame index to render during validation previews.")

    # Render novel views
    ap.add_argument("--render_path", action="store_true", help="Render a novel camera path after training.")
    ap.add_argument("--path_type", type=str, default="circle", choices=["circle", "spiral"])
    ap.add_argument("--path_frames", type=int, default=120)
    ap.add_argument("--path_fps", type=int, default=30)
    ap.add_argument("--path_radius", type=float, default=None)
    ap.add_argument("--path_elev_deg", type=float, default=0.0)
    ap.add_argument("--path_yaw_deg", type=float, default=360.0)
    ap.add_argument("--path_res_scale", type=float, default=1.0)

    args = ap.parse_args()

    # 1) Load YAML → sectioned config
    ydict: Dict[str, Any] = load_yaml_config(args.config)
    nerf_cfg = parse_nerf_config(ydict)

    # 2) Compose output directory
    out_dir = str(Path(args.out_root) / args.exp_name)

    # 3) CLI overrides (dot-paths)
    overrides: Dict[str, Any] = {"train.out_dir": out_dir}
    if args.data_root is not None:        overrides["data.root"] = args.data_root
    if args.device is not None:           overrides["train.device"] = args.device
    if args.seed is not None:             overrides["seed"] = args.seed
    if args.max_steps is not None:        overrides["train.max_steps"] = args.max_steps
    if args.rays_per_batch is not None:   overrides["train.rays_per_batch"] = args.rays_per_batch
    if args.val_every_steps is not None:  overrides["train.val_every"] = args.val_every_steps

    # 4) Freeze → runtime config; save resolved + original in out_dir
    rt_cfg = to_runtime_train_config(
        nerf_cfg,
        cli_overrides=overrides,
        save_dir=out_dir,
        original_yaml=ydict,
    )

    # 5) Train
    trainer = Trainer(rt_cfg)
    # NEW: set which frame to validate (Trainer uses this attr if present)
    setattr(trainer, "val_frame_idx", int(args.val_frame_idx))
    trainer.train()

    # 6) Render video
    if args.render_path:
        path_out = Path(out_dir) / "render_path"
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
