# NeRF (vanilla-compatible) — with validation tools, progress video, and robust training

This project is a refactor of a NeRF training pipeline with a focus on **exact vanilla NeRF behavior** (bmild/nerf) when `--vanilla` is set, plus quality-of-life features for long runs: auto-resume, staged validation rendering (including a *training progress video* built during training), GPU thermal safeguards, and controllable signal handling.

The code assumes the **Blender synthetic** dataset layout and reads image paths *exactly* from `transforms_{split}.json` (e.g., `"./train/r_0"` → `<root>/train/r_0.png`). Camera ray generation and pose utilities are **convention-aware** and match the dataset’s OpenGL-style transforms.

---

## Preview (YouTube)


<iframe width="560" height="315" src="https://youtube.com/shorts/a5gZcdhjUwg?si=YDkZmCX-dQ5efv_Y" title="NeRF training progress" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


---

## Features

- **Vanilla NeRF parity** (`--vanilla`)
  - 8×256 MLP with skip, positional encoders (Lx=10 / Ld=4), ReLU σ with raw noise, white background, bmild sampling (1024 rays, center-precrop warmup), etc.
- **Validation & visualization**
  - Select per-view validation via `--val_indices`.
  - Save **RGB / opacity / depth** per validation step.
  - **Training progress video**: one smooth camera path is precomputed and **rendered block-by-block during training**, matched to a user-defined validation schedule (e.g., dense early via power law).
  - Camera path generation is **convention-aware** and supports **circle** and **spiral**.
- **Schedules that match how models learn**
  - Validation events can follow a **power schedule** (dense early, sparse later) while the progress video keeps a **constant angular speed**.
- **Auto-resume & robust checkpoints**
  - Resume from latest or explicit checkpoint; validation plan/progress state is restored so progress video continues where it left off.
- **GPU thermal guard**
  - Periodically checks temps and throttles with micro-batches / sleeps to avoid thermal runaway. Logs to TensorBoard when enabled.
- **Controllable signal handling**
  - `Ctrl+C` (SIGINT): save an interrupt checkpoint and **exit cleanly** (no post-rendering).
  - Optional pause hooks if desired (e.g., SIGUSR1).
- **TensorBoard logging**
  - Scalars + images (RGB/opacity/depth). Works across resumes (same log dir).

---

## Quick start

### 1) Dataset

Use the Blender synthetic dataset (e.g., **lego**). Point `--data_root` to the directory containing:

- `transforms_train.json`
- `transforms_val.json` (or `transforms_test.json`)
- `<split>/<frames>.png`

File paths in the JSON are used **as-is**, with “.png” appended.

### 2) Train on LEGO with vanilla settings

# from the repo root
python nerf_experiments/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --max_steps 30000 \
  --ckpt_every 5000 \
  --use_tb \
  --num_val_steps 50 \
  --val_schedule power --val_power 2.5 \
  --val_indices 0,3,11 \
  --progress_video_during_training --progress_frames 300 \
  --render_path_after --path_frames 300 --path_type spiral

Notes:
- If you hit OOM during validation, try `--eval_chunk 2048` (or 1024) and/or `--val_res_scale 0.5`.
- CUDA fragmentation? The trainer sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically.

### 3) Resuming training

**Auto-resume (find latest checkpoint in the run directory):**

python nerf_experiments/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --auto_resume \
  --use_tb

**Or resume from a specific file:**

python nerf_experiments/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --resume_path /path/to/exp/lego_vanilla/checkpoints/ckpt_0005000.pt \
  --use_tb

On resume, the **validation schedule** and **progress-video state** are reloaded so the next validation block and progress frames continue exactly where they should.

---

## CLI tips

Show all options:

python nerf_experiments/source/scripts/train_nerf.py --help

Common knobs:
- `--rays_per_batch`, `--nc`, `--nf`, `--raw_noise_std`, `--sigma_activation`
- `--eval_chunk`, `--val_res_scale`
- `--num_val_steps`, `--val_schedule {uniform,power}`, `--val_power`
- `--val_indices 0,3,11` (skip the default “single view” behavior)
- `--progress_video_during_training --progress_frames 300`
- `--render_path_after --path_frames 300 --path_type {circle,spiral}`

---

## What “vanilla” flips under the hood

- Encoders: Lx=10 (pos), Ld=4 (dir); include original inputs.
- Model: 8×256 with mid-skip; ReLU σ, raw σ noise during training.
- Sampling: N_rand=1024 from a single image per step, **center precrop** warmup.
- Background: white; near/far default `[2.0, 6.0]`.
- Learning rate: `5e-4` (cosine/other schedulers optional).

---

## Acknowledgements

- Original NeRF: https://github.com/bmild/nerf
