# A Neural Radiance Field Sandbox

This project contains a PyTorch implementation of a Neural Radiance Field (NeRF) model with the aim of expandability to more advanced architectures and training paradigms. The initial version supports the [original paper's architecture](https://www.matthewtancik.com/nerf).

The custom Trainer class includes a wide suite of quality-of-life features such as:

- Auto-resume functionality to pause training and restart from the latest or a specific checkpoint
- GPU thermal safeguards helpful for laptops
- Signal handling to gracefully quit if training is interupted with Ctrl+C
- Comprehensive logging via TensorBoard, with loss, metrics, and validation images

As the intent of this project is to learn more about neural radiance fields and how they work, many visualization features have been added, including:

- Rendering of RGB, depth, and opacity as the model learns the scene
- Comprehensive validation scheduling to generate *training progress videos* that can be linear or power-based to generate more frames at earlier stages.
  - An example can be found here as a [YouTube Short](https://youtube.com/shorts/a5gZcdhjUwg?si=MgS1kwodNKym28jo)
- Camera path rendering for novel scene generation, both during training and after


## Features

- **Vanilla NeRF parity** (`--vanilla`)
  - 8×256 MLP with a skip connection, positional encoders (Lx=10 / Ld=4), ReLU σ with raw noise, white background, sampling (1024 rays, center-precrop warmup), etc.
- **Validation & visualization**
  - Select per-view validation via `--val_indices`.
  - Save **RGB / opacity / depth** per validation step.
  - **Training progress video**: one smooth camera path is precomputed and **rendered block-by-block during training**, matched to a user-defined validation schedule (e.g., dense early via power law).
  - Camera path generation is **convention-aware** and supports **circle** and **spiral**.
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

#### from the repo root
```
$ python nerf_experiments/source/scripts/train_nerf.py \
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
```
Notes:
- If you hit OOM during validation, try `--eval_chunk 2048` (or 1024) and/or `--val_res_scale 0.5`.
- CUDA fragmentation? The trainer sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically.

### 3) Resuming training

**Auto-resume (find latest checkpoint in the run directory):**
```
python nerf_experiments/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --auto_resume \
  --use_tb
```

**Or resume from a specific file:**
```
python nerf_experiments/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --resume_path /path/to/exp/lego_vanilla/checkpoints/ckpt_0005000.pt \
  --use_tb
```

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


**What `--vanilla` flips under the hood:**

- Encoders: Lx=10 (pos), Ld=4 (dir); include original inputs.
- Model: 8×256 with mid-skip; ReLU σ, raw σ noise during training.
- Sampling: N_rand=1024 from a single image per step, **center precrop** warmup.
- Background: white; near/far default `[2.0, 6.0]`.
- Learning rate: `5e-4` (cosine/other schedulers optional).

---

## Acknowledgements

- The "vanilla" mode is based off of the original NeRF paper: https://github.com/bmild/nerf
