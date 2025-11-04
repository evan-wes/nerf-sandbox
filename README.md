# A Neural Radiance Field Sandbox

This project contains a PyTorch implementation of a Neural Radiance Field (NeRF) model with the aim of expandability to more advanced architectures and training paradigms. The initial version supports the [original paper's architecture](https://www.matthewtancik.com/nerf).

The custom Trainer class includes a wide suite of quality-of-life features such as:

- Auto-resume functionality to pause training and restart from the latest or a specific checkpoint
- GPU thermal safeguards helpful for laptops
- Signal handling to gracefully quit if training is interupted with Ctrl+C
- Comprehensive logging via TensorBoard, with loss, metrics, and validation images

As the intent of this project is to learn more about neural radiance fields and how they work, many visualization features have been added, including:

- Camera path rendering for novel scene generation, both during training and after
- Rendering of RGB, depth, and opacity as the model learns the scene
- Comprehensive validation scheduling to generate *training progress videos* that can be linear or power-based to generate more frames at earlier stages.
  - An example for the Lego and Fern datasets are shown in this Gif:

![Demo](assets/training_progress_demo.gif)

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

### 2) Train on Lego or Fern with vanilla settings

Lego:
```
$ python3 nerf_sandbox/source/scripts/train_nerf.py \
  --data_kind blender \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir /path/to/lego_vanilla \
  --vanilla --device cuda:0 --seed 0 \
  --max_steps 50000 --log_every 100 --ckpt_every 10000 \
  --use_tb \
  --progress_video_during_training --progress_frames 150 --val_power 3 \
  --render_path_after
```

Fern:
```
$ python3 nerf_sandbox/source/scripts/train_nerf.py  \
  --data_kind llff \
  --data_root /path/to/nerf_llff_data/fern \
  --out_dir /path/to/fern_vanilla_final \
  --vanilla --device cuda:0 --seed 0 \
  --use_ndc --ndc_near_plane_world 1.0 \
  --downscale 8 \
  --max_steps 100000 --log_every 100 --ckpt_every 10000 \
  --use_tb \
  --progress_video_during_training --progress_frames 180 --val_power 3 \
  --render_path_after
```
Notes:
- If you hit OOM during validation, try `--eval_chunk 2048` (or 1024) and/or `--val_res_scale 0.5`.
- CUDA fragmentation? The trainer sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically.

### 3) Resuming training

**Auto-resume (find latest checkpoint in the run directory):**
```
python nerf_sandbox/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --auto_resume \
  --use_tb
```

**Or resume from a specific file:**
```
python nerf_sandbox/source/scripts/train_nerf.py \
  --data_root /path/to/nerf_synthetic/lego \
  --out_dir   /path/to/exp/lego_vanilla \
  --vanilla \
  --resume_path /path/to/exp/lego_vanilla/checkpoints/ckpt_0005000.pt \
  --use_tb
```

On resume, the **validation schedule** and **progress-video state** are reloaded so the next validation block and progress frames continue exactly where they should.

## Acknowledgements

- The "vanilla" mode is based off of the original NeRF paper: https://github.com/bmild/nerf
- PyTorch details and debugging of LLFF implementation was done with help of https://github.com/yenchenlin/nerf-pytorch with excellent blog post https://yconquesty.github.io/blog/ml/nerf/nerf_rendering.html