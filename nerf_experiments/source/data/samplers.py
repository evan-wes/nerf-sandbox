"""
Ray samplers for training:
- RandomPixelRaySampler: uniform over all frames
- GoldRandomPixelRaySampler: vanilla NeRF style (single image per step + precrop)
"""

from __future__ import annotations

import random
from typing import Iterator, Dict, Optional

import torch

from nerf_experiments.source.utils.render_utils import get_camera_rays


# Minimal protocol Scene/Frame:
# scene.frames[i].image: np.ndarray (H,W,3 or 4) in [0,1]
# scene.frames[i].K: (3,3) float
# scene.frames[i].c2w: (4,4) float
# scene.white_bkgd: bool


class RaySampler:
    pass


class RandomPixelRaySampler(RaySampler):
    """
    Configurable pixel-ray sampler for NeRF training.

    Modes
    -----
    - Mixed-frames mode (default): sample pixels uniformly across all frames; each
      batch mixes multiple images.
    - Single-frame mode (vanilla NeRF style): sample all rays for the batch from
      a single randomly-chosen frame.

    Precrop
    -------
    Optionally apply a center crop during the first `precrop_iters` steps. This
    affects the candidate pixel region in either mode.

    Parameters
    ----------
    scene : Scene
        Scene containing frames with (image, K, c2w).
    rays_per_batch : int
        Number of rays/pixels per yielded batch.
    device : str | torch.device | None
        Device where cached tensors/rays are placed. If None, stays on CPU.
    white_bg_composite : bool | None
        If True, composite RGBA onto white on the fly; if None, uses scene.white_bkgd.
    cache_images_on_device : bool
        Kept for signature parity; images are turned into tensors on `device` either way.
    sample_from_single_frame : bool
        If True, sample a single random frame per batch (vanilla NeRF style).
        If False, mix frames within a batch uniformly (default).
    precrop_iters : int
        Number of initial iterations to use center precropping.
    precrop_frac : float
        Fraction of the image side length used for the center crop (0<frac<=1).
    """

    def __init__(self,
                 scene,
                 rays_per_batch: int = 2048,
                 device: str | torch.device | None = None,
                 white_bg_composite: Optional[bool] = None,
                 cache_images_on_device: bool = False,
                 sample_from_single_frame: bool = False,
                 precrop_iters: int = 0,
                 precrop_frac: float = 0.5) -> None:

        self.scene = scene
        self.B = int(rays_per_batch)
        self.device = torch.device(device) if device is not None else None
        self.white_bkgd = scene.white_bkgd if white_bg_composite is None else bool(white_bg_composite)
        self.cache = bool(cache_images_on_device)

        # Sampling behavior toggles
        self.sample_from_single_frame = bool(sample_from_single_frame)
        self.precrop_iters = int(precrop_iters)
        self.precrop_frac = float(precrop_frac)
        self._global_step = 0  # incremented per yielded batch

        # Cache per-frame tensors (images, intrinsics, extrinsics)
        self._Ks = []
        self._c2ws = []
        self._imgs = []
        for f in self.scene.frames:
            self._Ks.append(torch.as_tensor(f.K, dtype=torch.float32, device=self.device))
            self._c2ws.append(torch.as_tensor(f.c2w, dtype=torch.float32, device=self.device))
            img = torch.as_tensor(f.image, dtype=torch.float32, device=self.device)
            self._imgs.append(img)

        # Assume consistent resolution across frames
        self.H = self.scene.frames[0].image.shape[0]
        self.W = self.scene.frames[0].image.shape[1]

    def _current_crop_bounds(self) -> tuple[int, int, int, int]:
        """
        Compute (h0, h1, w0, w1) for the current iteration based on precropping.
        Full-frame if precrop_iters exhausted or disabled.
        """
        H, W = self.H, self.W
        if self._global_step < self.precrop_iters and 0.0 < self.precrop_frac < 1.0:
            f = self.precrop_frac
            h0 = int(H * 0.5 * (1.0 - f))
            h1 = int(H * 0.5 * (1.0 + f))
            w0 = int(W * 0.5 * (1.0 - f))
            w1 = int(W * 0.5 * (1.0 + f))
        else:
            h0, h1, w0, w1 = 0, H, 0, W
        return h0, h1, w0, w1

    def _composite_rgb(self, pix: torch.Tensor) -> torch.Tensor:
        """
        Composite RGBA onto white if requested; otherwise return RGB as-is.
        pix: (..., C) where C is 3 or 4.
        """
        if self.white_bkgd and pix.shape[-1] == 4:
            return pix[..., :3] * pix[..., 3:4] + (1.0 - pix[..., 3:4])
        return pix[..., :3]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        num_frames = len(self._imgs)

        while True:
            h0, h1, w0, w1 = self._current_crop_bounds()

            if self.sample_from_single_frame:
                # -------- Single-frame mode (vanilla NeRF) --------
                frame_id = random.randrange(num_frames)
                K = self._Ks[frame_id]
                c2w = self._c2ws[frame_id]
                img = self._imgs[frame_id]

                # Sample pixel coordinates within the (possibly cropped) region
                ys = torch.randint(h0, h1, (self.B,), device=self.device)
                xs = torch.randint(w0, w1, (self.B,), device=self.device)

                # Build rays for this frame
                pixels_xy = torch.stack([xs.to(torch.float32), ys.to(torch.float32)], dim=-1)
                rays_o, rays_d = get_camera_rays(self.H, self.W, K, c2w,
                                                 device=self.device, pixels_xy=pixels_xy)

                # Gather RGB targets (composite if RGBA and white_bkgd)
                pix = img[ys.long(), xs.long()]
                rgb = self._composite_rgb(pix)

                batch = {"rays_o": rays_o, "rays_d": rays_d, "rgb": rgb}

            else:
                # -------- Mixed-frames mode (uniform across all frames) --------
                # Sample coordinates & frame indices independently
                ys = torch.randint(h0, h1, (self.B,), device=self.device)
                xs = torch.randint(w0, w1, (self.B,), device=self.device)
                frame_ids = torch.randint(0, num_frames, (self.B,), device=self.device)

                rays_o_list = []
                rays_d_list = []
                rgb_list = []

                # Build rays per unique frame to avoid recomputing K/c2w per pixel
                for fid in frame_ids.unique().tolist():
                    mask = (frame_ids == fid)
                    K = self._Ks[fid]
                    c2w = self._c2ws[fid]
                    img = self._imgs[fid]

                    px = torch.stack([xs[mask].to(torch.float32), ys[mask].to(torch.float32)], dim=-1)
                    ro, rd = get_camera_rays(self.H, self.W, K, c2w,
                                             device=self.device, pixels_xy=px)

                    pix = img[ys[mask].long(), xs[mask].long()]
                    rgb = self._composite_rgb(pix)

                    rays_o_list.append(ro)
                    rays_d_list.append(rd)
                    rgb_list.append(rgb)

                batch = {
                    "rays_o": torch.cat(rays_o_list, dim=0),
                    "rays_d": torch.cat(rays_d_list, dim=0),
                    "rgb":   torch.cat(rgb_list,   dim=0),
                }

            self._global_step += 1
            yield batch
