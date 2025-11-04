"""
Ray samplers for training:
- RandomPixelRaySampler: uniform over all frames
- GoldRandomPixelRaySampler: vanilla NeRF style (single image per step + precrop)
"""

from __future__ import annotations

import random
from typing import Iterator, Dict, Optional

import torch
import torch.nn.functional as F

from nerf_sandbox.source.utils.render_utils import get_camera_rays


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
    convention : str
        Convention for ray generation. Ex. "opengl" or "colmap"
    as_ndc : bool, optional
        Whether to return rays in NDC. Defaults to False
    near_plane : float, optional
        The near plane to use when generating camera rays. defaults to 1.0

    """

    def __init__(
        self,
        scene,
        rays_per_batch: int = 2048,
        device: str | torch.device | None = None,
        white_bg_composite: Optional[bool] = None,
        cache_images_on_device: bool = False,
        sample_from_single_frame: bool = False,
        precrop_iters: int = 0,
        precrop_frac: float = 0.5,
        convention: str = "opengl",
        as_ndc: bool = False,
        near_plane: float = 1.0
    ) -> None:

        self.scene = scene
        self.B = int(rays_per_batch)
        self.device = torch.device(device) if device is not None else None
        self.white_bkgd = scene.white_bkgd if white_bg_composite is None else bool(white_bg_composite)
        self.convention = convention
        self.as_ndc = as_ndc
        self.near_plane = near_plane

        self.sample_from_single_frame = bool(sample_from_single_frame)
        self.precrop_iters = int(precrop_iters)
        self.precrop_frac = float(precrop_frac)
        self._global_step = 0

        # --- Where should images live? ---
        on_cuda = (self.device is not None and self.device.type == "cuda")
        self.cache = bool(cache_images_on_device) and on_cuda
        img_device = (self.device if self.cache else torch.device("cpu"))
        self._imgs_on_gpu = (img_device.type != "cpu")  # used in __iter__

        # Cache frame tensors
        self._Ks, self._c2ws, self._imgs = [], [], []
        for f in self.scene.frames:
            # tiny → keep with model device
            self._Ks.append(torch.as_tensor(f.K,   dtype=torch.float32, device=self.device))
            self._c2ws.append(torch.as_tensor(f.c2w, dtype=torch.float32, device=self.device))

            # big → follow the cache policy (GPU or CPU pinned)
            img_t = torch.as_tensor(f.image, dtype=torch.float32, device=img_device)
            if img_t.device.type == "cpu":
                try:
                    img_t = img_t.pin_memory()
                except Exception:
                    pass
            self._imgs.append(img_t)

        self.H = int(self.scene.frames[0].image.shape[0])
        self.W = int(self.scene.frames[0].image.shape[1])

    def _current_crop_bounds(self) -> tuple[int, int, int, int]:
        H, W = self.H, self.W
        if self._global_step < self.precrop_iters and 0.0 < self.precrop_frac < 1.0:
            f = self.precrop_frac
            h0 = int(H * 0.5 * (1.0 - f)); h1 = int(H * 0.5 * (1.0 + f))
            w0 = int(W * 0.5 * (1.0 - f)); w1 = int(W * 0.5 * (1.0 + f))
        else:
            h0, h1, w0, w1 = 0, H, 0, W
        return h0, h1, w0, w1

    def _composite_rgb(self, pix: torch.Tensor) -> torch.Tensor:
        if self.white_bkgd and pix.shape[-1] == 4:
            return pix[..., :3] * pix[..., 3:4] + (1.0 - pix[..., 3:4])
        return pix[..., :3]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Device-safe iterator:
        • Index pixels on the image's device (CPU if not cached, GPU if cached)
        • Move gathered RGB (and xy) to self.device
        • Build rays on self.device with get_camera_rays
        """
        num_frames = len(self._imgs)

        while True:
            h0, h1, w0, w1 = self._current_crop_bounds()

            if self.sample_from_single_frame:
                # ---------- Single-frame mode ----------
                fid = random.randrange(num_frames)

                # Per-frame tensors
                K   = self._Ks[fid]    # expected on self.device
                c2w = self._c2ws[fid]  # expected on self.device
                img = self._imgs[fid]  # may be CPU (not cached) or GPU (cached)
                src_dev = img.device

                # Sample indices on the *image's* device and gather pixels there
                ys_src = torch.randint(h0, h1, (self.B,), device=src_dev)
                xs_src = torch.randint(w0, w1, (self.B,), device=src_dev)
                pix    = img[ys_src.long(), xs_src.long()]  # (B, 3 or 4) on src_dev

                # Normalize to [0,1] if needed, composite RGBA→RGB if needed
                if pix.dtype.is_floating_point and float(pix.max().detach().cpu()) > 1.5:
                    pix = pix / 255.0
                rgb = self._composite_rgb(pix)  # stays on src_dev

                # Move gathered data to working device for ray generation / training
                if rgb.device != self.device:
                    rgb = rgb.to(self.device, non_blocking=True)
                xs = xs_src.to(self.device, non_blocking=True)
                ys = ys_src.to(self.device, non_blocking=True)
                pixels_xy = torch.stack([xs.to(torch.float32), ys.to(torch.float32)], dim=-1)

                # Rays on self.device
                (
                    rays_o_world,
                    rays_d_world_unit,
                    rays_d_world_norm,
                    rays_o_marching,
                    rays_d_marching_unit,
                    rays_d_marching_norm,
                ) = get_camera_rays(
                    self.H, self.W, K, c2w,
                    device=self.device, dtype=torch.float32,
                    convention=self.convention,
                    pixel_center=True,
                    as_ndc=self.as_ndc, near_plane=self.near_plane,
                    pixels_xy=pixels_xy,
                )

                # if self.as_ndc:
                #     assert torch.allclose(rays_d_marching_norm, torch.ones_like(rays_d_marching_norm))

                batch = {
                    "rgb":                  rgb,
                    "rays_o_world":         rays_o_world,
                    "rays_d_world_unit":    rays_d_world_unit,
                    "rays_d_world_norm":    rays_d_world_norm,
                    "rays_o_marching":      rays_o_marching,
                    "rays_d_marching_unit": rays_d_marching_unit,
                    "rays_d_marching_norm": rays_d_marching_norm,
                }

            else:
                # ---------- Mixed-frames mode ----------
                # Sample once (CPU is fine), then fan out per-fid
                ys_all   = torch.randint(h0, h1, (self.B,), device="cpu")
                xs_all   = torch.randint(w0, w1, (self.B,), device="cpu")
                fids_all = torch.randint(0, num_frames, (self.B,), device="cpu")
                uniq_fids = torch.unique(fids_all).tolist()

                rgb_list = []
                rays_o_world_list, rays_d_world_unit_list, rays_d_world_norm_list = [], [], []
                rays_o_marching_list, rays_d_marching_unit_list, rays_d_marching_norm_list = [], [], []

                for fid in uniq_fids:
                    mask = (fids_all == fid)
                    if not bool(mask.any()):
                        continue

                    # Slice indices for this frame
                    xs_f_cpu = xs_all[mask]
                    ys_f_cpu = ys_all[mask]

                    # Per-frame tensors
                    K   = self._Ks[fid]    # on self.device
                    c2w = self._c2ws[fid]  # on self.device
                    img = self._imgs[fid]  # CPU or GPU
                    src_dev = img.device

                    # Index on the image's device
                    xs_src = xs_f_cpu.to(src_dev, non_blocking=True)
                    ys_src = ys_f_cpu.to(src_dev, non_blocking=True)
                    pix    = img[ys_src.long(), xs_src.long()]  # (Bi, 3 or 4) on src_dev

                    if pix.dtype.is_floating_point and float(pix.max().detach().cpu()) > 1.5:
                        pix = pix / 255.0
                    rgb = self._composite_rgb(pix)  # on src_dev
                    if rgb.device != self.device:
                        rgb = rgb.to(self.device, non_blocking=True)
                    rgb_list.append(rgb)

                    # Build rays for these pixels on self.device
                    px = torch.stack([
                        xs_f_cpu.to(self.device, non_blocking=True, dtype=torch.float32),
                        ys_f_cpu.to(self.device, non_blocking=True, dtype=torch.float32)
                    ], dim=-1)

                    (
                        rays_o_world,
                        rays_d_world_unit,
                        rays_d_world_norm,
                        rays_o_marching,
                        rays_d_marching_unit,
                        rays_d_marching_norm,
                    ) = get_camera_rays(
                        self.H, self.W, K, c2w,
                        device=self.device, dtype=torch.float32,
                        convention=self.convention,
                        pixel_center=True,
                        as_ndc=self.as_ndc, near_plane=self.near_plane,
                        pixels_xy=px,
                    )

                    # if self.as_ndc:
                    #     assert torch.allclose(rays_d_marching_norm, torch.ones_like(rays_d_marching_norm))

                    rays_o_world_list.append(rays_o_world)
                    rays_d_world_unit_list.append(rays_d_world_unit)
                    rays_d_world_norm_list.append(rays_d_world_norm)
                    rays_o_marching_list.append(rays_o_marching)
                    rays_d_marching_unit_list.append(rays_d_marching_unit)
                    rays_d_marching_norm_list.append(rays_d_marching_norm)

                batch = {
                    "rgb":                  torch.cat(rgb_list, dim=0),
                    "rays_o_world":         torch.cat(rays_o_world_list,   dim=0),
                    "rays_d_world_unit":    torch.cat(rays_d_world_unit_list,  dim=0),
                    "rays_d_world_norm":    torch.cat(rays_d_world_norm_list,  dim=0),
                    "rays_o_marching":      torch.cat(rays_o_marching_list,   dim=0),
                    "rays_d_marching_unit": torch.cat(rays_d_marching_unit_list,  dim=0),
                    "rays_d_marching_norm": torch.cat(rays_d_marching_norm_list,  dim=0),
                }

            # Sanity: marching norms finite/positive (NDC => ~1; non-NDC => you set to 1s)
            if (self._global_step % 200) == 0:
                rn = batch["rays_d_marching_norm"]
                assert torch.isfinite(rn).all() and (rn > 0).all(), "Invalid rays_d_marching_norm!"

            self._global_step += 1
            yield batch

