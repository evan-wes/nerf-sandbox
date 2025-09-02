"""
Contains the RayBatch, RaySampler and RandomPixelRaySampler classes
for sampling ray batches from images

Key abstractions:
- RaySampler: yields training batches of rays and ground-truth targets
- Renderer: interface for coarse→fine NeRF rendering (implementation later)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import numpy as np
import torch

from nerf_experiments.source.data.scene import Frame, Scene
from nerf_experiments.source.utils.ray_utils import get_camera_rays


class RayBatch(Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]):
    """Type hint: (rays_o [B,3], rays_d [B,3], rgb [B,3], mask [B,1] or None)."""
    pass

class RaySampler:
    """Abstract ray sampler.

    Implementations yield dictionaries or tuples containing per-ray batches.
    """
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def next_batch(self) -> Dict[str, torch.Tensor]:
        """Optionally provide an explicit pull API."""
        return next(iter(self))


class RandomPixelRaySampler(RaySampler):
    """Uniformly sample rays by picking random (image, y, x) coordinates.

    This is robust across datasets and easy to extend with masks/weights.
    """
    def __init__(
        self,
        scene: Scene,
        rays_per_batch: int = 2048,
        device: str | torch.device | None = None,
        white_bg_composite: bool | None = None,
        cache_images_on_device: bool = False,
    ):
        self.scene = scene.to_torch(device)
        self.B = int(rays_per_batch)
        self.device = torch.device(device) if device is not None else None
        self.white_bg = self.scene.white_bg if white_bg_composite is None else white_bg_composite
        self.cache_images_on_device = cache_images_on_device

        # Cache per-frame tensors on device for fast indexing
        self._Ks = []
        self._c2ws = []
        self._imgs = []
        for f in self.scene.frames:
            # Ensure float32 torch tensors on device
            Kt = torch.as_tensor(f.K, dtype=torch.float32, device=self.device)
            Tt = torch.as_tensor(f.c2w, dtype=torch.float32, device=self.device)
            It = torch.as_tensor(f.image, dtype=torch.float32, device=self.device)  # [H,W,3|4] in [0,1]
            self._Ks.append(Kt)
            self._c2ws.append(Tt)
            self._imgs.append(It)

        self.H = self.scene.frames[0].image.shape[0]
        self.W = self.scene.frames[0].image.shape[1]

    def __iter__(self):
        rng = np.random.default_rng()
        H, W = self.H, self.W
        device = self.device

        while True:
            # 1) Sample B (frame, y, x) tuples
            img_idx = torch.from_numpy(rng.integers(0, len(self._imgs), size=self.B)).to(device)
            ys      = torch.from_numpy(rng.integers(0, H, size=self.B)).to(device)
            xs      = torch.from_numpy(rng.integers(0, W, size=self.B)).to(device)

            # 2) Group by unique frame to avoid per-pixel loops
            uniq_imgs, inverse = torch.unique(img_idx, sorted=False, return_inverse=True)

            rays_o_chunks = []
            rays_d_chunks = []
            rgb_chunks    = []

            for ui, img_id in enumerate(uniq_imgs.tolist()):
                # Select all pixels in this batch that come from the same frame
                mask = (inverse == ui)
                xs_i = xs[mask]
                ys_i = ys[mask]
                n_i  = xs_i.numel()
                if n_i == 0:
                    continue

                # Gather intrinsics, pose, and image for this frame
                K = self._Ks[img_id]      # [3,3]
                T = self._c2ws[img_id]    # [4,4] or [3,4]
                I = self._imgs[img_id]    # [H,W,3|4], float32 in [0,1]

                # 3) Build rays for all pixels of this frame in one call
                pixels_xy = torch.stack([xs_i.to(torch.float32), ys_i.to(torch.float32)], dim=-1)  # [n_i,2]
                rays_o_i, rays_d_i = get_camera_rays(
                    image_h=H,
                    image_w=W,
                    intrinsic_matrix=K,
                    transform_camera_to_world=T,
                    device=device,
                    dtype=torch.float32,
                    normalize_dirs=True,
                    as_ndc=False,
                    near_plane=1.0,
                    pixels_xy=pixels_xy,
                )  # each [n_i,3]

                # 4) Vectorized pixel gather (+ optional white background compositing)
                # Advanced indexing supports (y,x) lists directly
                pix = I[ys_i.long(), xs_i.long()]  # [n_i, C], C=3 or 4
                if self.white_bg and pix.shape[-1] == 4:
                    rgb_i = pix[..., :3] * pix[..., 3:4] + (1.0 - pix[..., 3:4])
                else:
                    rgb_i = pix[..., :3]

                rays_o_chunks.append(rays_o_i)
                rays_d_chunks.append(rays_d_i)
                rgb_chunks.append(rgb_i)

            # 5) Concatenate chunks back to [B,3] in the original sample order
            # We built per-unique-frame chunks in arbitrary order; reassemble to match (img_idx, xs, ys) order
            # Build an output buffer and fill it:
            rays_o = torch.empty(self.B, 3, dtype=torch.float32, device=device)
            rays_d = torch.empty(self.B, 3, dtype=torch.float32, device=device)
            rgbs   = torch.empty(self.B, 3, dtype=torch.float32, device=device)

            # Compute start offsets per unique group to place chunks contiguously
            # Build an index list for each group
            group_positions = [torch.nonzero(inverse == g, as_tuple=False).flatten() for g in range(len(uniq_imgs))]
            # Now scatter each chunk into its positions
            for pos, ro, rd, c in zip(group_positions, rays_o_chunks, rays_d_chunks, rgb_chunks):
                rays_o[pos] = ro
                rays_d[pos] = rd
                rgbs[pos]   = c

            yield {"rays_o": rays_o, "rays_d": rays_d, "rgb": rgbs}
