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
        device: Optional[Union[str, torch.device]] = None,
        white_bg_composite: Optional[bool] = None,
        cache_images_on_device: bool = False,
    ):
        self.scene = scene.to_torch(device)
        self.B = int(rays_per_batch)
        self.device = torch.device(device) if device is not None else None
        self.white_bg = self.scene.white_bg if white_bg_composite is None else white_bg_composite
        self.cache_images_on_device = cache_images_on_device

        # Optional: pack images to a single tensor for faster indexing
        imgs = []
        for f in self.scene.frames:
            img = f.image
            if isinstance(img, torch.Tensor):
                t = img
            else:
                t = torch.as_tensor(img)
            if self.device is not None:
                t = t.to(self.device, non_blocking=True)
            imgs.append(t)
        self.images = imgs  # list of [H,W,3|4]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        H, W = self.scene.H, self.scene.W
        rng = np.random.default_rng()
        while True:
            idx = rng.integers(0, len(self.scene), size=self.B)
            ys = rng.integers(0, H, size=self.B)
            xs = rng.integers(0, W, size=self.B)

            # Gather GT
            rgbs = []
            rays_o = []
            rays_d = []
            for k in range(self.B):
                f = self.scene.frames[int(idx[k])]
                x, y = int(xs[k]), int(ys[k])
                # RGB(A)
                pix = self.images[int(idx[k])][y, x]  # [3|4]
                if pix.shape[-1] == 4 and self.white_bg:
                    rgb = pix[:3] * pix[3] + (1.0 - pix[3])
                else:
                    rgb = pix[..., :3]
                rgbs.append(rgb)

                # Rays for this pixel (build tiny batch of 1)
                o, d = get_camera_rays(
                    image_h=H,
                    image_w=W,
                    intrinsic_matrix=f.K,
                    transform_camera_to_world=f.c2w,
                    device=pix.device,
                    normalize_dirs=True,
                    pixels_xy=torch.tensor([x, y], device=pix.device)
                )
                rays_o.append(o[0])
                rays_d.append(d[0])

            batch = {
                "rays_o": torch.stack(rays_o, dim=0).to(self.device),  # [B,3]
                "rays_d": torch.stack(rays_d, dim=0).to(self.device),  # [B,3]
                "rgb": torch.stack(rgbs,   dim=0).to(self.device),     # [B,3]
            }
            yield batch