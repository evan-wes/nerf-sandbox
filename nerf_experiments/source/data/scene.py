"""
Contains the Frame and Scene classes

Includes:
- Frame: immutable record of a single view (image, intrinsics, pose, etc.)
- Scene: a collection of Frames + scene-wide metadata
- RaySampler: yields training batches of rays and ground-truth targets
- Renderer: interface for coarse→fine NeRF rendering (implementation later)

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import numpy as np
import torch

from nerf_experiments.source.utils.torch_utils import to_torch

# -----------------------------
# Camera / Data Records
# -----------------------------

ArrayLike = Union[np.ndarray, torch.Tensor]

@dataclass(frozen=True)
class Frame:
    """A single calibrated view.

    Attributes:
        image:  HxWx{3|4} RGB(A). Can be uint8 in [0,255] or float in [0,1].
        K:      (3,3) pinhole intrinsics.
        c2w:    (4,4) or (3,4) camera-to-world transform.
        mask:   Optional HxW (bool/float) foreground mask.
        dist:   Optional distortion parameters (OpenCV order) or None if undistorted.
        meta:   Arbitrary per-frame metadata (exposure, timestamp, camera id, etc.).
    """
    image: ArrayLike
    K: ArrayLike
    c2w: ArrayLike
    mask: Optional[ArrayLike] = None
    dist: Optional[Dict[str, float]] = None
    meta: Dict[str, Union[float, int, str]] = field(default_factory=dict)

    @property
    def H(self) -> int:
        return int(self.image.shape[0])

    @property
    def W(self) -> int:
        return int(self.image.shape[1])

    def to_torch(self, device: Optional[Union[str, torch.device]] = None) -> "Frame":
        """Return a cloned Frame where arrays are torch.Tensors on device.
        This does NOT modify the original (dataclass is frozen/immutable).
        """
        dev = torch.device(device) if device is not None else None

        return replace(
            self,
            image=to_torch(x=self.image, device=dev),
            K=to_torch(x=self.K, device=dev),
            c2w=to_torch(x=self.c2w, device=dev),
            mask=None if self.mask is None else to_torch(x=self.mask, device=dev),
        )

@dataclass
class Scene:
    """A collection of Frames with scene-level metadata and normalization.

    Attributes:
        frames:   List of calibrated views.
        white_bkgd: Whether to composite on white background during rendering.
        aabb:     Optional axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax]
                  in world coordinates for normalization / near-far estimation.
        near:     Default near bound (world units). Can be overridden per-call.
        far:      Default far bound (world units). Can be overridden per-call.
        scale:    Global scale factor applied to world coordinates (for normalization).
        origin:   Optional world translation applied to all frames (for centering).
    """
    frames: List[Frame]
    white_bkgd: bool = True
    aabb: Optional[Tuple[float, float, float, float, float, float]] = None
    near: Optional[float] = None
    far: Optional[float] = None
    scale: float = 1.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_torch(self, device: Optional[Union[str, torch.device]] = None) -> "Scene":
        return Scene(
            frames=[f.to_torch(device) for f in self.frames],
            white_bkgd=self.white_bkgd,
            aabb=self.aabb,
            near=self.near,
            far=self.far,
            scale=self.scale,
            origin=self.origin
        )

    @property
    def H(self) -> int:
        return self.frames[0].H

    @property
    def W(self) -> int:
        return self.frames[0].W

    def __len__(self) -> int:
        return len(self.frames)
