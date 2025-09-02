"""
Contains the BlenderLoader class designed to load the synthetic
data released with the original NeRF paper (Mildenhall et al. 2020).

The dataset consists of:
- images/ (PNG files with RGBA)
- transforms_{split}.json (camera poses, intrinsics, filepaths)


Each JSON entry contains:
- "file_path": relative to the data root
- "transform_matrix": 4x4 camera-to-world matrix
- "camera_angle_x": horizontal field of view (radians)

We parse these into Frame records and return a Scene instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

from nerf_experiments.source.data.scene import Frame, Scene


class BlenderSceneLoader:
    """
    Loader for NeRF Blender-format scenes.

    Parameters
    ----------
    root : str | pathlib.Path
        Path to the scene directory containing `transforms_{split}.json` and images.
    downscale : int, default=1
        Integer factor to downscale images and intrinsics (e.g., 2 halves W/H).
    white_bg : bool, default=True
        If True, marks the scene for white-background compositing at render time.
    scene_scale : float, default=1.0
        Global scale factor applied to world coordinates (camera centers).
    center_origin : bool, default=False
        If True, recenter poses so the mean camera center is at the world origin.

    Examples
    --------
    >>> loader = BlenderSceneLoader("/path/to/scene", downscale=2, white_bg=True)
    >>> scene_train = loader.load("train")
    >>> scene_val = loader.load("val")
    """

    def __init__(
        self,
        root: str | Path,
        downscale: int = 1,
        white_bg: bool = True,
        scene_scale: float = 1.0,
        center_origin: bool = False,
    ) -> None:
        self.root: Path = Path(root)
        self.downscale: int = int(downscale)
        self.white_bg: bool = bool(white_bg)
        self.scene_scale: float = float(scene_scale)
        self.center_origin: bool = bool(center_origin)

    @staticmethod
    def _compute_intrinsics_from_fov(W: int, H: int, camera_angle_x: float) -> np.ndarray:
        """
        Compute a pinhole intrinsics matrix from horizontal FOV.

        Parameters
        ----------
        W : int
            Image width in pixels.
        H : int
            Image height in pixels.
        camera_angle_x : float
            Horizontal field-of-view in radians (from Blender JSON).

        Returns
        -------
        K : numpy.ndarray, shape (3, 3), dtype float32
            Intrinsics matrix:
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,  0,   1]]
        """
        fx = W / (2.0 * np.tan(camera_angle_x * 0.5))
        fy = fx
        cx = W * 0.5
        cy = H * 0.5
        K: np.ndarray = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    @staticmethod
    def _imread_float(path: Path) -> np.ndarray:
        """
        Read an image into float32 in [0, 1].

        Parameters
        ----------
        path : pathlib.Path
            Path to an image file.

        Returns
        -------
        img : numpy.ndarray, shape (H, W, C), dtype float32
            Image array with values in [0, 1]. Preserves channels (RGB or RGBA).
        """
        img = imageio.imread(path)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype != np.float32:
            img = img.astype(np.float32)
        return img

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load an image and optionally downscale by the loader's factor.

        Parameters
        ----------
        path : pathlib.Path
            Path to an image file.

        Returns
        -------
        img : numpy.ndarray, shape (H, W, C), dtype float32
            Image array in [0, 1]; downscaled if `self.downscale > 1`.
        """
        img = self._imread_float(path)
        if self.downscale > 1:
            h, w = img.shape[:2]
            nh, nw = h // self.downscale, w // self.downscale
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((nw, nh), resample=Image.BILINEAR)
            img = np.asarray(pil).astype(np.float32) / 255.0
        return img

    def _resolve_image_path(self, rel: str) -> Path:
        """
        Resolve a relative image path from Blender JSON to an absolute Path.

        Parameters
        ----------
        rel : str
            Relative path from JSON (often starts with './') and typically
            lacks an extension (defaults to .png).

        Returns
        -------
        path : pathlib.Path
            Absolute path to the image file, with `.png` appended if missing.
        """
        p = self.root / rel.replace("./", "")
        return p if p.suffix else p.with_suffix(".png")

    def load(self, split: Literal["train", "val", "test"] = "train") -> "Scene":
        """
        Load a Blender-format NeRF split into a `Scene`.

        Parameters
        ----------
        split : {'train', 'val', 'test'}, default='train'
            Which split JSON to load (e.g., `transforms_train.json`).

        Returns
        -------
        scene : Scene
            Scene with:
            - frames: List[Frame] with image (float32 in [0,1]), K (3x3), c2w (4x4)
            - near, far: float defaults (≈2.0, 6.0 scaled by `scene_scale`)
            - white_bg: bool (copied from loader)
            - scale: float (`scene_scale`)

        Raises
        ------
        FileNotFoundError
            If the split JSON is not found at the expected location.
        ValueError
            If a pose (c2w) has an unexpected shape.
        """
        tf_path: Path = self.root / f"transforms_{split}.json"
        if not tf_path.exists():
            raise FileNotFoundError(f"Could not find {tf_path}")
        with open(tf_path, "r") as f:
            meta = json.load(f)

        frames_json = meta["frames"]

        # Infer image size from the first frame
        first_img_path = self._resolve_image_path(frames_json[0]["file_path"])
        img0 = self._imread_float(first_img_path)
        H, W = img0.shape[:2]

        # Intrinsics from FOV (adjust for downscale)
        K = self._compute_intrinsics_from_fov(W, H, meta["camera_angle_x"]).astype(np.float32)
        if self.downscale > 1:
            K = K.copy()
            K[0, 0] /= self.downscale; K[1, 1] /= self.downscale
            K[0, 2] /= self.downscale; K[1, 2] /= self.downscale

        # Load all images and poses
        images: list[np.ndarray] = []
        c2ws: list[np.ndarray] = []
        for fr in frames_json:
            img_path = self._resolve_image_path(fr["file_path"])
            images.append(self._load_image(img_path))
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            if c2w.shape != (4, 4):
                raise ValueError(f"Unexpected c2w shape: {c2w.shape}")
            c2ws.append(c2w)
        c2ws_np: np.ndarray = np.stack(c2ws, axis=0)  # [N, 4, 4]

        # Optional recentering & scaling
        if self.center_origin:
            centers = c2ws_np[:, :3, 3]
            mean_center = centers.mean(axis=0, keepdims=True)
            c2ws_np[:, :3, 3] = centers - mean_center
        if self.scene_scale != 1.0:
            c2ws_np = c2ws_np.copy()
            c2ws_np[:, :3, 3] *= self.scene_scale

        # Build frames
        frames: List["Frame"] = []
        for img, c2w in zip(images, c2ws_np):
            frames.append(Frame(image=img, K=K, c2w=c2w, mask=None, dist=None, meta={}))

        # Typical Blender near/far
        near: float = 2.0 * self.scene_scale
        far: float = 6.0 * self.scene_scale

        return Scene(frames=frames, white_bg=self.white_bg, near=near, far=far, scale=self.scene_scale)

