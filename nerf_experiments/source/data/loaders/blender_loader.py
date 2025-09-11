"""
BlenderSceneLoader for NeRF synthetic dataset (STRICT path resolution).

Assumptions (per your dataset):
- transforms_{split}.json is in the dataset root (e.g., .../nerf_synthetic/lego).
- Each frame entry has "file_path" like "./test/r_0" (or "./train/r_0", etc.).
- Actual images are PNGs located exactly at: <root>/<file_path>.png
  e.g., root="./lego", file_path="./test/r_0" → "./lego/test/r_0.png"

No directory guessing, no extension guessing beyond forcing '.png'.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
import imageio.v2 as imageio

from nerf_experiments.source.data.scene import Frame, Scene


class BlenderSceneLoader:
    def __init__(self,
                 root: str | Path,
                 downscale: int = 1,
                 white_bkgd: bool = True,
                 scene_scale: float = 1.0,
                 center_origin: bool = False,
                 composite_on_load: bool = True) -> None:
        self.root = Path(root)
        self.downscale = int(downscale)
        self.white_bkgd = bool(white_bkgd)
        self.scene_scale = float(scene_scale)
        self.center_origin = bool(center_origin)
        self.composite_on_load = bool(composite_on_load)

    def _imread_float(self, path: Path) -> np.ndarray:
        img = imageio.imread(path).astype(np.float32) / 255.0
        if self.downscale > 1:
            import cv2  # requires opencv-python
            H, W = img.shape[:2]
            img = cv2.resize(img, (W // self.downscale, H // self.downscale), interpolation=cv2.INTER_AREA)
        return img

    def _K_from_angle(self, W: int, H: int, camera_angle_x: float) -> np.ndarray:
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        K = np.array([[focal, 0, 0.5 * W],
                      [0, focal, 0.5 * H],
                      [0,     0,      1]], dtype=np.float32)
        return K

    def load(self, split: str = "train") -> Scene:
        tf_path = self.root / f"transforms_{split}.json"
        if not tf_path.exists():
            raise FileNotFoundError(f"Could not find transforms file: {tf_path}")
        with open(tf_path, "r") as f:
            meta = json.load(f)

        frames_meta = meta["frames"]

        # Probe first image to get size
        probe_path = self._resolve_img_path(frames_meta[0]["file_path"])
        im0 = self._imread_float(probe_path)
        H, W = im0.shape[:2]
        K = self._K_from_angle(W, H, float(meta["camera_angle_x"]))

        frames: List[Frame] = []
        c_all = []
        for fr in frames_meta:
            img_path = self._resolve_img_path(fr["file_path"])
            img = self._imread_float(img_path)
            if self.composite_on_load and self.white_bkgd and img.shape[-1] == 4:
                rgb, a = img[..., :3], img[..., 3:4]
                img = rgb * a + (1.0 - a)

            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            c_all.append(c2w[:3, 3].copy())
            frames.append(Frame(image=img, K=K.copy(), c2w=c2w))

        # Optionally recenter cameras
        if self.center_origin and len(c_all) > 0:
            mean_t = np.mean(np.stack(c_all, axis=0), axis=0)
            new_frames = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] -= mean_t
                new_frames.append(replace(fr, c2w=c2w_new))
            frames = new_frames

        # Optionally scale scene radius
        if self.scene_scale != 1.0:
            new_frames = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] *= float(self.scene_scale)
                new_frames.append(replace(fr, c2w=c2w_new))
            frames = new_frames

        return Scene(frames=frames, white_bkgd=self.white_bkgd)

    def _resolve_img_path(self, file_path: str) -> Path:
        """
        STRICT: interpret file_path as relative to self.root and force '.png'.
        Examples:
          root=/.../lego, file_path="./test/r_0" → /.../lego/test/r_0.png
          root=/.../lego, file_path="train/r_12" → /.../lego/train/r_12.png
        """
        p = Path(file_path)
        # absolute? still force .png
        if p.is_absolute():
            target = p.with_suffix(".png")
        else:
            # relative to dataset root (lego folder)
            target = (self.root / p).with_suffix(".png")
        target = target.resolve()

        if not target.exists():
            raise FileNotFoundError(
                "Image file not found.\n"
                f"  file_path in JSON : {file_path}\n"
                f"  dataset root      : {self.root}\n"
                f"  expected PNG path : {target}\n"
                "This loader does not search other directories. "
                "If your dataset uses a different layout, adjust 'file_path' strings or this resolver."
            )
        return target
