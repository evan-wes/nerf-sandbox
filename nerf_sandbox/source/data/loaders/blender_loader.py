from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Literal

import numpy as np
import imageio.v2 as imageio

from nerf_sandbox.source.data.scene import Frame, Scene


Centering = Literal["auto", "none"]


class BlenderSceneLoader:
    """
    Blender synthetic dataset loader (STRICT path resolution).

    Controls
    --------
    centering   : "auto" | "none"
        - "auto": subtract mean translation of camera centers (simple recenter).
        - "none": no recentering.  (Default for Blender.)
    scene_scale : float
        Uniform multiplier applied to camera translations after centering.

    Notes
    -----
    - Camera convention is OpenGL/Blender (+X right, +Y up, camera looks -Z).
    - This loader composites RGBA onto white if requested.
    - Images are expected exactly at <root>/<file_path>.png, where file_path is
      read from transforms_{split}.json.
    """

    def __init__(
        self,
        root: str | Path,
        downscale: int = 1,
        white_bkgd: bool = True,
        *,
        centering: Centering = "none",
        scene_scale: float = 1.0
    ) -> None:
        self.root = Path(root)
        self.downscale = int(downscale)
        self.white_bkgd = bool(white_bkgd)
        self.centering: Centering = centering
        self.scene_scale = float(scene_scale)
        self.camera_convention = "opengl"


    # ---------- internals ----------

    def _imread_float(self, path: Path) -> np.ndarray:
        img = imageio.imread(path).astype(np.float32) / 255.0
        if self.downscale > 1:
            import cv2
            H, W = img.shape[:2]
            img = cv2.resize(
                img, (W // self.downscale, H // self.downscale),
                interpolation=cv2.INTER_AREA
            )
        return img

    @staticmethod
    def _K_from_angle(W: int, H: int, camera_angle_x: float) -> np.ndarray:
        f = 0.5 * W / np.tan(0.5 * camera_angle_x)
        return np.array([[f, 0, 0.5 * W],
                         [0, f, 0.5 * H],
                         [0, 0, 1       ]], dtype=np.float32)

    def _resolve_img_path(self, file_path: str) -> Path:
        """
        STRICT: interpret file_path as relative to self.root and force '.png'.
        Examples:
          root=/.../lego, file_path="./test/r_0" → /.../lego/test/r_0.png
        """
        p = Path(file_path)
        target = (p if p.is_absolute() else (self.root / p)).with_suffix(".png")
        target = target.resolve()
        if not target.exists():
            raise FileNotFoundError(
                "Image file not found.\n"
                f"  file_path in JSON : {file_path}\n"
                f"  dataset root      : {self.root}\n"
                f"  expected PNG path : {target}\n"
            )
        return target

    # ---------- public API ----------

    def load(self, split: str = "train") -> Scene:
        tf_path = self.root / f"transforms_{split}.json"
        if not tf_path.exists():
            raise FileNotFoundError(f"Could not find transforms file: {tf_path}")
        with open(tf_path, "r") as f:
            meta = json.load(f)

        frames_meta = meta["frames"]

        # Probe first image for size (after downscale)
        probe_path = self._resolve_img_path(frames_meta[0]["file_path"])
        im0 = self._imread_float(probe_path)
        H, W = im0.shape[:2]
        K = self._K_from_angle(W, H, float(meta["camera_angle_x"]))

        frames: List[Frame] = []
        centers = []

        for fr in frames_meta:
            img_path = self._resolve_img_path(fr["file_path"])
            img = self._imread_float(img_path)

            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            centers.append(c2w[:3, 3].copy())
            frames.append(Frame(image=img, K=K.copy(), c2w=c2w))

        # ---- Centering (one method for Blender) ----
        if self.centering == "auto" and len(centers) > 0:
            mean_t = np.mean(np.stack(centers, axis=0), axis=0)
            new_frames: List[Frame] = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] -= mean_t
                new_frames.append(replace(fr, c2w=c2w_new))
            frames = new_frames

        # ---- Single user scaling (uniform) ----
        if self.scene_scale != 1.0:
            s = float(self.scene_scale)
            new_frames: List[Frame] = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] *= s
                new_frames.append(replace(fr, c2w=c2w_new))
            frames = new_frames

        return Scene(frames=frames, white_bkgd=self.white_bkgd)
