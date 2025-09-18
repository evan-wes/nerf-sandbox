from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple

import cv2
import numpy as np
import imageio.v2 as imageio


from nerf_sandbox.source.data.scene import Frame, Scene  # same as Blender loader

Split = Literal["train", "val", "test"]


class LLFFSceneLoader:
    """
    LLFF-style scene loader with built-in train/val/test splits via holdout.
    Directory layout (at least one must exist):
        <root>/poses_bounds.npy
        <root>/images/                  # preferred
        <root>/images_2, images_4, ...  # optional downsampled sets

    Mirrors the Blender loader behaviors:
      - downscale via cv2.INTER_AREA
      - composite RGBA over white if white_bkgd=True
      - center_origin subtracts mean translation (like Blender)
      - scene_scale multiplies translations
      - returns Scene(frames=[Frame(image, K, c2w)], white_bkgd=white_bkgd)
    """

    def __init__(self,
                 root: str | Path,
                 downscale: int = 1,
                 white_bkgd: bool = False,
                 scene_scale: float = 1.0,
                 center_origin: bool = False,
                 composite_on_load: bool = False,
                 holdout_every: int = 8,
                 holdout_offset: int = 0,
                 angle_sorted_holdout: bool = False,
                 use_llff_recenter: bool = False) -> None:
        self.root = Path(root)
        self.downscale = int(downscale)
        self.white_bkgd = bool(white_bkgd)
        self.scene_scale = float(scene_scale)
        self.center_origin = bool(center_origin)
        self.composite_on_load = bool(composite_on_load)
        self.camera_convention = "colmap"

        # split config living in the loader (not the trainer)
        self.holdout_every = int(holdout_every)
        self.holdout_offset = int(holdout_offset)
        self.angle_sorted_holdout = bool(angle_sorted_holdout)

        # optional classic LLFF "average-pose" recenter; off by default to mirror Blender semantics
        self.use_llff_recenter = bool(use_llff_recenter)

    # ---------- public API ----------

    def load(self, split: Split = "train") -> Scene:
        poses35, bounds = self._load_poses_bounds()          # (N,3,5), (N,2)
        img_dir, files = self._resolve_image_dir()           # path + sorted filenames

        # Build (N,4,4) c2w from 3x4
        c2w_all = self._c2w_from_llff(poses35)

        # Optional LLFF-style recentering (average pose)
        if self.use_llff_recenter:
            c2w_all = self._recenter_to_average_pose(c2w_all)

        # Intrinsics source from H,W,f per-frame stored in poses_bounds
        HWF = poses35[:, :, 4].astype(np.float32)  # (N,3) → [H,W,f]
        H_orig, W_orig, F_orig = HWF[:, 0], HWF[:, 1], HWF[:, 2]

        # Optionally sort by azimuth for uniform angular holdout
        order = np.arange(len(files))
        if self.angle_sorted_holdout:
            # azimuth of camera center (x,z)
            centers = c2w_all[:, :3, 3]
            az = np.arctan2(centers[:, 0], centers[:, 2])
            order = np.argsort(az)
            files = [files[i] for i in order]
            c2w_all = c2w_all[order]
            H_orig, W_orig, F_orig = H_orig[order], W_orig[order], F_orig[order]

        # Compute split indices inside the loader
        i_all = np.arange(len(files))
        if self.holdout_every > 0:
            i_test = i_all[self.holdout_offset :: self.holdout_every]
        else:
            i_test = np.array([], dtype=int)
        i_val = i_test  # LLFF convention; change here if you want a different val policy
        i_train = np.array([i for i in i_all if i not in set(i_test)])

        if split == "train":
            use_idx = i_train
        elif split in ("val", "test"):
            use_idx = i_val
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        frames: List[Frame] = []
        translations = []

        for idx in use_idx.tolist():
            img_path = img_dir / files[idx]
            img = self._imread_float(img_path)  # same pattern as Blender loader

            # Loader-level downscale (INTER_AREA), consistent with Blender loader
            if self.downscale > 1:
                H, W = img.shape[:2]
                img = cv2.resize(img, (W // self.downscale, H // self.downscale), interpolation=cv2.INTER_AREA)

            # Composite RGBA over white background (like Blender loader)
            if self.composite_on_load and self.white_bkgd and img.shape[-1] == 4:
                rgb, a = img[..., :3], img[..., 3:4]
                img = rgb * a + (1.0 - a)

            # Intrinsics per image: scale focal to the actually loaded width
            w_now, h_now = img.shape[1], img.shape[0]
            # Protect against odd metadata: recompute scale from stored W to current width
            scale_w = w_now / max(1.0, float(W_orig[idx]))
            f_now = float(F_orig[idx] * scale_w)

            K = np.array([[f_now, 0.0, 0.5 * w_now],
                          [0.0,   f_now, 0.5 * h_now],
                          [0.0,   0.0,   1.0]], dtype=np.float32)

            c2w = c2w_all[idx].astype(np.float32)
            translations.append(c2w[:3, 3].copy())

            frames.append(Frame(image=img, K=K, c2w=c2w, meta={"file_path": str(img_path), "basename": img_path.name}))

        # Mirror Blender semantics: optional centering (subtract mean t) then scene-scale
        if self.center_origin and len(translations) > 0:
            mean_t = np.mean(np.stack(translations, axis=0), axis=0)
            new_frames: List[Frame] = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] -= mean_t
                new_frames.append(Frame(image=fr.image, K=fr.K, c2w=c2w_new))
            frames = new_frames

        if self.scene_scale != 1.0:
            new_frames: List[Frame] = []
            for fr in frames:
                c2w_new = fr.c2w.copy()
                c2w_new[:3, 3] *= float(self.scene_scale)
                new_frames.append(Frame(image=fr.image, K=fr.K, c2w=c2w_new))
            frames = new_frames

        return Scene(frames=frames, white_bkgd=self.white_bkgd)

    def get_global_near_far(self, percentile=(5.0, 95.0)) -> tuple[float, float]:
        """
        Infer global near/far from LLFF bounds with percentiles, then apply scene_scale.
        Returns floats usable by your sampler / renderers (non-NDC pipeline).
        """
        poses, bounds = self._load_poses_bounds()   # (N,3,5), (N,2)
        lo = float(np.percentile(bounds[:, 0], float(percentile[0])))
        hi = float(np.percentile(bounds[:, 1], float(percentile[1])))
        s  = float(self.scene_scale or 1.0)
        return lo * s, hi * s

    # ---------- internals ----------

    def _imread_float(self, path: Path) -> np.ndarray:
        # mirrors your Blender loader's image read + float convert
        img = imageio.imread(path).astype(np.float32) / 255.0
        return img

    def _load_poses_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        npy = self.root / "poses_bounds.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Missing {npy} – generate LLFF poses first.")
        arr = np.load(npy)  # (N,17)
        poses = arr[:, :-2].reshape([-1, 3, 5]).astype(np.float32)  # (N,3,5)
        bounds = arr[:, -2:].reshape([-1, 2]).astype(np.float32)    # (N,2)
        return poses, bounds

    def _resolve_image_dir(self) -> Tuple[Path, list[str]]:
        base = self.root
        # Prefer full-res images/
        primary = base / "images"
        if primary.exists():
            files = sorted([p.name for p in primary.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
            if not files:
                raise RuntimeError(f"No images in {primary}")
            return primary, files
        # Fallback: pick the smallest images_k/ available
        candidates = []
        for k in (2, 4, 8, 16, 32):
            d = base / f"images_{k}"
            if d.exists():
                candidates.append((k, d))
        if not candidates:
            raise FileNotFoundError(f"No images/ or images_{'{k}'} directories in {base}")
        k_min, chosen = sorted(candidates, key=lambda x: x[0])[0]
        files = sorted([p.name for p in chosen.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not files:
            raise RuntimeError(f"No images in {chosen}")
        return chosen, files

    @staticmethod
    def _c2w_from_llff(poses35: np.ndarray) -> np.ndarray:
        """LLFF: 3x5 per-image block → 4x4 c2w (homogeneous).
        LLFF rotation cols are [down, right, back].
        We convert to cols [right, down, forward] to match 'colmap' convention.
        """
        poses35 = poses35.astype(np.float32)
        N = poses35.shape[0]

        R_llff = poses35[:, :, :3]       # (N,3,3) columns: [down, right, back]
        t      = poses35[:, :, 3]        # (N,3)

        R_colmap = np.stack([
            R_llff[:, :, 1],             # right
            R_llff[:, :, 0],             # down
        -R_llff[:, :, 2],             # forward = -back
        ], axis=2).astype(np.float32)    # (N,3,3)

        mats = np.repeat(np.eye(4, dtype=np.float32)[None, ...], N, axis=0)
        mats[:, :3, :3] = R_colmap
        mats[:, :3,  3] = t
        return mats

    @staticmethod
    def _poses_average(c2w_all: np.ndarray) -> np.ndarray:
        centers = c2w_all[:, :3, 3]
        z = _normalize(c2w_all[:, :3, 2].mean(0))
        up = _normalize(c2w_all[:, :3, 1].mean(0))
        right = _normalize(np.cross(up, z))
        up = _normalize(np.cross(z, right))
        avg = np.eye(4, dtype=np.float32)
        avg[:3, 0], avg[:3, 1], avg[:3, 2] = right, up, z
        avg[:3, 3] = centers.mean(0)
        return avg

    def _recenter_to_average_pose(self, c2w_all: np.ndarray) -> np.ndarray:
        avg = self._poses_average(c2w_all)
        w2c = np.linalg.inv(avg)
        return (w2c[None, ...] @ c2w_all)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)
