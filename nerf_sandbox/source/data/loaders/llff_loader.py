from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Optional

import numpy as np
import imageio.v2 as imageio

# ✅ Your classes
from nerf_sandbox.source.data.scene import Frame, Scene

Split = Literal["train", "val", "test"]


def _normalize_np(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


class LLFFSceneLoader:
    """
    LLFF loader (faithful to nerf-pytorch):
      - Read poses_bounds.npy
      - Reorder to OpenGL [right, up, back, t, hwf]
      - Scale by sc = 1/(min(bounds)*bd_factor)
      - Recenter to average pose (nerf-pytorch implementation)
      - Prefer pre-minified images_{downscale}; no per-frame resize
      - PNGs read with apply_gamma=True to match nerf-pytorch
      - Intrinsics per frame: K = [[f,0,W/2],[0,f,H/2],[0,0,1]]
      - Holdout split: single test view chosen via argmin center distance
    """

    def __init__(
        self,
        root: str | Path,
        downscale: int = 1,
        white_bkgd: bool = True,
        *,
        bd_factor: float = 0.75,
        use_llff_holdout: bool = True,   # nerf-pytorch style single test view
        holdout_every: int = 0,          # if >0, overrides with periodic split
        holdout_offset: int = 0
    ) -> None:
        self.root = Path(root)
        self.downscale = int(downscale)
        self.white_bkgd = bool(white_bkgd)
        self.bd_factor = float(bd_factor)
        self.use_llff_holdout = bool(use_llff_holdout)
        self.holdout_every = int(holdout_every)
        self.holdout_offset = int(holdout_offset)

        self.camera_convention = "opengl"
        self._norm_scale = 1.0
        self._effective_scale = 1.0
        self._chosen_img_dir: Path | None = None
        self._factor_used: int = 1  # integer factor like nerf-pytorch

    # ---------- public API ----------

    def load(self, split: Split = "train") -> Scene:
        poses_345_gl, bounds_2, files, H_loaded, W_loaded = self._load_poses_bounds_bmild()
        # Scale by sc BEFORE recenter (nerf-pytorch)
        sc = 1.0 / (float(bounds_2.min()) * self.bd_factor)
        poses_345_gl[:, 3, :] *= sc
        bounds_2 *= sc

        # Recenter (exact nerf-pytorch implementation on (N,3,5) layout)
        poses_n = np.moveaxis(poses_345_gl, -1, 0).astype(np.float32)  # (N,3,5)
        poses_n = self._recenter_poses_np(poses_n)                     # (N,3,5)
        poses_345_gl = np.moveaxis(poses_n, 0, -1).astype(np.float32)  # (3,5,N)

        # Pick split
        N = poses_345_gl.shape[-1]
        if self.holdout_every > 0:  # periodic split (optional)
            i_all = np.arange(N)
            i_test = i_all[self.holdout_offset::self.holdout_every]
            i_val = i_test
            i_train = np.array([i for i in i_all if i not in set(i_test)], int)
        elif self.use_llff_holdout:
            i_test = np.array([self._choose_llff_test_idx(poses_n)], int)
            i_val = i_test
            i_train = np.array([i for i in range(N) if i != int(i_test[0])], int)
        else:
            i_test = np.array([], int)
            i_val = i_test
            i_train = np.arange(N, dtype=int)

        use_idx = i_train if split == "train" else i_val

        frames: List[Frame] = []
        img_dir = self._chosen_img_dir
        assert img_dir is not None

        for i in use_idx.tolist():
            # Per-frame H, W, f are in poses[:,4,i] *after* factor update we applied
            H = int(round(poses_345_gl[0, 4, i]))
            W = int(round(poses_345_gl[1, 4, i]))
            f = float(poses_345_gl[2, 4, i])

            img_path = img_dir / files[i]
            img = self._imread_float_png_gamma(img_path)  # matches nerf-pytorch

            if img.shape[0] != H or img.shape[1] != W:
                # Should not happen if images_{factor} exists; warn only.
                # (We keep the image as-loaded to avoid reinterpolation.)
                pass


            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = poses_345_gl[:, :4, i]

            K = np.array([
                [f, 0.0, 0.5 * W],
                [0.0, f, 0.5 * H],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

            frames.append(Frame(
                image=img.astype(np.float32),
                K=K,
                c2w=c2w.astype(np.float32),
                meta={"file_path": str(img_path), "basename": img_path.name, "convention": self.camera_convention},
            ))

        self._norm_scale = sc
        self._effective_scale = sc
        print(f"[llff] dir={img_dir}  factor={self._factor_used}  norm_scale={self._norm_scale:.6f}")
        return Scene(frames=frames, white_bkgd=self.white_bkgd)

    def get_global_near_far(self, percentile: Tuple[float, float] = (5.0, 95.0)) -> tuple[float, float]:
        poses_345_gl, bounds_2, _, _, _ = self._load_poses_bounds_bmild()
        sc = 1.0 / (float(bounds_2.min()) * self.bd_factor)
        b = bounds_2 * sc
        lo = float(np.percentile(b, float(percentile[0])))
        hi = float(np.percentile(b, float(percentile[1])))
        return lo, hi

    # ---------- internals (faithful to nerf-pytorch) ----------

    def _choose_img_dir_and_factor(self) -> tuple[Path, int]:
        """Prefer images_{downscale}. If missing and downscale==1, use images/."""
        cand = self.root / f"images_{self.downscale}"
        if cand.exists() and cand.is_dir():
            return cand, self.downscale
        base = self.root / "images"
        if self.downscale != 1:
            raise FileNotFoundError(
                f"Expected pre-minified folder {cand}. "
                f"To mirror nerf-pytorch, create it (e.g., via their _minify), or set downscale=1."
            )
        if not base.exists():
            raise FileNotFoundError(f"Missing images directory: {base}")
        return base, 1

    def _load_poses_bounds_bmild(self) -> tuple[np.ndarray, np.ndarray, List[str], int, int]:
        npy = self.root / "poses_bounds.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Missing {npy}")
        arr = np.load(npy)  # (N,17)

        # Original LLFF layout (3,5,N) with axes: [down, right, back, t, hwf]
        poses = arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # (3,5,N)
        bds   = arr[:, -2:].transpose([1, 0]).astype(np.float32)      # (2,N)

        # Reorder to OpenGL: [right, up, back, t, hwf]
        poses_gl = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]],
            axis=1
        ).astype(np.float32)  # (3,5,N)

        img_dir, factor = self._choose_img_dir_and_factor()
        self._chosen_img_dir = img_dir
        self._factor_used = factor

        files = sorted([p.name for p in img_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if poses_gl.shape[-1] != len(files):
            raise RuntimeError(f"Mismatch between imgs ({len(files)}) and poses ({poses_gl.shape[-1]})")

        # Set H,W from actual (first) image in chosen directory, and scale f by 1/factor
        im0 = self._imread_float_png_gamma(img_dir / files[0])
        H_loaded, W_loaded = int(im0.shape[0]), int(im0.shape[1])
        poses_gl[0, 4, :] = float(H_loaded)
        poses_gl[1, 4, :] = float(W_loaded)
        poses_gl[2, 4, :] = poses_gl[2, 4, :] * (1.0 / float(factor))

        # Return bounds as (N,2) to match your callers
        return poses_gl, bds.T.astype(np.float32), files, H_loaded, W_loaded

    @staticmethod
    def _poses_avg_np(poses_n: np.ndarray) -> np.ndarray:
        # poses_n: (N, 3, 5)
        Rcols = poses_n[:, :, :3]    # (N,3,3)
        t_all = poses_n[:, :, 3]     # (N,3)

        z = _normalize_np(Rcols[:, :, 2].mean(axis=0))
        up = _normalize_np(Rcols[:, :, 1].mean(axis=0))
        right = _normalize_np(np.cross(up, z))
        up = _normalize_np(np.cross(z, right))

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = z
        c2w[:3, 3] = t_all.mean(axis=0)
        return c2w

    def _recenter_poses_np(self, poses_n: np.ndarray) -> np.ndarray:
        # poses_n: (N, 3, 5) in OpenGL order [right, up, back, t, hwf]
        N = poses_n.shape[0]

        # --- average pose (4x4) ---
        c2w_avg = self._poses_avg_np(poses_n)              # (4,4)
        w2c = np.linalg.inv(c2w_avg).astype(np.float32)    # (4,4)

        out = poses_n.copy()
        for i in range(N):
            # make a 4x4, apply recenter, write back the top 3x4
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = out[i, :, :4]
            c2w = (w2c @ c2w).astype(np.float32)
            out[i, :, :4] = c2w[:3, :4]
        return out

    @staticmethod
    def _choose_llff_test_idx(poses_n: np.ndarray) -> int:
        # As in nerf-pytorch: pick the camera closest to the mean center
        c2w = LLFFSceneLoader._poses_avg_np(poses_n)          # (4,4)
        center = c2w[:3, 3]                                   # (3,)
        cams = poses_n[:, :3, 3]                              # (N,3)
        dists = np.sum((cams - center[None, :]) ** 2, axis=-1)
        return int(np.argmin(dists))

    @staticmethod
    def _imread_float_png_gamma(path: Path) -> np.ndarray:
        """
        Read image as float32 in [0,1]; for PNG, apply_gamma=True like nerf-pytorch.
        """
        if path.suffix.lower() == ".png":
            try:
                arr = imageio.imread(path, apply_gamma=True)
            except TypeError:
                # older imageio: fallback (closest behavior)
                arr = imageio.imread(path)
        else:
            arr = imageio.imread(path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr
