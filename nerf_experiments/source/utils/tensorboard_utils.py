"""
TensorBoard utilities.

- Lazy writer initialization (only when first used)
- Scalar logging
- Image logging helpers:
  * RGB in linear space (with optional linear->sRGB conversion)
  * Grayscale (opacity / masks)
  * Depth normalized to [near, far]
- Optional downscaling on GPU before copying to CPU
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter  # requires `pip install tensorboard`
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from nerf_experiments.source.utils.render_utils import linear_to_srgb


class TensorBoardLogger:
    def __init__(self, enabled: bool, logdir: str, image_max_side: int = 512) -> None:
        self.enabled = bool(enabled)
        self.logdir = str(logdir)
        self.image_max_side = int(image_max_side)
        self.writer: Optional[SummaryWriter] = None

    # ---------- lifecycle ----------
    def _ensure_writer(self) -> bool:
        if not self.enabled:
            return False
        if self.writer is not None:
            return True
        if SummaryWriter is None:
            print("[TB] Disabled (tensorboard package not found).")
            self.enabled = False
            return False
        try:
            Path(self.logdir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)
            print(f"[TB] Writing logs to: {self.logdir}")
            return True
        except Exception as e:
            print(f"[TB] Failed to init SummaryWriter: {e}")
            self.enabled = False
            return False

    def flush(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            except Exception:
                pass
            self.writer = None

    # ---------- scalars ----------
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if not self._ensure_writer():
            return
        try:
            self.writer.add_scalar(tag, float(value), int(step))
        except Exception:
            pass

    # ---------- images ----------
    @staticmethod
    def _to_hwc01(x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Accept HWC float/uint8 or CHW torch/numpy and return HWC float in [0,1]."""
        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
            if t.dim() == 3 and t.shape[0] in (1, 3):  # CHW
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            arr = np.asarray(x)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    def _resize_if_needed(self, arr: np.ndarray) -> np.ndarray:
        if self.image_max_side <= 0:
            return arr
        H, W = int(arr.shape[0]), int(arr.shape[1])
        m = max(H, W)
        if m <= self.image_max_side:
            return arr
        scale = self.image_max_side / float(m)
        H2, W2 = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        try:
            import cv2
            arr = cv2.resize(arr, (W2, H2), interpolation=cv2.INTER_AREA)
        except Exception:
            # fallback to PIL if cv2 not available
            from PIL import Image
            arr = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize((W2, H2), Image.BOX)) / 255.0
        return arr

    def add_image(self, tag: str, img: torch.Tensor | np.ndarray, step: int) -> None:
        if not self._ensure_writer():
            return
        arr = self._to_hwc01(img)
        arr = self._resize_if_needed(arr)
        try:
            self.writer.add_image(tag, arr, global_step=int(step), dataformats="HWC")
        except Exception:
            pass

    def log_validation_images(
        self,
        val: Dict[str, torch.Tensor],
        step: int,
        near: float,
        far: float,
        convert_to_srgb: bool = True,
        image_max_side: Optional[int] = None,
    ) -> None:
        if not self._ensure_writer():
            return

        if image_max_side is not None:
            self.image_max_side = int(image_max_side)

        rgb = val["rgb"]  # (H,W,3) linear [0,1]
        if convert_to_srgb:
            rgb = linear_to_srgb(rgb)
        acc = val["acc"].squeeze(-1)          # (H,W)
        depth = val["depth"].squeeze(-1)      # (H,W)
        depth_norm = (depth - near) / (max(1e-8, (far - near)))

        self.add_image("val/rgb", rgb, step)
        self.add_image("val/acc", acc, step)
        self.add_image("val/depth", depth_norm, step)
        self.flush()

