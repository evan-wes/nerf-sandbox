"""Contains the Renderer class"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
import numpy as np
import torch

class Renderer:
    """Abstract renderer interface.

    Implementations should expose a `render_rays` method that performs
    coarse sampling → MLP forward → volume rendering → hierarchical resampling → fine pass.
    """
    def render_rays(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def eval_image(self, frame: Frame, cfg: Dict) -> Dict[str, torch.Tensor]:
        """Render a full image for validation (no jitter, det=True)."""
        raise NotImplementedError