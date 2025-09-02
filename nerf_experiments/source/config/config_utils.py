"""
Contains a dataclasses designed for encapsulating separate sections ;
of configs, and a dataclass meant for containing them
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import yaml


# ---------- helpers ----------

def _parse_int_like(x: Any) -> int:
    """Parse ints that may be strings like '200k', '5K', '1_000'."""
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip().replace("_", "").lower()
        mult = 1
        if s.endswith("k"):
            mult = 1_000; s = s[:-1]
        elif s.endswith("m"):
            mult = 1_000_000; s = s[:-1]
        return int(float(s) * mult)
    raise ValueError(f"Cannot parse int-like value: {x!r}")


# ---------- section dataclasses ----------

@dataclass
class DataConfig:
    kind: str = "blender"                 # blender|llff|colmap|custom
    root: str = "./"
    undistort: bool = False
    white_bg: bool = True
    aabb: Optional[Tuple[float, float, float, float, float, float]] = None
    split: str = "train"
    downscale: int = 1

@dataclass
class CameraConfig:
    ndc: bool = False
    near: float = 2.0
    far: float = 6.0
    model: str = "pinhole"                # 'opencv' if you use distortion

@dataclass
class TrainSection:
    device: str = "cuda"
    rays_per_batch: int = 4096
    max_steps: int = 200_000
    amp: bool = True
    val_every: int = 5_000
    log_every: int = 100
    ckpt_every: int = 10_000
    lr: float = 5e-4
    out_dir: str = "outputs"

@dataclass
class RenderConfig:
    Nc: int = 64
    Nf: int = 128
    det_fine: bool = False

@dataclass
class PEConfig:
    num_freqs: int = 10
    max_freq_log2: Optional[int] = None    # optional; many encoders infer from num_freqs
    include_input: bool = True

@dataclass
class EncodingConfig:
    pos: PEConfig = field(default_factory=lambda: PEConfig(num_freqs=10, max_freq_log2=9, include_input=True))
    dir: PEConfig = field(default_factory=lambda: PEConfig(num_freqs=4,  max_freq_log2=3, include_input=True))


# ---------- unified config ----------

@dataclass
class NerfConfig:
    data: DataConfig = field(default_factory=DataConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    train: TrainSection = field(default_factory=TrainSection)
    render: RenderConfig = field(default_factory=RenderConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)

    # Simple validations / normalizations
    def validate(self) -> None:
        if self.data.kind not in ("blender", "llff", "colmap", "custom"):
            raise ValueError(f"Unknown data.kind: {self.data.kind}")
        if self.data.downscale < 1:
            raise ValueError("downscale must be >= 1")
        if self.camera.ndc and self.data.kind != "llff":
            # NDC is typically forward-facing/LLFF; it's okay to warn instead of error
            pass
        if self.render.Nc <= 0 or self.render.Nf < 0:
            raise ValueError("Nc must be >0 and Nf >=0")
        if self.train.max_steps <= 0:
            raise ValueError("max_steps must be > 0")


# ---------- YAML loading ----------

def _merge_defaults(defaults: dict, user: dict) -> dict:
    """Shallow-merge user dict into defaults dict."""
    out = dict(defaults)
    out.update(user or {})
    return out

def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def parse_nerf_config(yaml_cfg: dict) -> NerfConfig:
    # Pull sections (user may omit some)
    data = _merge_defaults(DataConfig().__dict__, yaml_cfg.get("data", {}))
    camera = _merge_defaults(CameraConfig().__dict__, yaml_cfg.get("camera", {}))
    train = _merge_defaults(TrainSection().__dict__, yaml_cfg.get("train", {}))
    render = _merge_defaults(RenderConfig().__dict__, yaml_cfg.get("render", {}))
    enc_user = yaml_cfg.get("encoding", {})
    pos_user = enc_user.get("pos", {})
    dir_user = enc_user.get("dir", {})

    # Parse int-like fields
    train["rays_per_batch"] = _parse_int_like(train["rays_per_batch"])
    train["max_steps"] = _parse_int_like(train["max_steps"])
    train["val_every"] = _parse_int_like(train["val_every"])
    train["log_every"] = _parse_int_like(train["log_every"])
    train["ckpt_every"] = _parse_int_like(train["ckpt_every"])

    # Build dataclasses
    dc = DataConfig(
        kind=str(data["kind"]),
        root=str(data["root"]),
        undistort=bool(data["undistort"]),
        white_bg=bool(data["white_bg"]),
        aabb=tuple(data["aabb"]) if data.get("aabb") is not None else None,
        split=str(data.get("split", "train")),
        downscale=int(data.get("downscale", 1)),
    )
    cc = CameraConfig(
        ndc=bool(camera["ndc"]),
        near=float(camera["near"]),
        far=float(camera["far"]),
        model=str(camera["model"]),
    )
    tr = TrainSection(
        device=str(train["device"]) if "device" in train else "cuda",
        rays_per_batch=int(train["rays_per_batch"]),
        max_steps=int(train["max_steps"]),
        amp=bool(train["amp"]),
        val_every=int(train["val_every"]),
        log_every=int(train["log_every"]),
        ckpt_every=int(train["ckpt_every"]),
        lr=float(train["lr"]),
        out_dir=str(train["out_dir"]),
    )
    rc = RenderConfig(
        Nc=int(render["Nc"]),
        Nf=int(render["Nf"]),
        det_fine=bool(render["det_fine"]),
    )
    ec = EncodingConfig(
        pos=PEConfig(
            num_freqs=int(pos_user.get("num_freqs", 10)),
            max_freq_log2=(int(pos_user["max_freq_log2"]) if "max_freq_log2" in pos_user else None),
            include_input=bool(pos_user.get("include_input", True)),
        ),
        dir=PEConfig(
            num_freqs=int(dir_user.get("num_freqs", 4)),
            max_freq_log2=(int(dir_user["max_freq_log2"]) if "max_freq_log2" in dir_user else None),
            include_input=bool(dir_user.get("include_input", True)),
        ),
    )

    cfg = NerfConfig(data=dc, camera=cc, train=tr, render=rc, encoding=ec)
    cfg.validate()
    return cfg


# ---------- Adapter: NerfConfig -> Trainer.TrainConfig ----------

# Import here (or move this into trainer.py) to avoid circulars
try:
    from trainer import TrainConfig as RuntimeTrainConfig
except Exception:
    RuntimeTrainConfig = None  # if trainer not imported yet

def to_runtime_train_config(cfg: NerfConfig) -> "RuntimeTrainConfig":
    """
    Convert a NerfConfig (sectioned) into the flat TrainConfig used by Trainer.
    """
    if RuntimeTrainConfig is None:
        raise RuntimeError("RuntimeTrainConfig unavailable; import trainer.TrainConfig first.")

    return RuntimeTrainConfig(
        data_root=cfg.data.root,
        split=cfg.data.split,
        downscale=cfg.data.downscale,
        white_bg=cfg.data.white_bg,
        device=cfg.train.device,
        rays_per_batch=cfg.train.rays_per_batch,
        nc=cfg.render.Nc,
        nf=cfg.render.Nf,
        det_fine=cfg.render.det_fine,
        jitter=True,  # you can expose this later if needed
        pos_freqs=cfg.encoding.pos.num_freqs,
        dir_freqs=cfg.encoding.dir.num_freqs,
        include_input=cfg.encoding.pos.include_input,  # keep pos/dir consistent for now
        lr=cfg.train.lr,
        max_steps=cfg.train.max_steps,
        log_every=cfg.train.log_every,
        val_every=cfg.train.val_every,
        ckpt_every=cfg.train.ckpt_every,
        amp=cfg.train.amp,
        out_dir=cfg.train.out_dir,
        near_override=cfg.camera.near,
        far_override=cfg.camera.far,
    )
