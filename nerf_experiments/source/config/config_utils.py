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
class ModelConfig:
    init_acc: float = 0.05   # initial accumulated opacity
    # ... maybe also hidden_dim, n_layers, etc.

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
    model: ModelConfig = field(default_factory=ModelConfig)
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
    model = _merge_defaults(ModelConfig().__dict__, yaml_cfg.get("model", {}))
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
    md = ModelConfig(
        init_acc=float(model["init_acc"])
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

    cfg = NerfConfig(data=dc, camera=cc, model=md, train=tr, render=rc, encoding=ec)
    cfg.validate()
    return cfg

# NEW: extract optional sections into a plain dict for Trainer “extras”
def parse_extras(yaml_cfg: dict) -> dict:
    def _get(d, path, default=None):
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    extras = {}
    # eval
    extras["eval_nc"] = _parse_int_like(_get(yaml_cfg, "eval.eval_nc", 0)) or None
    extras["eval_nf"] = _parse_int_like(_get(yaml_cfg, "eval.eval_nf", 0)) or None
    extras["eval_chunk"] = _parse_int_like(_get(yaml_cfg, "eval.eval_chunk", 0)) or None
    # memory
    extras["micro_chunks"] = _parse_int_like(_get(yaml_cfg, "memory.micro_chunks", 0)) or None
    extras["ckpt_mlp"] = bool(_get(yaml_cfg, "memory.ckpt_mlp", False))
    extras["cuda_expandable_segments"] = bool(_get(yaml_cfg, "memory.cuda_expandable_segments", False))
    # logging
    extras["use_tb"] = bool(_get(yaml_cfg, "logging.tensorboard", False))
    extras["tb_logdir"] = _get(yaml_cfg, "logging.tb_logdir", None)
    extras["tb_group_root"] = _get(yaml_cfg, "logging.tb_group_root", None)
    # safety
    extras["thermal_throttle"] = bool(_get(yaml_cfg, "safety.thermal_throttle", False))
    extras["thermal_throttle_max_micro"] = _parse_int_like(_get(yaml_cfg, "safety.thermal_throttle_max_micro", 0)) or None
    extras["thermal_throttle_sleep"] = float(_get(yaml_cfg, "safety.thermal_throttle_sleep", 0.0) or 0.0)
    extras["gpu_temp_threshold"] = _parse_int_like(_get(yaml_cfg, "safety.gpu_temp_threshold", 0)) or None
    extras["gpu_temp_check_every"] = _parse_int_like(_get(yaml_cfg, "safety.gpu_temp_check_every", 0)) or None
    # path render
    extras["render_path"] = bool(_get(yaml_cfg, "path.render_path", False))
    extras["path_type"] = _get(yaml_cfg, "path.path_type", None)
    extras["path_frames"] = _parse_int_like(_get(yaml_cfg, "path.path_frames", 0)) or None
    extras["path_fps"] = _parse_int_like(_get(yaml_cfg, "path.path_fps", 0)) or None
    extras["path_res_scale"] = float(_get(yaml_cfg, "path.path_res_scale", 0.0) or 0.0) or None
    # color
    extras["targets_are_srgb"] = bool(_get(yaml_cfg, "color.targets_are_srgb", True))
    extras["sigma_noise_std"] = float(_get(yaml_cfg, "color.sigma_noise_std", 1e-3))
    return extras