"""
Contains the RuntimeTrainConfig dataclass meant to be a stable interface for the Trainer class
"""


from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict, replace, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import warnings
import yaml

# Import your sectioned config
from nerf_experiments.source.config.config_utils import NerfConfig, _parse_int_like, parse_nerf_config, load_yaml_config, parse_extras


# -----------------------------
# Runtime config (frozen)
# -----------------------------
@dataclass(frozen=True)
class RuntimeTrainConfig:
    # Data / camera
    data_root: str
    split: str
    downscale: int
    white_bg: bool
    ndc: bool
    near: float
    far: float

    # Model init
    init_acc: float

    # Device / IO
    device: str
    out_dir: str
    amp: bool
    seed: int

    # Training knobs
    rays_per_batch: int
    max_steps: int
    log_every: int
    val_every: int
    ckpt_every: int

    # Optimizer / scheduler
    lr: float
    scheduler: str                 # e.g. "cosine"
    scheduler_params: Dict[str, Any]  # fully-resolved params, e.g. {"T_max": 200000, "eta_min": 5e-5}

    # Rendering / sampling
    nc: int
    nf: int
    det_fine: bool

    # Encoders (resolved sizes/flags only)
    pos_num_freqs: int
    pos_include_input: bool
    dir_num_freqs: int
    dir_include_input: bool


# -----------------------------
# Helpers
# -----------------------------
def _asdict_dc(dc) -> Dict[str, Any]:
    return asdict(dc) if is_dataclass(dc) else dict(dc)

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _dot_set(obj: Any, dotpath: str, value: Any) -> bool:
    """
    Apply a 'a.b.c' override into a dataclass hierarchy.
    Returns True if applied, False if path not found.
    """
    parts = dotpath.split(".")
    cur = obj
    for i, part in enumerate(parts):
        # Handle dataclass (attribute) or dict (key)
        if is_dataclass(cur):
            if not hasattr(cur, part):
                return False
            if i == len(parts) - 1:
                # Replace dataclass with a new instance (immutably) at top-level only
                try:
                    object.__setattr__(cur, part, value)  # may work if not frozen
                    return True
                except Exception:
                    # reconstruct the node
                    fields = {f: getattr(cur, f) for f in cur.__dataclass_fields__}
                    fields[part] = value
                    parent = replace(cur, **fields)  # not applied to original
                    return False  # signal caller we couldn't patch in-place
            else:
                cur = getattr(cur, part)
        elif isinstance(cur, dict):
            if part not in cur:
                return False
            if i == len(parts) - 1:
                cur[part] = value
                return True
            cur = cur[part]
        else:
            return False
    return False

def _apply_overrides_copy(cfg: NerfConfig, overrides: Dict[str, Any]) -> NerfConfig:
    """
    Returns a NEW NerfConfig with dotpath overrides applied by reconstructing
    sections as needed. Supports overrides like:
      {"train.device": "cuda:1", "render.Nc": 96, "train.max_steps": "200k"}
    """
    if not overrides:
        return cfg

    # Convert to nested dict, apply, then rebuild dataclasses (simplest, robust).
    base = {
        "data": _asdict_dc(cfg.data),
        "camera": _asdict_dc(cfg.camera),
        "model": _asdict_dc(cfg.model),
        "train": _asdict_dc(cfg.train),
        "render": _asdict_dc(cfg.render),
        "encoding": {
            "pos": _asdict_dc(cfg.encoding.pos),
            "dir": _asdict_dc(cfg.encoding.dir),
        },
    }

    # apply dot-path overrides into the nested dict
    for k, v in overrides.items():
        # Normalize int-like strings on common fields
        if any(k.endswith(suf) for suf in ("rays_per_batch", "max_steps", "val_every", "log_every", "ckpt_every")):
            v = _parse_int_like(v)

        node = base
        parts = k.split(".")
        ok = True
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                warnings.warn(f"Override ignored (unknown path): {k}")
                ok = False
                break
            node = node[p]
        if not ok:
            continue
        leaf = parts[-1]
        if leaf not in node:
            warnings.warn(f"Override ignored (unknown field): {k}")
            continue
        node[leaf] = v

    # rebuild dataclasses
    from dataclasses import fields
    new_cfg = NerfConfig(
        data=cfg.data.__class__(**base["data"]),
        camera=cfg.camera.__class__(**base["camera"]),
        model=cfg.model.__class__(**base["model"]),
        train=cfg.train.__class__(**base["train"]),
        render=cfg.render.__class__(**base["render"]),
        encoding=cfg.encoding.__class__(
            pos=cfg.encoding.pos.__class__(**base["encoding"]["pos"]),
            dir=cfg.encoding.dir.__class__(**base["encoding"]["dir"]),
        ),
    )
    # re-validate
    new_cfg.validate()
    return new_cfg

def _resolve_scheduler(cfg: NerfConfig) -> Tuple[str, Dict[str, Any]]:
    """
    For now we fix to cosine; extend here to support more schedulers later.
    """
    name = "cosine"
    T_max = int(cfg.train.max_steps)
    eta_min = float(cfg.train.lr) * 0.1  # common default
    return name, {"T_max": T_max, "eta_min": eta_min}

def _resolve_seed(overrides: Optional[Dict[str, Any]]) -> int:
    # Allow override key 'train.seed' or 'seed'; default 42.
    if overrides:
        if "train.seed" in overrides:
            return int(overrides["train.seed"])
        if "seed" in overrides:
            return int(overrides["seed"])
    return 42

def save_resolved_config(rt: RuntimeTrainConfig, out_dir: str | Path, original_yaml: Optional[dict] = None) -> None:
    """
    Save the fully resolved runtime config, and optionally the original YAML,
    into the run directory for provenance.
    """
    out = _ensure_dir(out_dir)
    with open(out / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(asdict(rt), f, sort_keys=False)
    if original_yaml is not None:
        with open(out / "input_config.yaml", "w") as f:
            yaml.safe_dump(original_yaml, f, sort_keys=False)


# -----------------------------
# Main adapter
# -----------------------------
def to_runtime_train_config(
    cfg: NerfConfig,
    *,
    cli_overrides: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str | Path] = None,
    original_yaml: Optional[dict] = None,
) -> RuntimeTrainConfig:
    """
    Convert a sectioned NerfConfig into a frozen RuntimeTrainConfig that
    the Trainer consumes. Optionally apply CLI overrides (dot paths),
    compute scheduler params, set seed, and save the resolved config.

    Precedence: CLI overrides > YAML > defaults.
    """
    cfg2 = _apply_overrides_copy(cfg, cli_overrides or {})

    # Scheduler spec
    sched_name, sched_params = _resolve_scheduler(cfg2)

    # Seed
    seed = _resolve_seed(cli_overrides)

    rt = RuntimeTrainConfig(
        # Data / camera
        data_root=str(cfg2.data.root),
        split=str(cfg2.data.split),
        downscale=int(cfg2.data.downscale),
        white_bg=bool(cfg2.data.white_bg),
        ndc=bool(cfg2.camera.ndc),
        near=float(cfg2.camera.near),
        far=float(cfg2.camera.far),

        # Model init
        init_acc=float(cfg2.model.init_acc),

        # Device / IO
        device=str(cfg2.train.device),
        out_dir=str(cfg2.train.out_dir),
        amp=bool(cfg2.train.amp),
        seed=int(seed),

        # Training
        rays_per_batch=int(cfg2.train.rays_per_batch),
        max_steps=int(cfg2.train.max_steps),
        log_every=int(cfg2.train.log_every),
        val_every=int(cfg2.train.val_every),
        ckpt_every=int(cfg2.train.ckpt_every),

        # Optimizer / scheduler
        lr=float(cfg2.train.lr),
        scheduler=sched_name,
        scheduler_params=sched_params,

        # Rendering
        nc=int(cfg2.render.Nc),
        nf=int(cfg2.render.Nf),
        det_fine=bool(cfg2.render.det_fine),

        # Encoders
        pos_num_freqs=int(cfg2.encoding.pos.num_freqs),
        pos_include_input=bool(cfg2.encoding.pos.include_input),
        dir_num_freqs=int(cfg2.encoding.dir.num_freqs),
        dir_include_input=bool(cfg2.encoding.dir.include_input),
    )

    # Optional: write resolved + original YAML to output dir for provenance
    if save_dir is not None:
        save_resolved_config(rt, save_dir, original_yaml=original_yaml)

    return rt

# NEW: end-to-end loader that returns both the frozen runtime config and a dict of extras
def load_configs_with_extras(yaml_path: str | Path,
                             cli_overrides: Optional[Dict[str, Any]],
                             save_dir: Optional[str | Path] = None):
    """
    Load YAML, apply CLI dot-path overrides, return (RuntimeTrainConfig, extras, original_yaml_dict).
    `extras` contains optional features (tensorboard, micro-chunks, eval samples, thermal guard, etc.).
    """
    original_yaml = load_yaml_config(yaml_path)
    # parse -> dataclasses
    base_cfg = parse_nerf_config(original_yaml)
    # Split overrides: core vs extras
    core_prefixes = ("data.", "train.", "camera.", "model.", "render.", "encoding.")
    cli_overrides = cli_overrides or {}
    core_overrides   = {k: v for k, v in cli_overrides.items() if k.startswith(core_prefixes)}
    extras_overrides = {k: v for k, v in cli_overrides.items() if k not in core_overrides}

    # Build runtime (frozen core) with core overrides only
    rt = to_runtime_train_config(base_cfg,
                                 cli_overrides=core_overrides,
                                 save_dir=save_dir,
                                 original_yaml=original_yaml)

    # Apply BOTH core+extras overrides to a YAML dict copy, then parse extras from that
    patched = deepcopy(original_yaml)
    def _set(d, dot, v):
        cur = d; parts = dot.split(".")
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                return
            cur = cur[p]
        # create leaf if missing (extras may not exist yet)
        if isinstance(cur, dict):
            cur[parts[-1]] = v
    for k, v in {**core_overrides, **extras_overrides}.items():
        # normalize some int-like values commonly passed as strings
        if any(k.endswith(suf) for suf in ("eval_nc","eval_nf","micro_chunks",
                                           "thermal_throttle_max_micro","gpu_temp_threshold",
                                           "gpu_temp_check_every","path_frames","path_fps")):
            try: v = _parse_int_like(v)
            except: pass
        _set(patched, k, v)
    extras = parse_extras(patched)
    return rt, extras, original_yaml

