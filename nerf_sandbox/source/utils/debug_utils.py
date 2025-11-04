# utils/debug_run.py
from __future__ import annotations
import json, sys, platform, traceback, math, time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # not strictly required, but often handy


# ------------------------- small helpers -------------------------

@torch.no_grad()
def _to_py(x: Any):
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.item())
            return [float(v) for v in x.flatten().tolist()[:8]]  # truncate long tensors
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            return [float(v) for v in x.flatten().tolist()[:8]]
        if isinstance(x, (float, int, str, bool)) or x is None:
            return x
        return str(x)
    except Exception:
        return str(x)

@torch.no_grad()
def _stats(name: str, t: torch.Tensor) -> Dict[str, Any]:
    try:
        if isinstance(t, torch.Tensor):
            t = t.detach()
        return {
            "name": name,
            "shape": list(t.shape),
            "min": float(torch.min(t).item()),
            "max": float(torch.max(t).item()),
            "mean": float(torch.mean(t).item()),
            "std": float(torch.std_mean(t)[0].item()) if t.numel() > 1 else 0.0,
        }
    except Exception as e:
        return {"name": name, "error": repr(e)}

def _safe_add(d: Dict[str, Any], key: str, fn: Callable[[], Any]):
    try:
        d[key] = fn()
    except Exception as e:
        d[key] = {"error": repr(e), "trace": traceback.format_exc(limit=1)}

def summarize_nerf_module(nerf: torch.nn.Module) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["class"]       = nerf.__class__.__name__
        info["skip_pos"]    = _to_py(getattr(nerf, "skip_pos", None))
        info["n_layers"]    = _to_py(getattr(nerf, "n_layers", None))
        info["hidden_dim"]  = _to_py(getattr(nerf, "hidden_dim", None))
        info["enc_pos_dim"] = _to_py(getattr(nerf, "enc_pos_dim", None))
        info["enc_dir_dim"] = _to_py(getattr(nerf, "enc_dir_dim", None))
        linears = []
        for m in nerf.modules():
            if isinstance(m, torch.nn.Linear):
                linears.append([int(m.in_features), int(m.out_features)])
        info["linear_layers"] = linears
    except Exception as e:
        info["error"] = repr(e)
    return info


# ------------------------- tiny forward probe -------------------------

@torch.no_grad()
def _tiny_forward_probe(
    trainer,
    scene,
    device: torch.device,
    *,
    get_camera_rays: Callable[..., Tuple[torch.Tensor, ...]],
    nerf_forward_pass: Callable[..., Tuple[torch.Tensor, ...]],
    probe_grid: int = 8,
    probe_Nc: int = 64,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    fr = scene.frames[0]
    H, W = fr.image.shape[:2]
    K = np.array(fr.K, dtype=np.float32)

    # small grid of pixels (xy)
    ys = torch.linspace(0, H - 1, steps=probe_grid, device=device).round().long()
    xs = torch.linspace(0, W - 1, steps=probe_grid, device=device).round().long()
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2).to(torch.float32)

    # WORLD rays (for viewdirs)
    (
        rays_o_world,            # (N,3)
        rays_d_world_unit,       # (N,3)
        rays_d_world_norm,       # (N,1)
        rays_o_marching_w,       # (N,3)
        rays_d_marching_unit_w,  # (N,3)
        rays_d_marching_norm_w,  # (N,1)
    ) = get_camera_rays(
        H, W, K, fr.c2w, device=device, dtype=torch.float32,
        convention=getattr(trainer, "convention", "opengl"),
        pixel_center=True,
        as_ndc=False, near_plane=float(getattr(trainer, "near_world", 1.0)),
        pixels_xy=grid,
    )

    # Marching rays (world or NDC)
    if getattr(trainer, "use_ndc", False):
        (
            _oW, _dWu, _dWn,
            rays_o_marching, rays_d_marching_unit, rays_d_marching_norm
        ) = get_camera_rays(
            H, W, K, fr.c2w, device=device, dtype=torch.float32,
            convention=getattr(trainer, "convention", "opengl"),
            pixel_center=True,
            as_ndc=True, near_plane=float(getattr(trainer, "ndc_near_plane_world", 1.0)),
            pixels_xy=grid,
        )
        s_near, s_far = 0.0, 1.0
    else:
        rays_o_marching      = rays_o_marching_w
        rays_d_marching_unit = rays_d_marching_unit_w
        rays_d_marching_norm = rays_d_marching_norm_w
        s_near = float(getattr(trainer, "near_world", 0.0))
        s_far  = float(getattr(trainer, "far_world", 1.0))

    Bprobe = rays_o_marching.shape[0]
    Nc = min(int(getattr(trainer, "nc", probe_Nc)), probe_Nc)
    t = torch.linspace(0.0, 1.0, steps=Nc, device=device)
    zc = (s_near * (1.0 - t) + s_far * t).expand(Bprobe, Nc).contiguous()

    comp, w, acc, depth = nerf_forward_pass(
        rays_o=rays_o_marching,
        rays_d_unit=rays_d_marching_unit,
        z_vals=zc,
        pos_enc=trainer.pos_enc, dir_enc=trainer.dir_enc, nerf=trainer.nerf_c,
        white_bkgd=bool(getattr(trainer, "white_bkgd", True)),
        ray_norms=rays_d_marching_norm,
        viewdirs_world_unit=rays_d_world_unit,
        sigma_activation=str(getattr(trainer, "sigma_activation", "relu")),
        raw_noise_std=0.0, training=False,
        mlp_chunk=min(Bprobe * Nc, int(getattr(trainer, "eval_chunk", 8192))),
        infinite_last_bin=bool(getattr(trainer, "infinite_last_bin", False)),
    )

    out["acc_mean"] = float(acc.mean().item())
    out["acc_minmax"] = [float(acc.min().item()), float(acc.max().item())]
    out["weights_sum_minus_acc_maxabs"] = float((w.sum(-1) - acc.squeeze(-1)).abs().max().item())
    out["comp_rgb_stats"] = _stats("comp_rgb", comp)
    out["depth_stats"] = _stats("depth", depth)
    out["z_range"] = [float(zc.min().item()), float(zc.max().item())]
    out["delta_metric_mean"] = float(((zc[:, 1:] - zc[:, :-1]) * rays_d_marching_norm).mean().item())
    return out


# ------------------------- main entry point -------------------------

@torch.no_grad()
def dump_run_debug(
    trainer,
    scene,
    out_dir: str | Path,
    *,
    get_camera_rays: Callable[..., Tuple[torch.Tensor, ...]],
    nerf_forward_pass: Callable[..., Tuple[torch.Tensor, ...]],
    sample_pdf: Callable[..., torch.Tensor],
    probe_grid: int = 8,
    probe_Nc: int = 64,
) -> Path:
    """
    Write a comprehensive JSON debug report to <out_dir>/run_debug.json.
    Pass function refs to avoid circular imports.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_debug.json"

    report: Dict[str, Any] = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(getattr(trainer, "device", "cpu")),
            "amp_enabled": bool(getattr(trainer, "amp", False)),
        }
    }

    # ---- Static config ----
    cfg: Dict[str, Any] = {}
    _safe_add(cfg, "data_kind",             lambda: _to_py(getattr(trainer.cfg, "data_kind", None)))
    _safe_add(cfg, "data_root",             lambda: _to_py(getattr(trainer.cfg, "data_root", None)))
    _safe_add(cfg, "convention",            lambda: _to_py(getattr(trainer, "convention", "opengl")))
    _safe_add(cfg, "use_ndc",               lambda: bool(getattr(trainer, "use_ndc", False)))
    _safe_add(cfg, "ndc_near_plane_world",  lambda: float(getattr(trainer, "ndc_near_plane_world", 1.0)))
    _safe_add(cfg, "white_bkgd",            lambda: bool(getattr(trainer, "white_bkgd", True)))
    _safe_add(cfg, "composite_on_load",     lambda: bool(getattr(scene, "composite_on_load", getattr(scene, "white_bkgd", True))))
    _safe_add(cfg, "near_world",            lambda: float(getattr(trainer, "near_world", 0.0)))
    _safe_add(cfg, "far_world",             lambda: float(getattr(trainer, "far_world", 1.0)))
    _safe_add(cfg, "nc_nf",                 lambda: {"nc": int(getattr(trainer, "nc", 64)), "nf": int(getattr(trainer, "nf", 128))})
    _safe_add(cfg, "train_batch",           lambda: int(getattr(trainer, "rays_per_batch", 4096)))
    _safe_add(cfg, "micro_chunks",          lambda: int(getattr(trainer, "train_micro_chunks", 1)))
    _safe_add(cfg, "train_mlp_chunk",       lambda: int(getattr(trainer, "train_mlp_chunk", 0)))
    _safe_add(cfg, "eval_chunk",            lambda: int(getattr(trainer, "eval_chunk", 32768)))
    _safe_add(cfg, "sigma_activation",      lambda: str(getattr(trainer, "sigma_activation", "relu")))
    _safe_add(cfg, "raw_noise_std",         lambda: float(getattr(trainer, "raw_noise_std", 0.0)))
    _safe_add(cfg, "infinite_last_bin",     lambda: bool(getattr(trainer, "infinite_last_bin", False)))
    _safe_add(cfg, "max_steps",             lambda: int(getattr(trainer.cfg, "max_steps", -1)))
    _safe_add(cfg, "global_step",           lambda: int(getattr(trainer, "global_step", 0)))
    report["config"] = cfg

    # ---- Optimizer / Scheduler ----
    opt: Dict[str, Any] = {}
    try:
        opt["optimizer"] = trainer.opt.__class__.__name__
        opt["param_groups"] = [
            {
                "lr": float(pg["lr"]),
                "betas": _to_py(pg.get("betas", None)),
                "weight_decay": _to_py(pg.get("weight_decay", 0.0)),
            } for pg in trainer.opt.param_groups
        ]
    except Exception as e:
        opt["error"] = repr(e)
    try:
        if getattr(trainer, "sched", None) is None:
            opt["scheduler"] = None
        else:
            opt["scheduler"] = {
                "class": trainer.sched.__class__.__name__,
                "state": {k: _to_py(v) for k, v in trainer.sched.state_dict().items()},
            }
    except Exception as e:
        opt["scheduler_error"] = repr(e)
    report["optim"] = opt

    # ---- Encoders / MLPs ----
    enc: Dict[str, Any] = {}
    _safe_add(enc, "pos_encoder_dim", lambda: int(getattr(trainer.pos_enc, "out_dim", getattr(trainer.pos_enc, "D", -1))))
    _safe_add(enc, "dir_encoder_dim", lambda: int(getattr(trainer.dir_enc, "out_dim", getattr(trainer.dir_enc, "D", -1))))
    report["encoders"] = enc

    mlp = {}
    mlp["coarse"] = summarize_nerf_module(trainer.nerf_c)
    mlp["fine"]   = summarize_nerf_module(trainer.nerf_f) if getattr(trainer, "nerf_f", None) is not None else None
    report["mlp"] = mlp

    # ---- Scene snapshot ----
    sc: Dict[str, Any] = {}
    try:
        sc["num_frames"] = len(scene.frames)
        fr0 = scene.frames[0]
        sc["frame0_image_shape"] = list(fr0.image.shape)
        sc["frame0_K"] = [[float(x) for x in row] for row in np.array(fr0.K, dtype=np.float32)]
        sc["frame0_c2w_firstrow"] = [float(x) for x in np.array(fr0.c2w[:1]).reshape(-1).tolist()]
    except Exception as e:
        sc["error"] = repr(e)
    report["scene"] = sc

    # ---- Rays & pose sanity (frame 0) ----
    rays: Dict[str, Any] = {}
    try:
        fr = scene.frames[0]
        H, W = fr.image.shape[:2]
        K = np.array(fr.K, dtype=np.float32)
        (
            rays_o_world, rays_d_world_unit, rays_d_world_norm,
            _r_o_m, _r_d_m_u, _r_d_m_n
        ) = get_camera_rays(
            H, W, K, fr.c2w, device=getattr(trainer, "device", "cpu"), dtype=torch.float32,
            convention=getattr(trainer, "convention", "opengl"), pixel_center=True,
            as_ndc=False, near_plane=float(getattr(trainer, "near_world", 1.0)),
            pixels_xy=None,
        )
        rays["world_o_stats"]  = _stats("rays_o_world",        rays_o_world)
        rays["world_du_stats"] = _stats("rays_d_world_unit",   rays_d_world_unit)
        rays["world_dn_stats"] = _stats("rays_d_world_norm",   rays_d_world_norm)

        cx = int(round(float(K[0, 2]))); cy = int(round(float(K[1, 2])))
        idx = cy * W + cx
        R = np.asarray(fr.c2w[:3, :3])
        fwd_world = (R[:, 2] if getattr(trainer, "convention", "opengl") in ("colmap", "opencv") else -R[:, 2])
        v_world = rays_d_world_unit[idx].detach().cpu().numpy()
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v_world, fwd_world) / (np.linalg.norm(v_world) + 1e-9))))))
        rays["center_angle_deg"] = ang

        if getattr(trainer, "use_ndc", False):
            (
                _oW,_dWu,_dWn,
                rays_o_ndc, rays_d_ndc_unit, rays_d_ndc_norm
            ) = get_camera_rays(
                H, W, K, fr.c2w, device=getattr(trainer, "device", "cpu"), dtype=torch.float32,
                convention=getattr(trainer, "convention", "opengl"), pixel_center=True,
                as_ndc=True, near_plane=float(getattr(trainer, "ndc_near_plane_world", 1.0)),
                pixels_xy=None,
            )
            rays["ndc_o_stats"]  = _stats("rays_o_ndc",       rays_o_ndc)
            rays["ndc_du_stats"] = _stats("rays_d_ndc_unit",  rays_d_ndc_unit)
            rays["ndc_dn_stats"] = _stats("rays_d_ndc_norm",  rays_d_ndc_norm)
    except Exception as e:
        rays["error"] = repr(e)
    report["rays"] = rays

    # ---- Forward probe ----
    _safe_add(report, "forward_probe", lambda: _tiny_forward_probe(
        trainer, scene, getattr(trainer, "device", torch.device("cpu")),
        get_camera_rays=get_camera_rays,
        nerf_forward_pass=nerf_forward_pass,
        probe_grid=probe_grid,
        probe_Nc=probe_Nc,
    ))

    # ---- Hierarchical sampling sanity ----
    hs: Dict[str, Any] = {}
    try:
        B = 4
        Nc = int(getattr(trainer, "nc", 64))
        t = torch.linspace(0.0, 1.0, steps=Nc, device=getattr(trainer, "device", "cpu"))
        zc = (0.0 * (1.0 - t) + 1.0 * t).expand(B, Nc).contiguous()
        w = torch.randn(B, Nc, device=getattr(trainer, "device", "cpu")).abs() + 1e-3
        bins_mid     = 0.5 * (zc[:, 1:] + zc[:, :-1])
        weights_bins = 0.5 * (w[:, 1:] + w[:, :-1])
        zf = sample_pdf(bins_mid, weights_bins, n_samples=min(32, Nc), deterministic=True)
        hs["bins_mid_shape"]    = list(bins_mid.shape)
        hs["weights_bins_shape"]= list(weights_bins.shape)
        hs["zf_shape"]          = list(zf.shape)
        hs["zf_range"]          = [float(zf.min().item()), float(zf.max().item())]
    except Exception as e:
        hs["error"] = repr(e)
    report["hier_sampling"] = hs

    # ---- Save ----
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_to_py)
        print(f"[debug] wrote {path} ({time.time() - t0:.2f}s)")
    except Exception as e:
        print(f"[debug] FAILED to write {path}: {e}")
    return path


@torch.no_grad()
def debug_topk_fine_hit(
    step: int,
    bins_mid: torch.Tensor,         # (B, M) midpoint per coarse interval
    weights_bins: torch.Tensor,     # (B, M?) interval weights (before/after detach OK)
    z_fine: torch.Tensor,           # (B, Nf) fine samples
    *,
    topk: int = 4,
    logger: Callable[[str], None] = print,
    tag: str = "train",
) -> Dict[str, float]:
    """
    Compute how many fine samples fall into the top-k heaviest coarse intervals.

    Robust to small shape mismatches and non-contiguous inputs; reorders weights if bins_mid
    aren’t strictly sorted. Prints a one-line diagnostic and returns summary stats.
    """
    # Ensure 2D (B, M) and contiguous for searchsorted
    if bins_mid.dim() == 1:      bins_mid = bins_mid.unsqueeze(0)
    if weights_bins.dim() == 1:  weights_bins = weights_bins.unsqueeze(0)
    if z_fine.dim() == 1:        z_fine = z_fine.unsqueeze(0)

    bins_mid     = bins_mid.contiguous()
    weights_bins = weights_bins.contiguous()
    z_fine       = z_fine.contiguous()

    Bm, M_b = bins_mid.shape
    Bw, M_w = weights_bins.shape
    Bz, _   = z_fine.shape
    assert Bm == Bw == Bz, f"batch mismatch: bins_mid={Bm}, weights_bins={Bw}, z_fine={Bz}"

    # Align along M if needed
    if M_b != M_w:
        M = min(M_b, M_w)
        logger(f"[diag] step={step} {tag}: aligning bins: bins_mid={M_b} weights_bins={M_w} → M={M}")
        bins_mid     = bins_mid[:, :M]
        weights_bins = weights_bins[:, :M]
        M_b = M_w = M

    # If bins_mid isn’t strictly sorted along last dim, sort and permute weights accordingly
    needs_sort = (bins_mid[:, 1:] < bins_mid[:, :-1]).any().item()
    if needs_sort:
        order = torch.argsort(bins_mid, dim=-1)
        # Gather per-row using the order
        row_idx = torch.arange(bins_mid.size(0), device=bins_mid.device).unsqueeze(-1)
        bins_mid     = bins_mid[row_idx, order]
        weights_bins = weights_bins[row_idx, order]

    M = bins_mid.shape[-1]
    k = int(max(1, min(topk, M)))

    # Normalize interval weights per ray
    w = weights_bins.clamp_min(0)
    wsum = w.sum(-1, keepdim=True) + 1e-9
    wnorm = w / wsum  # (B, M)

    # Expected top-k mass
    top_mass, top_idx = torch.topk(wnorm, k=k, dim=-1, largest=True, sorted=False)  # (B,k)
    expected_mass = top_mass.sum(-1)                                                # (B,)

    # Map each fine sample to its interval index
    # searchsorted gives first j with bins_mid[j] >= z; interval below is j-1
    idx = torch.searchsorted(bins_mid, z_fine, right=False)
    idx = (idx - 1).clamp(min=0, max=M - 1)  # (B, Nf)

    # Hit if interval index ∈ top-k set
    hits = (idx.unsqueeze(-1) == top_idx.unsqueeze(1)).any(dim=-1)  # (B, Nf)
    hit_rate = hits.float().mean(dim=-1)                             # (B,)

    hit_mean = float(hit_rate.mean().item())
    hit_std  = float(hit_rate.std().item())
    exp_mean = float(expected_mass.mean().item())
    exp_std  = float(expected_mass.std().item())

    logger(
        f"[diag] step={step} {tag}: fine→top-{k} hit={hit_mean*100:.1f}±{hit_std*100:.1f}% "
        f"| expected~{exp_mean*100:.1f}±{exp_std*100:.1f}%"
    )

    return {
        "hit_rate_mean": hit_mean,
        "hit_rate_std":  hit_std,
        "expected_mean": exp_mean,
        "expected_std":  exp_std,
    }
