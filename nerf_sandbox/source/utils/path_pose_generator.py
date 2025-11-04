from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class PathPoseGenerator:
    """
    Unified camera-path generator (single public API):
      - path_type == "blender"     -> official Blender spherical (pose_spherical)
      - path_type == "llff_spiral" -> official LLFF spiral (from poses_bounds.npy)
      - path_type == "llff_zflat"  -> official LLFF spiral 'zflat' variant

    Public API:
        generate(scene_val, n_frames, *, path_type, res_scale=..., data_root=..., **knobs)
          -> (poses: List[4x4 float32], H, W, K)
    """

    # ---------- ctor ----------
    def __init__(self, debug: bool = False) -> None:
        self.debug = bool(debug)

    # ---------- PUBLIC ----------
    def generate(
        self,
        scene_val,
        n_frames: int,
        *,
        path_type: str,
        res_scale: float = 1.0,
        # Blender knobs:
        bl_phi_deg: float = -30.0,
        bl_radius: Optional[float] = None,
        bl_theta_start_deg: float = -180.0,
        bl_rots: float = 1.0,
        # LLFF knobs:
        data_root: Optional[str | Path] = None,
        rots: float = 2.0,
        zrate: float = 0.5,
        path_zflat: bool = False,
        bd_factor: float = 0.75,
    ) -> Tuple[List[np.ndarray], int, int, np.ndarray]:
        """
        Returns: (poses: List[4x4 float32], H: int, W: int, K: 3x3 float32)
        """
        H, W, K = self._scaled_hwk(scene_val, res_scale)

        ptype = str(path_type).lower().strip()
        if ptype == "blender":
            if bl_radius is None:
                bl_radius = self._median_radius_from_scene(scene_val) or 4.0
            poses = self._generate_blender_path(
                n_frames=n_frames,
                phi_deg=float(bl_phi_deg),
                radius=float(bl_radius),
                theta_start_deg=float(bl_theta_start_deg),
                rots=float(bl_rots),
            )
            return poses, H, W, K

        if ptype in ("llff_spiral", "llff_zflat"):
            if data_root is None:
                raise ValueError("PathPoseGenerator.generate: LLFF paths require data_root pointing to the LLFF scene (poses_bounds.npy).")
            poses = self._generate_llff_spiral_from_poses_bounds(
                data_root=Path(data_root),
                n_frames=int(n_frames),
                rots=float(rots),
                zrate=float(zrate),
                path_zflat=bool(path_zflat),
                bd_factor=float(bd_factor),
            )
            return poses, H, W, K

        raise ValueError(f"Unsupported path_type '{path_type}'. Use 'blender' | 'llff_spiral' | 'llff_zflat'.")

    # ---------- HWK scaling ----------
    @staticmethod
    def _scaled_hwk(scene_val, res_scale: float) -> Tuple[int, int, np.ndarray]:
        base = scene_val.frames[0]
        H0, W0 = int(base.image.shape[0]), int(base.image.shape[1])
        K0 = np.array(base.K, dtype=np.float32)

        s = float(res_scale)
        if s != 1.0:
            H = max(1, int(round(H0 * s)))
            W = max(1, int(round(W0 * s)))
            K = K0.copy()
            K[0, 0] *= s; K[1, 1] *= s
            K[0, 2] *= s; K[1, 2] *= s
        else:
            H, W, K = H0, W0, K0
        return H, W, K

    # ---------- median radius (for Blender default) ----------
    @staticmethod
    def _median_radius_from_scene(scene_val) -> float:
        centers = np.stack([np.asarray(fr.c2w, np.float32)[:3, 3] for fr in scene_val.frames], axis=0)
        r = np.median(np.linalg.norm(centers, axis=1)).astype(np.float32)
        return float(r) if np.isfinite(r) and r > 1e-6 else 4.0

    # =========================================================
    # Blender spherical (official pose_spherical)
    # =========================================================
    @staticmethod
    def _trans_t(t: float) -> np.ndarray:
        M = np.eye(4, dtype=np.float32); M[2, 3] = float(t); return M

    @staticmethod
    def _rot_phi(phi_rad: float) -> np.ndarray:
        c, s = np.cos(phi_rad), np.sin(phi_rad)
        M = np.eye(4, dtype=np.float32)
        M[1,1]=c; M[1,2]=-s; M[2,1]=s; M[2,2]=c
        return M

    @staticmethod
    def _rot_theta(th_rad: float) -> np.ndarray:
        c, s = np.cos(th_rad), np.sin(th_rad)
        M = np.eye(4, dtype=np.float32)
        M[0,0]=c; M[0,2]=-s; M[2,0]=s; M[2,2]=c
        return M

    @classmethod
    def _pose_spherical_opengl(cls, theta_deg: float, phi_deg: float, radius: float) -> np.ndarray:
        c2w = cls._trans_t(radius)
        c2w = cls._rot_phi(np.deg2rad(phi_deg)) @ c2w
        c2w = cls._rot_theta(np.deg2rad(theta_deg)) @ c2w
        fix = np.array([[-1,0,0,0],
                        [ 0,0,1,0],
                        [ 0,1,0,0],
                        [ 0,0,0,1]], dtype=np.float32)
        return (fix @ c2w).astype(np.float32)

    def _generate_blender_path(
        self,
        n_frames: int,
        *,
        phi_deg: float,
        radius: float,
        theta_start_deg: float,
        rots: float,
    ) -> List[np.ndarray]:
        thetas = np.linspace(
            float(theta_start_deg),
            float(theta_start_deg) + 360.0 * float(rots),
            num=int(n_frames),
            endpoint=False,
            dtype=np.float32,
        )
        poses: List[np.ndarray] = [
            self._pose_spherical_opengl(float(th), float(phi_deg), float(radius))
            for th in thetas
        ]
        if self.debug:
            self._debug_facing_origin(poses, tag="[blender]")
        return poses

    # =========================================================
    # LLFF spiral (official, from poses_bounds.npy)
    # =========================================================
    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + eps)

    @staticmethod
    def _viewmatrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
        vec2 = PathPoseGenerator._normalize(z)
        vec1_avg = up
        vec0 = PathPoseGenerator._normalize(np.cross(vec1_avg, vec2))
        vec1 = PathPoseGenerator._normalize(np.cross(vec2, vec0))
        return np.stack([vec0, vec1, vec2, pos], axis=1).astype(np.float32)  # (3,4)

    @staticmethod
    def _recenter_poses(poses_3x5: np.ndarray) -> np.ndarray:
        poses = poses_3x5.copy()
        bottom = np.array([0, 0, 0, 1.], dtype=np.float32)[None, None, :]

        def poses_avg(poses):
            center = poses[:, :3, 3].mean(0)
            vec2   = PathPoseGenerator._normalize(poses[:, :3, 2].sum(0))
            up     = poses[:, :3, 1].sum(0)
            return PathPoseGenerator._viewmatrix(vec2, up, center)

        c2w = poses_avg(poses)
        c2w_4x4 = np.concatenate([c2w, np.array([[0,0,0,1]], np.float32)], 0)

        poses_4x4 = np.concatenate([poses[:, :3, :4], np.tile(bottom, (poses.shape[0],1,1))], 1)
        poses_4x4 = np.linalg.inv(c2w_4x4) @ poses_4x4
        poses[:, :3, :4] = poses_4x4[:, :3, :4]
        return poses

    @staticmethod
    def _render_path_spiral_official(
        c2w_3x5: np.ndarray,
        up_vec: np.ndarray,
        rads: np.ndarray,
        focal: float,
        zdelta: float,
        zrate: float,
        rots: float,
        N: int
    ) -> List[np.ndarray]:
        render_poses = []
        rads4 = np.array(list(rads) + [1.0], dtype=np.float32)
        hwf   = c2w_3x5[:, 4:5]

        for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1, dtype=np.float32)[:-1]:
            p4 = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0], dtype=np.float32) * rads4
            c  = (c2w_3x5[:3, :4] @ p4).astype(np.float32)
            p_focus = np.array([0.0, 0.0, -focal, 1.0], dtype=np.float32)
            z = PathPoseGenerator._normalize(c - (c2w_3x5[:3, :4] @ p_focus))
            vm = PathPoseGenerator._viewmatrix(z, up_vec, c)
            render_poses.append(np.concatenate([vm, hwf], axis=1))  # (3,5)

        return render_poses

    def _generate_llff_spiral_from_poses_bounds(
        self,
        *,
        data_root: Path,
        n_frames: int,
        rots: float,
        zrate: float,
        path_zflat: bool,
        bd_factor: float,
    ) -> List[np.ndarray]:
        pb_path = data_root / "poses_bounds.npy"
        if not pb_path.exists():
            raise FileNotFoundError(f"poses_bounds.npy not found at: {pb_path}")

        pb = np.load(str(pb_path)).astype(np.float32)          # (N,17)
        poses = pb[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # (3,5,N)
        bds   = pb[:, -2:].transpose([1, 0])                         # (2,N)

        # axis fix (y, -x, z)
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)         # (N,3,5)

        # scale so near ~ 1
        sc = 1.0 / (float(bds.min()) * float(bd_factor))
        poses[:, :3, 3] *= sc
        bds *= sc

        # recenter (exact)
        poses = self._recenter_poses(poses)                           # (N,3,5)

        # up & focal per bmild
        up = self._normalize(poses[:, :3, 1].sum(0))
        close_depth, inf_depth = float(bds.min() * 0.9), float(bds.max() * 5.0)
        dt = 0.75
        focal  = 1.0 / (((1.0 - dt) / close_depth) + (dt / inf_depth))
        zdelta = close_depth * 0.2

        # spiral radii from |centers|
        tt   = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, axis=0).astype(np.float32)  # (3,)

        # base avg pose (3x4) + hwf column (H,W,f) — H/W may be unset here; f used only for focus
        c2w = self._viewmatrix(self._normalize(poses[:, :3, 2].sum(0)), poses[:, :3, 1].sum(0), poses[:, :3, 3].mean(0))
        c2w_3x5 = np.concatenate([c2w, np.array([[0],[0],[focal]], np.float32)], 1)  # (3,5)

        # zflat tweak
        if path_zflat:
            zloc = -close_depth * 0.1
            c2w_3x5[:3, 3] = c2w_3x5[:3, 3] + zloc * c2w_3x5[:3, 2]
            rads[2] = 0.0
            rots = 1.0

        # build path (3x5) -> 4x4
        render_poses_3x5 = self._render_path_spiral_official(
            c2w_3x5=c2w_3x5, up_vec=up, rads=rads, focal=float(focal),
            zdelta=float(zdelta), zrate=float(zrate), rots=float(rots), N=int(n_frames)
        )

        poses_4x4: List[np.ndarray] = []
        for m in render_poses_3x5:
            c2w_3x4 = m[:, :4]
            c2w_4x4 = np.eye(4, dtype=np.float32)
            c2w_4x4[:3, :4] = c2w_3x4
            poses_4x4.append(c2w_4x4)

        if self.debug:
            fwd = np.stack([-p[:3, 2] for p in poses_4x4], 0)
            P   = np.stack([p[:3, 3] for p in poses_4x4], 0)
            focus_world = (c2w_3x5[:3, :4] @ np.array([0,0,-float(focal),1.0], np.float32))[:3]
            to_focus = (focus_world[None, :] - P); to_focus /= (np.linalg.norm(to_focus, 1, keepdims=True) + 1e-8)
            ang = np.degrees(np.arccos(np.clip((fwd * to_focus).sum(-1), -1.0, 1.0)))
            print(f"[llff_spiral_official] facing-to-FOCUS median={np.median(ang):.2f}°, p95={np.percentile(ang,95):.2f}°")
            print(f"[llff] rads={np.round(rads,6)}, focal={float(focal):.6f}")

        return poses_4x4

    # ---------- tiny debug ----------
    def _debug_facing_origin(self, poses: List[np.ndarray], tag: str = "") -> None:
        fwd = np.stack([-p[:3, 2] for p in poses], 0)  # OpenGL forward = -Z_cam
        P = np.stack([p[:3, 3] for p in poses], 0)
        to0 = -P; to0 /= (np.linalg.norm(to0, axis=1, keepdims=True) + 1e-8)
        ang = np.degrees(np.arccos(np.clip((fwd * to0).sum(-1), -1.0, 1.0)))
        print(f"{tag} facing-to-origin median={np.median(ang):.2f}°, p95={np.percentile(ang,95):.2f}°")
