"""Trajectory metrics: ATE (after SE(3) Umeyama alignment), RPE, and the official
KITTI odometry drift metric (translation % and rotation deg/100m over segments)."""
import numpy as np


def umeyama_se3(src, dst):
    """Find R,t minimising ||R src + t - dst|| (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return R, mu_d - R @ mu_s


def ate(est, gt):
    """est, gt: (N,4,4). Returns RMSE of aligned positions in metres."""
    p_e, p_g = est[:, :3, 3], gt[:, :3, 3]
    R, t = umeyama_se3(p_e, p_g)
    err = (p_e @ R.T + t) - p_g
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))


def rpe(est, gt, delta=1):
    """Relative pose error over `delta` frames: (trans RMSE m, rot RMSE deg)."""
    dt, dr = [], []
    for i in range(len(est) - delta):
        de = np.linalg.inv(est[i]) @ est[i + delta]
        dg = np.linalg.inv(gt[i]) @ gt[i + delta]
        e = np.linalg.inv(dg) @ de
        dt.append(np.linalg.norm(e[:3, 3]))
        dr.append(np.degrees(np.arccos(np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1))))
    return float(np.sqrt(np.mean(np.square(dt)))), float(np.sqrt(np.mean(np.square(dr))))


def _traj_dist(T):
    d = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(d)])


def kitti_drift(est, gt, lengths=(100, 200, 300, 400, 500, 600, 700, 800), step=10):
    """Official KITTI devkit metric: for every start frame (every `step`) and every
    segment length, relative translational error (%) and rotational error (deg/m)."""
    dist = _traj_dist(gt)
    errs = []
    for first in range(0, len(gt), step):
        for L in lengths:
            idx = np.searchsorted(dist, dist[first] + L)
            if idx >= len(gt):
                continue
            dg = np.linalg.inv(gt[first]) @ gt[idx]
            de = np.linalg.inv(est[first]) @ est[idx]
            e = np.linalg.inv(de) @ dg
            t_err = np.linalg.norm(e[:3, 3]) / L
            r_err = np.arccos(np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1)) / L
            errs.append((L, t_err, r_err))
    if not errs:
        return dict(t_rel_pct=float("nan"), r_rel_deg_per_100m=float("nan"), n_segments=0, per_length={})
    errs = np.array(errs)
    per = {int(L): (float(errs[errs[:, 0] == L, 1].mean() * 100), float(np.degrees(errs[errs[:, 0] == L, 2].mean()) * 100))
           for L in np.unique(errs[:, 0])}
    return dict(t_rel_pct=float(errs[:, 1].mean() * 100),
                r_rel_deg_per_100m=float(np.degrees(errs[:, 2].mean()) * 100),
                n_segments=len(errs), per_length=per)


def path_length(T):
    return float(_traj_dist(T)[-1])


def summarize(est, gt):
    est, gt = np.asarray(est), np.asarray(gt)
    d = kitti_drift(est, gt)
    t1, r1 = rpe(est, gt, 1)
    return dict(frames=len(est), path_m=path_length(gt), ate_m=ate(est, gt),
                rpe1_t_m=t1, rpe1_r_deg=r1, end_err_m=float(np.linalg.norm(est[-1, :3, 3] - gt[-1, :3, 3])),
                kitti_t_pct=d["t_rel_pct"], kitti_r_deg_100m=d["r_rel_deg_per_100m"], kitti_segments=d["n_segments"])
