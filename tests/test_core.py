"""Unit tests: run with `pytest -q`."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lidar_slam.evaluation import ate, rpe, kitti_drift, umeyama_se3
from lidar_slam.loop_closure import ScanContext
from lidar_slam.pose_graph import optimize
from lidar_slam.mapping import OccupancyGrid


def _rot_z(a):
    T = np.eye(4); T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]; return T


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = _rot_z(0.7)[:3, :3]; t = np.array([1, 2, 3.0])
    R2, t2 = umeyama_se3(src, src @ R.T + t)
    assert np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-8)


def test_metrics_zero_for_identical_trajectories():
    T = np.tile(np.eye(4), (100, 1, 1)); T[:, 0, 3] = np.arange(100) * 2.0
    assert ate(T, T) < 1e-9
    assert rpe(T, T)[0] < 1e-9
    d = kitti_drift(T, T, lengths=(100,))
    assert d["n_segments"] > 0 and d["t_rel_pct"] < 1e-9


def test_scan_context_is_yaw_invariant():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-40, 40, size=(5000, 3)); pts[:, 2] = rng.uniform(-1, 3, 5000)
    sc = ScanContext()
    d1 = sc.describe(pts)
    yaw = 2 * np.pi * 12 / sc.n_sector                     # rotate by exactly 12 sectors
    d2 = sc.describe((_rot_z(yaw)[:3, :3] @ pts.T).T)
    dist, shift = ScanContext.distance(d1, d2)
    assert dist < 0.05 and shift in (12, sc.n_sector - 12)


def test_pose_graph_closes_loop():
    poses = [np.eye(4)]
    for _ in range(60):                                     # drifting odometry: 1 m/frame + lateral bias
        T = poses[-1].copy(); T[:3, 3] += [1.0, 0.05, 0.0]; poses.append(T)
    T_true_last = np.eye(4); T_true_last[0, 3] = 60.0       # loop says: frame 60 is at (60,0,0) w.r.t. frame 0
    loops = [(60, 0, T_true_last, 0.95, 0.1)]
    out, _ = optimize(poses, loops)
    assert np.linalg.norm(out[-1, :3, 3] - [60, 0, 0]) < 0.3


def test_occupancy_grid_marks_hit_and_free():
    g = OccupancyGrid(-10, 10, -10, 10, res=0.5)
    pts = np.array([[5.0, 0.0, 0.0]] * 20)
    for _ in range(5):
        g.integrate(np.eye(4), pts, free_every=1)
    r, c = g.cell(np.array([[5.0, 0.0]]))
    r0, c0 = g.cell(np.array([[2.0, 0.0]]))
    assert g.log_odds[r, c] > 0 and g.log_odds[r0, c0] < 0


def test_occupancy_short_ray_is_cleared_along_its_whole_length():
    """Regression: with a 20 m and a 5 m ray in the same scan, the 5 m ray must clear free space
    all the way to ~4.5 m (not just its first metre), and no free cell may be hit more than once per ray."""
    g = OccupancyGrid(-25, 25, -25, 25, res=0.5)
    pts = np.array([[20.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    g.integrate(np.eye(4), pts, free_every=1)
    r, c = g.cell(np.array([[0.0, 4.0]]))          # 1 m before the 5 m hit, along +y
    assert g.log_odds[r, c] < 0
    r, c = g.cell(np.array([[0.0, 1.0]]))
    assert abs(g.log_odds[r, c] - g.l_free) < 1e-5  # exactly one free update, not several
    r, c = g.cell(np.array([[19.0, 0.0]]))
    assert g.log_odds[r, c] < 0


def test_verify_loops_counts_true_and_false_positives():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from tools_verify_loops import verify_loops
    gt = np.tile(np.eye(4), (4, 1, 1))
    gt[1, :3, 3] = [3, 0, 0]; gt[2, :3, 3] = [50, 0, 0]; gt[3, :3, 3] = [52, 0, 0]
    rows, s = verify_loops([(1, 0, 0.9, 0.1), (2, 0, 0.8, 0.2), (3, 2, 0.95, 0.05)], gt, tp_dist=10.0)
    assert s["n_loops"] == 3 and s["n_true_positive"] == 2 and s["n_false_positive"] == 1
    assert abs(rows[1]["gt_dist_m"] - 50.0) < 1e-9 and rows[2]["tp"]
