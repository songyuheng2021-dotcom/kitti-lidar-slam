"""LiDAR odometry: scan-to-local-map point-to-plane ICP with a robust kernel,
constant-velocity motion prior and an adaptive correspondence threshold
(in the spirit of KISS-ICP, but written from scratch on top of Open3D primitives)."""
import numpy as np
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def make_pcd(xyz):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pc


class LocalMap:
    """Sliding-window voxel map of the last `window` scans, with normals."""

    def __init__(self, voxel=0.4, window=25, max_range=80.0):
        self.voxel, self.window, self.max_range = voxel, window, max_range
        self.scans = []          # list of (N,3) world-frame arrays
        self.pcd = None

    def add(self, xyz_world):
        self.scans.append(xyz_world)
        if len(self.scans) > self.window:
            self.scans.pop(0)

    def build(self, center):
        pts = np.concatenate(self.scans, 0)
        pts = pts[np.linalg.norm(pts[:, :2] - center[:2], axis=1) < self.max_range]
        pc = make_pcd(pts).voxel_down_sample(self.voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 3, max_nn=20))
        self.pcd = pc
        return pc


class LidarOdometry:
    def __init__(self, voxel_src=0.5, map_voxel=0.4, window=25, robust=True,
                 point_to_plane=True, scan_to_map=True, adaptive=True, init_thresh=3.0, max_thresh=2.0):
        self.voxel_src, self.robust = voxel_src, robust
        self.point_to_plane, self.scan_to_map, self.adaptive = point_to_plane, scan_to_map, adaptive
        self.map = LocalMap(map_voxel, window if scan_to_map else 1)
        self.poses = []          # list of 4x4 world-from-sensor
        self.thresh, self.max_thresh = init_thresh, max_thresh
        self.model_dev = []      # deviation between predicted and estimated motion

    # -------- helpers
    def _estimation(self):
        kernel = o3d.pipelines.registration.TukeyLoss(k=self.thresh) if self.robust else None
        if self.point_to_plane:
            return (o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel) if kernel
                    else o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return (o3d.pipelines.registration.TransformationEstimationPointToPoint())

    def _adapt(self, T_pred, T_est):
        """Adaptive threshold: 3 sigma of the recent prediction error (KISS-ICP style).
        The initial value must cover the per-frame motion (KITTI at 10 Hz moves ~1 m/frame,
        this course subset is subsampled to ~5 m/frame), otherwise ICP locks onto an
        aliased solution on self-similar scenes such as highways (seq 01)."""
        d = np.linalg.inv(T_pred) @ T_est
        dev = np.linalg.norm(d[:3, 3]) + 2 * 30 * np.sin(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)) / 2)
        self.model_dev.append(dev)
        if self.adaptive and len(self.model_dev) > 5:
            s = np.sqrt(np.mean(np.square(self.model_dev[-50:])))
            self.thresh = float(np.clip(3 * s, 0.3, self.max_thresh))

    # -------- main step
    def process(self, xyz):
        src_full = make_pcd(xyz)
        src = src_full.voxel_down_sample(self.voxel_src)
        if not self.poses:
            T = np.eye(4)
            self.poses.append(T)
            self.map.add(xyz)
            return T
        # constant-velocity prediction
        if len(self.poses) >= 2:
            T_pred = self.poses[-1] @ (np.linalg.inv(self.poses[-2]) @ self.poses[-1])
        else:
            T_pred = self.poses[-1].copy()
        target = self.map.build(self.poses[-1][:3, 3])
        est = self._estimation()
        T = T_pred
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
        for th in (self.thresh * 2, self.thresh):
            res = o3d.pipelines.registration.registration_icp(src, target, th, T, est, crit)
            T = res.transformation
        T = np.array(T)
        self._adapt(T_pred, T)
        self.poses.append(T)
        self.map.add((T[:3, :3] @ xyz.T).T + T[:3, 3])
        return T
