"""Loop closure: Scan Context place recognition (Kim & Kim, IROS 2018) written from
scratch, followed by geometric verification with ICP against a keyframe submap."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .odometry import make_pcd


class ScanContext:
    def __init__(self, n_ring=20, n_sector=60, max_range=60.0):
        self.n_ring, self.n_sector, self.max_range = n_ring, n_sector, max_range

    def describe(self, xyz):
        r = np.linalg.norm(xyz[:, :2], axis=1)
        m = r < self.max_range
        r, x, y, z = r[m], xyz[m, 0], xyz[m, 1], xyz[m, 2] + 2.0   # shift z so heights are positive
        ring = np.minimum((r / self.max_range * self.n_ring).astype(int), self.n_ring - 1)
        theta = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
        sector = np.minimum((theta * self.n_sector).astype(int), self.n_sector - 1)
        desc = np.zeros((self.n_ring, self.n_sector))
        np.maximum.at(desc, (ring, sector), z)
        return desc

    @staticmethod
    def ring_key(desc):
        return desc.mean(1)

    @staticmethod
    def distance(d1, d2):
        """Cosine distance between column vectors, minimised over column (yaw) shifts."""
        best, best_shift = 1.0, 0
        n1 = np.linalg.norm(d1, axis=0) + 1e-9
        for s in range(d2.shape[1]):
            d2s = np.roll(d2, s, axis=1)
            n2 = np.linalg.norm(d2s, axis=0) + 1e-9
            cos = np.sum(d1 * d2s, axis=0) / (n1 * n2)
            dist = 1 - cos.mean()
            if dist < best:
                best, best_shift = dist, s
        return best, best_shift


class LoopDetector:
    def __init__(self, min_gap=60, ring_k=8, sc_thresh=0.22, icp_fitness=0.8, icp_rmse=0.3,
                 max_dist_prior=None, verbose=False):
        self.sc = ScanContext()
        self.min_gap, self.ring_k, self.sc_thresh = min_gap, ring_k, sc_thresh
        self.icp_fitness, self.icp_rmse, self.max_dist_prior, self.verbose = icp_fitness, icp_rmse, max_dist_prior, verbose
        self.descs, self.keys, self.scans, self.frame_ids = [], [], [], []
        self.loops = []          # (i_query, j_match, T_j_i, fitness)
        self.candidates_checked = 0

    def add_keyframe(self, frame_id, xyz):
        self.frame_ids.append(frame_id)
        d = self.sc.describe(xyz)
        self.descs.append(d)
        self.keys.append(self.sc.ring_key(d))
        self.scans.append(xyz)

    def _submap(self, k, poses, half=3):
        """Stack keyframes k-half..k+half in the frame of keyframe k."""
        pts = []
        Tk_inv = np.linalg.inv(poses[self.frame_ids[k]])
        for j in range(max(0, k - half), min(len(self.scans), k + half + 1)):
            T = Tk_inv @ poses[self.frame_ids[j]]
            pts.append((T[:3, :3] @ self.scans[j].T).T + T[:3, 3])
        pc = make_pcd(np.concatenate(pts)).voxel_down_sample(0.4)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.2, max_nn=20))
        return pc

    def detect(self, poses):
        """Try to close a loop for the most recent keyframe. `poses` = current odometry poses."""
        q = len(self.descs) - 1
        old = [k for k in range(q) if self.frame_ids[q] - self.frame_ids[k] > self.min_gap]
        if len(old) < self.ring_k:
            return None
        tree = cKDTree(np.array([self.keys[k] for k in old]))
        _, nn = tree.query(self.keys[q], k=self.ring_k)
        best = None
        for n in np.atleast_1d(nn):
            k = old[n]
            if self.max_dist_prior is not None:
                dp = np.linalg.norm(poses[self.frame_ids[q]][:3, 3] - poses[self.frame_ids[k]][:3, 3])
                if dp > self.max_dist_prior:
                    continue
            dist, shift = self.sc.distance(self.descs[q], self.descs[k])
            if dist < self.sc_thresh and (best is None or dist < best[0]):
                best = (dist, k, shift)
        if best is None:
            return None
        dist, k, shift = best
        self.candidates_checked += 1
        # geometric verification: align query scan to submap around k
        yaw = -shift / self.sc.n_sector * 2 * np.pi
        init = np.eye(4)
        init[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        src = make_pcd(self.scans[q]).voxel_down_sample(0.5)
        tgt = self._submap(k, poses)
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=1.0))
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        T = init
        for th in (4.0, 2.0, 1.0):
            res = o3d.pipelines.registration.registration_icp(src, tgt, th, T, est, crit)
            T = res.transformation
        res = o3d.pipelines.registration.evaluate_registration(src, tgt, 0.5, T)
        if self.verbose:
            print(f"  loop cand {self.frame_ids[q]}->{self.frame_ids[k]} sc={dist:.3f} fit={res.fitness:.2f} rmse={res.inlier_rmse:.2f}")
        if res.fitness < self.icp_fitness or res.inlier_rmse > self.icp_rmse:
            return None
        loop = (self.frame_ids[q], self.frame_ids[k], np.array(T), res.fitness, dist)
        self.loops.append(loop)
        return loop
