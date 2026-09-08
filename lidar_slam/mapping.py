"""Global map products from a trajectory: a 2-D log-odds occupancy grid
(vectorised ray casting) and a voxelised 3-D point-cloud map."""
import numpy as np
import open3d as o3d
from .odometry import make_pcd


class OccupancyGrid:
    def __init__(self, xmin, xmax, ymin, ymax, res=0.25, l_occ=np.log(9), l_free=np.log(1 / 3), l_max=8.0):
        self.res, self.xmin, self.ymin = res, xmin, ymin
        self.W = int(np.ceil((xmax - xmin) / res)) + 1
        self.H = int(np.ceil((ymax - ymin) / res)) + 1
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)
        self.l_occ, self.l_free, self.l_max = l_occ, l_free, l_max

    def cell(self, xy):
        c = np.floor((xy[:, 0] - self.xmin) / self.res).astype(int)
        r = np.floor((xy[:, 1] - self.ymin) / self.res).astype(int)
        return np.clip(r, 0, self.H - 1), np.clip(c, 0, self.W - 1)

    def integrate(self, T, xyz, free_every=6, z_band=(-1.0, 2.5)):
        """T: sensor pose (4x4 world-from-sensor). xyz: points in sensor frame."""
        z = xyz[:, 2]
        pts = xyz[(z > z_band[0]) & (z < z_band[1])]
        pw = (T[:3, :3] @ pts.T).T + T[:3, 3]
        o = T[:2, 3]
        # free space along a subset of rays (vectorised "linspace Bresenham")
        ends = pw[::free_every, :2]
        n = np.maximum(2, np.ceil(np.linalg.norm(ends - o, axis=1) / self.res).astype(int))
        nmax = n.max()
        s = np.linspace(0, 1, nmax)[None, :, None]
        ray = o[None, None, :] + (ends - o)[:, None, :] * s          # M x nmax x 2
        valid = (np.arange(nmax)[None, :] < (n[:, None] - 1))           # exclude endpoint cell
        r, c = self.cell(ray[valid])
        np.add.at(self.log_odds, (r, c), self.l_free)
        r, c = self.cell(pw[:, :2])
        np.add.at(self.log_odds, (r, c), self.l_occ)
        np.clip(self.log_odds, -self.l_max, self.l_max, out=self.log_odds)

    def image(self):
        p = 1 / (1 + np.exp(-self.log_odds))
        img = np.full(p.shape, 0.5, dtype=np.float32)
        img[p > 0.65] = 0.0
        img[p < 0.35] = 1.0
        return img


def build_occupancy(poses, scans, res=0.25, margin=70.0):
    P = np.asarray(poses)[:, :2, 3]
    g = OccupancyGrid(P[:, 0].min() - margin, P[:, 0].max() + margin,
                      P[:, 1].min() - margin, P[:, 1].max() + margin, res=res)
    for T, xyz in zip(poses, scans):
        g.integrate(T, xyz)
    return g


def build_pointcloud_map(poses, scans, voxel=0.3, every=1):
    pts = []
    for T, xyz in list(zip(poses, scans))[::every]:
        pts.append((T[:3, :3] @ xyz.T).T + T[:3, 3])
    return make_pcd(np.concatenate(pts)).voxel_down_sample(voxel)
