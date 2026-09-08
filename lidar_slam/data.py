"""KITTI odometry loader (works on the compact 'lite' frames or raw .bin files)."""
import os
import numpy as np


def load_calib(path):
    with open(path) as f:
        for line in f:
            if line.startswith("Tr:"):
                v = np.array([float(x) for x in line.split(":", 1)[1].split()])
                Tr = np.eye(4)
                Tr[:3] = v.reshape(3, 4)
                return Tr  # velodyne -> cam0
    raise ValueError("no Tr in calib")


def load_poses(path):
    P = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(P), 1, 1))
    T[:, :3] = P
    return T  # T_w_cam0


class KittiSequence:
    """Frames are returned in the Velodyne frame (x fwd, y left, z up).
    Ground truth is converted to the Velodyne frame and expressed relative to frame 0."""

    def __init__(self, root, seq, max_frames=None):
        self.root, self.seq = root, seq
        self.Tr = load_calib(os.path.join(root, "calib", seq, "calib.txt"))
        Tc = load_poses(os.path.join(root, "poses", f"{seq}.txt"))
        Tv = Tc @ self.Tr[None]                     # T_w_velo (world = cam0 of frame 0)
        Tv = np.linalg.inv(Tv[0])[None] @ Tv        # relative to first velodyne pose
        self.gt = Tv
        lite = os.path.join(root, "lite", seq)
        if os.path.isdir(lite):
            self.files = sorted(os.path.join(lite, f) for f in os.listdir(lite) if f.endswith(".npy"))
        else:
            d = os.path.join(root, "odometry", seq, "velodyne")
            self.files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".bin"))
        n = min(len(self.files), len(self.gt))
        if max_frames:
            n = min(n, max_frames)
        self.files, self.gt = self.files[:n], self.gt[:n]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        if f.endswith(".npy"):
            return np.load(f).astype(np.float32)
        p = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        r = np.linalg.norm(p[:, :2], axis=1)
        m = (r < 60) & (r > 2.5) & (p[:, 2] > -1.4) & (p[:, 2] < 3.0)
        return p[m, :3]
