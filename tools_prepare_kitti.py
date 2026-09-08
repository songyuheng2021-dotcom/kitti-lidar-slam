#!/usr/bin/env python3
"""Convert raw KITTI Velodyne .bin scans into compact filtered .npy frames (the `lite` format
used by this repo): range 2.5-60 m, height -1.4..3.0 m above the sensor (drops most ground),
0.25 m voxel de-duplication, float16. ~80 KB per frame instead of ~2 MB.

    python tools_prepare_kitti.py --root /path/to/KITTI --seq 00 01 02 03
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seq", nargs="+", default=["00"])
ap.add_argument("--voxel", type=float, default=0.25)
a = ap.parse_args()
for seq in a.seq:
    d = os.path.join(a.root, "odometry", seq, "velodyne")
    out = os.path.join(a.root, "lite", seq); os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(d) if f.endswith(".bin"))
    for f in files:
        p = np.fromfile(os.path.join(d, f), dtype=np.float32).reshape(-1, 4)
        r = np.linalg.norm(p[:, :2], axis=1)
        p = p[(r < 60) & (r > 2.5) & (p[:, 2] > -1.4) & (p[:, 2] < 3.0), :3]
        _, idx = np.unique(np.floor(p / a.voxel).astype(np.int32), axis=0, return_index=True)
        np.save(os.path.join(out, f[:-4] + ".npy"), p[idx].astype(np.float16))
    print(seq, len(files), "frames")
