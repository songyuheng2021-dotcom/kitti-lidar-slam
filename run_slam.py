#!/usr/bin/env python3
"""Run the full LiDAR SLAM pipeline on one KITTI sequence.

    python run_slam.py --data ../data --seq 00 --out results/00
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d

from lidar_slam.data import KittiSequence
from lidar_slam.odometry import LidarOdometry
from lidar_slam.loop_closure import LoopDetector
from lidar_slam.pose_graph import optimize
from lidar_slam.mapping import build_occupancy, build_pointcloud_map
from lidar_slam.evaluation import summarize, umeyama_se3


def align(est, gt):
    R, t = umeyama_se3(est[:, :3, 3], gt[:, :3, 3])
    return est[:, :3, 3] @ R.T + t


def plot_traj(gt, trajs, loops, path, title):
    plt.figure(figsize=(7, 7))
    plt.plot(gt[:, 0, 3], gt[:, 1, 3], "k--", lw=1.2, label="ground truth")
    for name, T, style in trajs:
        p = align(T, gt)
        plt.plot(p[:, 0], p[:, 1], style, lw=1.2, label=name)
    if loops:
        T = trajs[-1][1]
        p = align(T, gt)
        for (i, j, *_) in loops:
            plt.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], color="tab:green", lw=0.6, alpha=0.8)
        plt.plot([], [], color="tab:green", lw=0.6, label=f"{len(loops)} loop closures")
    plt.axis("equal"); plt.grid(alpha=0.3); plt.legend(); plt.title(title)
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--keyframe_every", type=int, default=2)
    ap.add_argument("--sc_thresh", type=float, default=0.22)
    ap.add_argument("--map_res", type=float, default=0.25)
    ap.add_argument("--gif", action="store_true")
    # ablation switches
    ap.add_argument("--no_robust", action="store_true")
    ap.add_argument("--point_to_point", action="store_true")
    ap.add_argument("--scan_to_scan", action="store_true")
    ap.add_argument("--no_adaptive", action="store_true")
    a = ap.parse_args()
    out = a.out or f"results/{a.seq}"
    os.makedirs(out, exist_ok=True)

    seq = KittiSequence(a.data, a.seq, a.max_frames)
    odo = LidarOdometry(robust=not a.no_robust, point_to_plane=not a.point_to_point,
                        scan_to_map=not a.scan_to_scan, adaptive=not a.no_adaptive)
    det = LoopDetector(sc_thresh=a.sc_thresh, verbose=True)
    scans, loops = [], []
    t0 = time.time()
    for i in range(len(seq)):
        xyz = seq[i]
        scans.append(xyz)
        odo.process(xyz)
        if not a.no_loop and i % a.keyframe_every == 0:
            det.add_keyframe(i, xyz)
            lp = det.detect(odo.poses)
            if lp is not None:
                loops.append(lp)
                print(f"[{i:4d}] loop closure {lp[0]} -> {lp[1]}  fitness={lp[3]:.2f}")
        if i % 100 == 0:
            print(f"[{i:4d}/{len(seq)}] {time.time() - t0:.0f}s  thresh={odo.thresh:.2f}")
    t_odo = time.time() - t0
    odom = np.array(odo.poses)

    result = {"seq": a.seq, "frames": len(seq), "time_s": t_odo, "fps": len(seq) / t_odo,
              "odometry": summarize(odom, seq.gt), "loops": len(loops)}
    trajs = [("LiDAR odometry", odom, "tab:red")]
    final = odom
    if loops:
        t1 = time.time()
        opt, info = optimize(list(odom), loops)
        result["pgo_time_s"] = time.time() - t1
        result["pgo"] = info
        result["optimized"] = summarize(opt, seq.gt)
        trajs.append(("odometry + loop closure (PGO)", opt, "tab:blue"))
        final = opt
    result["loop_list"] = [(int(i), int(j), float(f), float(d)) for (i, j, T, f, d) in loops]
    print(json.dumps({k: v for k, v in result.items() if k != "loop_list"}, indent=1))
    json.dump(result, open(f"{out}/metrics.json", "w"), indent=1)
    np.save(f"{out}/poses_odom.npy", odom)
    np.save(f"{out}/poses_final.npy", final)
    np.save(f"{out}/poses_gt.npy", seq.gt)

    plot_traj(seq.gt, trajs, loops, f"{out}/trajectory.png",
              f"KITTI {a.seq}: ATE odom {result['odometry']['ate_m']:.2f} m"
              + (f", PGO {result['optimized']['ate_m']:.2f} m" if loops else ""))

    # maps
    grid = build_occupancy(final, scans, res=a.map_res)
    img = grid.image()
    plt.imsave(f"{out}/occupancy_map.png", img, cmap="gray", origin="lower")
    pcd = build_pointcloud_map(final, scans, voxel=0.3)
    o3d.io.write_point_cloud(f"{out}/map.pcd", pcd)
    # overlay trajectory on map
    plt.figure(figsize=(9, 9))
    plt.imshow(img, cmap="gray", origin="lower",
               extent=[grid.xmin, grid.xmin + grid.W * grid.res, grid.ymin, grid.ymin + grid.H * grid.res])
    plt.plot(final[:, 0, 3], final[:, 1, 3], "r-", lw=0.8, label="estimated trajectory")
    plt.legend(); plt.title(f"KITTI {a.seq} occupancy grid ({a.map_res} m) + trajectory")
    plt.tight_layout(); plt.savefig(f"{out}/map_with_trajectory.png", dpi=150); plt.close()

    if a.gif:
        import imageio
        frames = []
        P = final[:, :2, 3]
        g2 = build_occupancy(final[:1], scans[:1], res=0.5)
        from lidar_slam.mapping import OccupancyGrid
        g2 = OccupancyGrid(P[:, 0].min() - 70, P[:, 0].max() + 70, P[:, 1].min() - 70, P[:, 1].max() + 70, res=0.5)
        step = max(1, len(seq) // 120)
        for k in range(len(seq)):
            g2.integrate(final[k], scans[k])
            if k % step == 0 or k == len(seq) - 1:
                fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111)
                ax.imshow(g2.image(), cmap="gray", origin="lower",
                          extent=[g2.xmin, g2.xmin + g2.W * g2.res, g2.ymin, g2.ymin + g2.H * g2.res])
                ax.plot(P[:k + 1, 0], P[:k + 1, 1], "r-", lw=0.8); ax.plot(P[k, 0], P[k, 1], "ro", ms=3)
                ax.set_title(f"KITTI {a.seq}  frame {k}"); ax.set_xticks([]); ax.set_yticks([])
                fig.tight_layout(); fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
        imageio.mimsave(f"{out}/mapping.gif", frames, duration=0.08)
    print("done ->", out)


if __name__ == "__main__":
    main()
