#!/usr/bin/env python3
"""Collect metrics from results/ into markdown tables and a summary figure."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lidar_slam.evaluation import umeyama_se3


def align(est, gt):
    R, t = umeyama_se3(est[:, :3, 3], gt[:, :3, 3])
    return est[:, :3, 3] @ R.T + t


def main():
    rows, seqs = [], []
    for f in sorted(glob.glob("results/0*/metrics.json")):
        m = json.load(open(f)); o = m["odometry"]; p = m.get("optimized")
        seqs.append(m["seq"])
        rows.append(f"| {m['seq']} | {m['frames']} | {o['path_m']:.0f} | {m['loops']} | "
                    f"{o['kitti_t_pct']:.2f} / {o['kitti_r_deg_100m']:.2f} | {o['ate_m']:.2f} | "
                    f"{p['ate_m']:.2f} | {o['end_err_m']:.1f} | {p['end_err_m']:.1f} |" if p else
                    f"| {m['seq']} | {m['frames']} | {o['path_m']:.0f} | 0 | {o['kitti_t_pct']:.2f} / {o['kitti_r_deg_100m']:.2f} | "
                    f"{o['ate_m']:.2f} | - | {o['end_err_m']:.1f} | - |")
    main_tbl = ("| seq | frames | path [m] | loops | odom drift t[%] / r[deg/100m] | ATE odom [m] | ATE +PGO [m] | "
                "end err odom [m] | end err +PGO [m] |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))

    names = {"full": "full system (scan-to-map, point-to-plane, Tukey, adaptive thr.)",
             "p2p": "point-to-point instead of point-to-plane",
             "s2s": "scan-to-scan instead of scan-to-local-map",
             "norobust": "no robust kernel",
             "noadaptive": "fixed 3 m threshold instead of adaptive"}
    ab = []
    for k in ["full", "p2p", "s2s", "norobust", "noadaptive"]:
        f = f"results/ablation/{k}/metrics.json"
        if not os.path.exists(f):
            continue
        o = json.load(open(f))["odometry"]
        ab.append(f"| {names[k]} | {o['kitti_t_pct']:.2f} | {o['kitti_r_deg_100m']:.2f} | {o['ate_m']:.2f} | {o['end_err_m']:.1f} |")
    ab_tbl = ("| variant (seq 00, odometry only) | drift t [%] | drift r [deg/100m] | ATE [m] | end err [m] |\n"
              "|---|---|---|---|---|\n" + "\n".join(ab))
    open("results/tables.md", "w").write("## Per-sequence results\n\n" + main_tbl + "\n\n## Ablation\n\n" + ab_tbl + "\n")
    print(main_tbl); print(); print(ab_tbl)

    # summary figure
    fig, axes = plt.subplots(1, len(seqs), figsize=(4.2 * len(seqs), 4.2))
    for ax, s in zip(np.atleast_1d(axes), seqs):
        gt = np.load(f"results/{s}/poses_gt.npy"); od = np.load(f"results/{s}/poses_odom.npy"); fn = np.load(f"results/{s}/poses_final.npy")
        ax.plot(gt[:, 0, 3], gt[:, 1, 3], "k--", lw=1, label="GT")
        p = align(od, gt); ax.plot(p[:, 0], p[:, 1], "tab:red", lw=0.9, label="odometry")
        p = align(fn, gt); ax.plot(p[:, 0], p[:, 1], "tab:blue", lw=0.9, label="+ loop closure")
        ax.set_title(f"seq {s}"); ax.axis("equal"); ax.grid(alpha=0.3); ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/summary_trajectories.png", dpi=140)


if __name__ == "__main__":
    main()
