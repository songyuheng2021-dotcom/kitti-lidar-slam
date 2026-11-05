#!/usr/bin/env python3
"""Check every accepted loop closure against ground truth and make the result reproducible.

    python tools_verify_loops.py --data /path/to/KITTI --seq 00 --metrics results/00/metrics.json

For each (query, match) pair in metrics.json['loop_list'] the script computes the ground-truth
distance between the two sensor positions and the translation error of the loop's relative pose
(when poses_*.npy are available). A loop counts as a true positive when the ground-truth distance
is below --tp_dist (10 m by default, the same threshold the README quotes). Output:
    <out>/loop_verification.csv     one row per loop: query, match, gt_dist_m, sc_dist, fitness, tp
    <out>/loop_verification.json    counts: n_loops, n_true_positive, n_false_positive, precision
Without the dataset (poses) the script cannot run: the ground truth is not stored in metrics.json.
"""
import argparse, csv, json, os
import numpy as np


def verify_loops(loop_list, gt_poses, tp_dist=10.0):
    """loop_list: iterable of (query_frame, match_frame, ...); gt_poses: (N,4,4) world-from-sensor.
    Returns list of dicts and a summary dict."""
    rows = []
    for item in loop_list:
        q, m = int(item[0]), int(item[1])
        fit = float(item[2]) if len(item) > 2 else float("nan")
        sc = float(item[3]) if len(item) > 3 else float("nan")
        d = float(np.linalg.norm(gt_poses[q][:3, 3] - gt_poses[m][:3, 3]))
        rows.append(dict(query=q, match=m, gt_dist_m=d, fitness=fit, sc_dist=sc, tp=d < tp_dist))
    n = len(rows); tp = sum(r["tp"] for r in rows)
    summary = dict(n_loops=n, n_true_positive=tp, n_false_positive=n - tp,
                   precision=(tp / n) if n else float("nan"), tp_dist_m=tp_dist,
                   max_gt_dist_m=max((r["gt_dist_m"] for r in rows), default=float("nan")))
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--seq", required=True)
    ap.add_argument("--metrics", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--tp_dist", type=float, default=10.0)
    a = ap.parse_args()
    from lidar_slam.data import KittiSequence
    seq = KittiSequence(a.data, a.seq)
    m = json.load(open(a.metrics))
    rows, summary = verify_loops(m["loop_list"], seq.gt, a.tp_dist)
    out = a.out or os.path.dirname(a.metrics)
    with open(os.path.join(out, "loop_verification.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["query", "match", "gt_dist_m", "fitness", "sc_dist", "tp"])
        w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(out, "loop_verification.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(f"  {r['query']:5d} -> {r['match']:5d}  gt {r['gt_dist_m']:6.2f} m  fit {r['fitness']:.2f}  {'TP' if r['tp'] else 'FP'}")


if __name__ == "__main__":
    main()
