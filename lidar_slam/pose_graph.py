"""SE(3) pose-graph optimisation with GTSAM (Levenberg-Marquardt, robust loop factors)."""
import numpy as np
import gtsam


def optimize(poses, loops, odom_sigma=(0.05, 0.05, 0.05, 0.01, 0.01, 0.01),
             loop_sigma=(0.3, 0.3, 0.3, 0.03, 0.03, 0.03), huber=1.0):
    """Two-stage optimisation: (1) all loop factors with a plain Gaussian model, which lets
    large-but-correct loop residuals (accumulated drift) pull the graph into place;
    (2) re-optimise from that solution with Huber-robustified loop factors so that any
    remaining inconsistent loop is down-weighted instead of distorting the trajectory.
    poses: list of 4x4 odometry poses. loops: (i, j, T_j_i, ...) with T_j_i = pose_j^-1 pose_i."""
    stage1, info1 = _optimize(poses, loops, odom_sigma, loop_sigma, huber=None)
    out, info2 = _optimize(list(stage1), loops, odom_sigma, loop_sigma, huber=huber, init_poses=stage1, odom_from=poses)
    return out, dict(stage1=info1, stage2=info2)


def _optimize(poses, loops, odom_sigma, loop_sigma, huber=None, init_poses=None, odom_from=None):
    odom_from = poses if odom_from is None else odom_from
    init_poses = poses if init_poses is None else init_poses
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4] * 6))
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odom_sigma[3:] + odom_sigma[:3]))  # gtsam order: rot, trans
    loop_base = gtsam.noiseModel.Diagonal.Sigmas(np.array(loop_sigma[3:] + loop_sigma[:3]))
    loop_noise = (gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(huber), loop_base)
                  if huber else loop_base)

    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(odom_from[0]), prior))
    for i in range(1, len(poses)):
        rel = np.linalg.inv(odom_from[i - 1]) @ odom_from[i]
        graph.add(gtsam.BetweenFactorPose3(i - 1, i, gtsam.Pose3(rel), odom_noise))
    for (i, j, T_j_i, *_) in loops:
        graph.add(gtsam.BetweenFactorPose3(j, i, gtsam.Pose3(T_j_i), loop_noise))
    for i, T in enumerate(init_poses):
        init.insert(i, gtsam.Pose3(T))
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    result = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
    out = np.array([result.atPose3(i).matrix() for i in range(len(poses))])
    return out, dict(error_before=graph.error(init), error_after=graph.error(result))
