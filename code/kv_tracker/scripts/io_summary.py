#!/usr/bin/env python3
"""Input/output summary of one KV-Tracker run on a TUM scene.

    python scripts/io_summary.py <results_dir> <scene_dir>

Prints what went in (frames, resolution) and what came out (traj.npy, kf_poses.npy,
kf_idx.npy, pcd_*.ply, kf_*.png) with shapes and sanity checks, and writes
<results_dir>/io_summary.png: sample input frames, est-vs-GT trajectory (Sim(3)
aligned with evo, same as the score), the last exported point cloud, keyframes.
Run under the kvtracker env.
"""
import glob
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from evo.core.trajectory import PoseTrajectory3D

sys.path.insert(0, "/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
from kv_tracker.dataloaders.tum import TUMLoader  # noqa: E402

res, scene = Path(sys.argv[1]), Path(sys.argv[2])

# ---------------- INPUT ----------------
rgb_paths = sorted(glob.glob(str(scene / "rgb" / "*.png")))
h, w = cv2.imread(rgb_paths[0]).shape[:2]
print("=== INPUT ===")
print(f"scene         : {scene}")
print(f"RGB frames    : {len(rgb_paths)}  ({w}x{h} px, monocular, no depth used)")
print(f"ground truth  : {scene/'groundtruth.txt'}  (motion capture, used only for scoring)")

# ---------------- OUTPUT ----------------
traj = np.load(res / "traj.npy")
kf_poses = np.load(res / "kf_poses.npy")
kf_idx = np.load(res / "kf_idx.npy")
plys = sorted(glob.glob(str(res / "pcd_*.ply")), key=lambda p: int(Path(p).stem.split("_")[1]))
kf_pngs = sorted(glob.glob(str(res / "kf_*.png")))
print("\n=== OUTPUT ===", res)
print(f"traj.npy      : {traj.shape}  one 4x4 camera-to-world pose per input frame")
print(f"kf_poses.npy  : {kf_poses.shape}  poses of the keyframes held in the KV cache")
print(f"kf_idx.npy    : {kf_idx.shape}  frame indices of those keyframes: {kf_idx.tolist()}")
print(f"pcd_*.ply     : {len(plys)} files (one per accepted keyframe, cumulative scene)")
pcd = None
if plys:
    pcd = o3d.io.read_point_cloud(plys[-1])
    pts = np.asarray(pcd.points)
    print(f"  last: {Path(plys[-1]).name}  {len(pts)} points, "
          f"bbox {np.round(pts.max(0)-pts.min(0), 2).tolist()} (tracker units)")
print(f"kf_*.png      : {len(kf_pngs)} keyframe images")

# sanity
R = traj[:, :3, :3]
orth = np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max()
dets = np.linalg.det(R)
print("\n=== SANITY ===")
print(f"rotations orthonormal (max |R R^T - I|): {orth:.2e}, det in [{dets.min():.4f},{dets.max():.4f}]")
print(f"first pose is identity: {np.allclose(traj[0], np.eye(4), atol=1e-5)}")
step = np.linalg.norm(np.diff(traj[:, :3, 3], axis=0), axis=1)
print(f"per-frame translation step: median {step.median() if hasattr(step,'median') else np.median(step):.4f}, max {step.max():.4f} (tracker units)")

# ---------------- ALIGN est vs GT (same recipe as score_tum.py) ----------------
gt = TUMLoader.load_gt(str(scene))
n = min(len(traj), len(gt))
ts = np.arange(n) * 0.1
t_ref = PoseTrajectory3D(poses_se3=gt[:n], timestamps=ts)
t_est = PoseTrajectory3D(poses_se3=traj[:n], timestamps=ts)
Ra, ta, s = t_est.align(t_ref, correct_scale=True)
est_xyz = t_est.positions_xyz
gt_xyz = t_ref.positions_xyz
err = np.linalg.norm(est_xyz - gt_xyz, axis=1)
print(f"\nSim(3) to GT: scale {s:.3f}  ->  ATE RMSE {np.sqrt((err**2).mean())*100:.2f} cm, "
      f"max {err.max()*100:.1f} cm")

# ---------------- FIGURE ----------------
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.6, 1])

# row 0: input frames
for k, i in enumerate(np.linspace(0, len(rgb_paths) - 1, 4).astype(int)):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(cv2.imread(rgb_paths[i])[:, :, ::-1]); ax.axis("off")
    ax.set_title(f"INPUT frame {i}/{len(rgb_paths)}", fontsize=9)

# row 1: trajectory 3D + top-down, point cloud
ax = fig.add_subplot(gs[1, 0:2], projection="3d")
ax.plot(*gt_xyz.T, "k-", lw=1, label="ground truth (mocap)")
ax.plot(*est_xyz.T, "r-", lw=1, label=f"KV-Tracker, Sim(3) aligned")
ax.scatter(*est_xyz[0], c="g", s=30, label="start")
ax.set_title(f"OUTPUT traj.npy: {n} camera poses  |  ATE {np.sqrt((err**2).mean())*100:.2f} cm", fontsize=10)
ax.legend(fontsize=8); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")

ax = fig.add_subplot(gs[1, 2])
ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], "k-", lw=1); ax.plot(est_xyz[:, 0], est_xyz[:, 1], "r-", lw=1)
ax.set_aspect("equal"); ax.set_title("top-down (x,y) [m]", fontsize=9); ax.grid(alpha=.3)

ax = fig.add_subplot(gs[1, 3])
ax.plot(np.arange(n) / 30.0, err * 100)
ax.set_xlabel("time [s]"); ax.set_ylabel("position error [cm]"); ax.set_title("per-frame error after alignment", fontsize=9)
ax.grid(alpha=.3)

# row 2: point cloud (2 views) + keyframes
if pcd is not None:
    pts = np.asarray(pcd.points); cols = np.asarray(pcd.colors)
    sel = np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)
    p, c = pts[sel], cols[sel]
    # bring the cloud into the GT frame with the same Sim(3) so it sits with the trajectory
    p_al = s * (p @ Ra.T) + ta
    kf_al = s * (kf_poses[:, :3, 3] @ Ra.T) + ta
    ax = fig.add_subplot(gs[2, 0], projection="3d")
    ax.scatter(*p_al.T, c=c, s=0.2); ax.scatter(*kf_al.T, c="r", s=12, marker="^")
    ax.set_title(f"OUTPUT {Path(plys[-1]).name}: {len(pts)} pts + {len(kf_poses)} keyframes", fontsize=9)
    ax.set_axis_off()
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(p_al[:, 0], p_al[:, 2], c=c, s=0.2); ax.scatter(kf_al[:, 0], kf_al[:, 2], c="r", s=12, marker="^")
    ax.set_aspect("equal"); ax.set_title("point cloud, x-z projection", fontsize=9); ax.axis("off")
if kf_pngs:
    for k, i in enumerate(np.linspace(0, len(kf_pngs) - 1, min(2, len(kf_pngs))).astype(int)):
        ax = fig.add_subplot(gs[2, 2 + k])
        ax.imshow(cv2.imread(kf_pngs[i])[:, :, ::-1]); ax.axis("off")
        ax.set_title(f"OUTPUT {Path(kf_pngs[i]).name} (frame {kf_idx[i] if i < len(kf_idx) else '?'})", fontsize=9)

fig.suptitle(f"KV-Tracker on {scene.name}: {len(rgb_paths)} RGB frames in -> poses + point cloud out", fontsize=12)
fig.tight_layout()
out = res / "io_summary.png"
fig.savefig(out, dpi=110)
print(f"\nfigure -> {out}")
