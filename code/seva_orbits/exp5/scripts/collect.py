#!/usr/bin/env python3
"""Turn KV-Tracker's raw outputs into (a) ViewCrafter-workspace-format clouds and (b) pose scores.

    ~/.conda/envs/kvtracker/bin/python scripts/collect.py

(a) The tracker's last pcd_<k>.ply holds every keyframe's confident points in its own
    world frame. The backdrop rule from experiment 2 / run_pi3.py (saturation > 0.15 or
    value < 0.70 = object) is applied to the point colours, and the result is written as
    experiment3/outputs/<obj>__seva360fk<N>/points_object.ply (+ points_all.ply, stats.json)
    so it is scored on identical terms to the VGGT and Pi3 clouds of exp3/exp4.
(b) traj.npy (one c2w per frame) is scored against the cameras SEVA was *asked* to render
    (exp1 transforms.json, OpenGL -> OpenCV), Sim(3)-aligned with evo exactly as the TUM
    run was. Pi3's batch poses (exp4, pi3.npz) get the same treatment for comparison.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, open3d as o3d
from evo.core.trajectory import PoseTrajectory3D
sys.path.insert(0, "/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
from kv_tracker.eval_tools.evo_utils import align_pair  # noqa: E402

# Same backdrop rule and PLY layout as exp4/scripts/run_pi3.py (copied: importing it loads Pi3).
SUBJ_SAT, SUBJ_VAL = 0.15, 0.70
def subject_mask(rgb):                       # (N,3) in [0,1] -> True where object rather than backdrop
    mx, mn = rgb.max(1), rgb.min(1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > SUBJ_SAT) | (mx < SUBJ_VAL)
def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32); rgb = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T; rec["r"], rec["g"], rec["b"] = rgb.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()); f.write(rec.tobytes())

E1 = Path("/home/isha.tariq/stable_virtual_camera/exp1"); E5 = Path("/home/isha.tariq/stable_virtual_camera/exp5")
EXP3 = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/outputs")
OBJ = [r["object"] for r in json.loads((E1 / "inputs/manifest.json").read_text())]
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

def target_cams(obj):
    t = json.loads((E1 / f"inputs/{obj}/transforms.json").read_text()); ids = json.loads((E1 / f"inputs/{obj}/train_test_split_8.json").read_text())["test_ids"]
    return np.stack([np.asarray(t["frames"][i]["transform_matrix"], np.float64) @ GL2CV for i in ids])

def pose_score(est, gt):
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    ate, rpe_t, rpe_r = align_pair({"traj_gt": gt, "traj_est": est}, ret_np=False)
    ts = np.arange(n) * 0.1; te = PoseTrajectory3D(poses_se3=est, timestamps=ts); te.align(PoseTrajectory3D(poses_se3=gt, timestamps=ts), correct_scale=True)
    return {"frames": int(n), "ate_m": float(ate), "rpe_trans_m": float(rpe_t), "rpe_rot_deg": float(np.degrees(rpe_r))}, te.positions_xyz

poses, stats_all = [], {}
for obj in OBJ:
    gt = target_cams(obj); radius = float(np.linalg.norm(gt[:, :3, 3] - gt[:, :3, 3].mean(0), axis=1).mean())
    aligned = {"target": gt[:, :3, 3]}
    for cond, src in [("pi3_batch", EXP3 / f"{obj}__seva360fp/pi3.npz")]:
        if src.exists():
            r, xyz = pose_score(np.load(src)["camera_poses"].astype(np.float64), gt); r.update(object=obj, condition=cond, orbit_radius_m=radius); poses.append(r); aligned[cond] = xyz
    for kf in (1, 5, 10):
        d = E5 / f"outputs/{obj}__kf{kf}"; cond = f"seva360fk{kf}"
        plys = sorted(glob.glob(str(d / "pcd_*.ply")), key=lambda p: int(Path(p).stem.split("_")[1]))
        if not plys or not (d / "traj.npy").exists():
            print(f"!! {obj} kf{kf}: no output"); continue
        traj = np.load(d / "traj.npy").astype(np.float64); kf_idx = np.load(d / "kf_idx.npy"); kf_poses = np.load(d / "kf_poses.npy")
        r, xyz = pose_score(traj, gt); r.update(object=obj, condition=cond, orbit_radius_m=radius, keyframes=int(len(kf_idx))); poses.append(r); aligned[cond] = xyz
        pc = o3d.io.read_point_cloud(plys[-1]); pts = np.asarray(pc.points, np.float32); col = np.asarray(pc.colors, np.float32)
        subj = subject_mask(col)
        out = EXP3 / f"{obj}__{cond}"; out.mkdir(exist_ok=True)
        write_ply(out / "points_all.ply", pts, col); write_ply(out / "points_object.ply", pts[subj], col[subj])
        np.savez_compressed(out / "kv.npz", traj=traj, kf_idx=kf_idx, kf_poses=kf_poses, object_points=pts[subj], object_colors=col[subj], all_points=pts, all_colors=col)
        st = {"reconstructor": "kv_tracker", "frames": int(len(traj)), "keyframes": int(len(kf_idx)), "kf_auto": kf, "resize_dim": 518,
              "points_all": int(len(pts)), "points_object": int(subj.sum()), "subject_fraction": round(float(subj.mean()), 4), "source_ply": Path(plys[-1]).name}
        (out / "stats.json").write_text(json.dumps(st, indent=2)); stats_all[f"{obj}__{cond}"] = st
        print(f"{obj:22s} kf{kf:<3d} {len(traj)} poses, {len(kf_idx)} kf, {len(pts):7d} pts -> {subj.sum():7d} object   ATE {r['ate_m']*100:5.2f} cm ({r['ate_m']/radius*100:4.1f}% of radius)  RPE {r['rpe_trans_m']*100:.2f} cm / {r['rpe_rot_deg']:.2f} deg")
    np.savez_compressed(E5 / f"outputs/{obj}__poses_aligned.npz", **aligned)
(E5 / "outputs/poses.json").write_text(json.dumps(poses, indent=2)); (E5 / "outputs/runs.json").write_text(json.dumps(stats_all, indent=2))
print("\npose summary (ATE cm, mean over objects):")
for cond in ("seva360fk1", "seva360fk5", "seva360fk10", "pi3_batch"):
    rows = [r for r in poses if r["condition"] == cond]
    if rows: print(f"  {cond:12s} ATE {np.mean([r['ate_m'] for r in rows])*100:5.2f} cm  = {np.mean([r['ate_m']/r['orbit_radius_m'] for r in rows])*100:4.1f}% of radius   RPE {np.mean([r['rpe_trans_m'] for r in rows])*100:.2f} cm / {np.mean([r['rpe_rot_deg'] for r in rows]):.2f} deg")
