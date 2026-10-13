#!/usr/bin/env python3
"""KV-Tracker's raw outputs -> the same cloud layout run_pi3.py writes, plus pose scores.

    ~/.conda/envs/kvtracker/bin/python scripts/collect_kv.py

seva_to_shaper version of ~/stable_virtual_camera/exp5/scripts/collect.py. For each
object, outputs/kv/<obj>/pcd_<k>.ply (the newest cumulative cloud, open3d binary) is read,
the backdrop rule from run_pi3.py is applied to the colours, and
outputs/kv/<obj>/points_object.ply + kv.npz + stats.json are written. traj.npy (one c2w
per frame) and Pi3's batch poses are scored against the cameras SEVA was asked for
(inputs/<obj>/transforms.json test_ids, OpenGL -> OpenCV) with evo's Sim(3) alignment.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, open3d as o3d
sys.path.insert(0, "/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
from kv_tracker.eval_tools.evo_utils import align_pair  # noqa: E402

E = Path("/home/isha.tariq/seva_to_shaper")
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
SUBJ_SAT, SUBJ_VAL = 0.15, 0.70


def subject_mask(rgb):
    mx, mn = rgb.max(1), rgb.min(1); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > SUBJ_SAT) | (mx < SUBJ_VAL)


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32); rgb = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T; rec["r"], rec["g"], rec["b"] = rgb.T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def target_cams(obj):
    t = json.loads((E / f"inputs/{obj}/transforms.json").read_text())
    split = next((E / f"inputs/{obj}").glob("train_test_split_*.json"))
    ids = json.loads(split.read_text())["test_ids"]
    return np.stack([np.asarray(t["frames"][i]["transform_matrix"], np.float64) @ GL2CV for i in ids])


def pose_score(est, gt):
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    ate, rpe_t, rpe_r = align_pair({"traj_gt": gt, "traj_est": est}, ret_np=False)
    return {"frames": int(n), "ate_m": float(ate), "rpe_trans_m": float(rpe_t), "rpe_rot_deg": float(np.degrees(rpe_r))}


poses, runs = [], {}
for obj in json.loads((E / "outputs/sets.json").read_text()):
    gt = target_cams(obj); radius = float(np.linalg.norm(gt[:, :3, 3] - gt[:, :3, 3].mean(0), axis=1).mean())
    p3 = E / f"outputs/pi3/{obj}/pi3.npz"
    if p3.exists():
        r = pose_score(np.load(p3)["camera_poses"].astype(np.float64), gt); r.update(object=obj, reconstructor="pi3", orbit_radius_m=radius); poses.append(r)
    d = E / f"outputs/kv/{obj}"
    plys = sorted(glob.glob(str(d / "pcd_*.ply")), key=lambda p: int(Path(p).stem.split("_")[1]))
    if not plys or not (d / "traj.npy").exists():
        print(f"!! {obj}: no KV-Tracker output"); continue
    traj = np.load(d / "traj.npy").astype(np.float64); kf_idx = np.load(d / "kf_idx.npy")
    r = pose_score(traj, gt); r.update(object=obj, reconstructor="kv", orbit_radius_m=radius, keyframes=int(len(kf_idx))); poses.append(r)
    pc = o3d.io.read_point_cloud(plys[-1]); pts = np.asarray(pc.points, np.float32); col = np.asarray(pc.colors, np.float32)
    subj = subject_mask(col)
    write_ply(d / "points_all.ply", pts, col); write_ply(d / "points_object.ply", pts[subj], col[subj])
    np.savez_compressed(d / "kv.npz", traj=traj, kf_idx=kf_idx, object_points=pts[subj], object_colors=col[subj])
    st = {"reconstructor": "kv_tracker", "frames": int(len(traj)), "keyframes": int(len(kf_idx)), "kf_auto": 1, "resize_dim": 518,
          "points_all": int(len(pts)), "points_object": int(subj.sum()), "subject_fraction": round(float(subj.mean()), 4), "source_ply": Path(plys[-1]).name}
    (d / "stats.json").write_text(json.dumps(st, indent=2)); runs[obj] = st
    print(f"{obj:26s} {len(traj)} poses, {len(kf_idx)} kf, {len(pts):8d} pts -> {subj.sum():8d} object   ATE {r['ate_m']*100:5.2f} cm ({r['ate_m']/radius*100:4.1f}% of radius)")
(E / "outputs/poses.json").write_text(json.dumps(poses, indent=2)); (E / "outputs/kv/runs.json").write_text(json.dumps(runs, indent=2))
for rec in ("pi3", "kv"):
    rows = [r for r in poses if r["reconstructor"] == rec]
    if rows: print(f"  {rec:4s} ATE {np.mean([r['ate_m'] for r in rows])*100:5.2f} cm = {np.mean([r['ate_m']/r['orbit_radius_m'] for r in rows])*100:4.1f}% of radius")
