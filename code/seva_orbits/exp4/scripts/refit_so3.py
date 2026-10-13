#!/usr/bin/env python3
"""Re-fit every cloud the exp4 deck shows to the SLAM points with a full SO(3) search.

    ~/.conda/envs/shaper/bin/python scripts/refit_so3.py [--only cond ...]

Why: experiment 3's `fit` seeds ICP from Rz(az)*Rx(tilt), tilt in {0, 60, 120} deg --
two of the three rotation angles. Nothing in that seed set lies within 60 deg of an
upside-down pose, and ICP then settles in the flipped basin. That is what the
"inverted" clouds in the first deck were: not a reconstruction error but an
alignment that had been allowed to land mirrored. exp5/scripts/fit_so3.py replaces
the seeds with a grid over all of SO(3) (az x tilt x roll at 30 deg, 744 starts, best
48 to ICP), same cost, same ICP, same scale grid. Applied here to the seven
conditions on the deck so every number is on one fit.

Outputs, in the layout scripts/render_slam_reference.py reads:
  outputs/so3/<obj>__<cond>__aligned_to_slam.npy   the fitted cloud (N,3)
  outputs/so3/<obj>__<cond>__slam_to_vggt.npy      distance from each SLAM point to it
  outputs/so3/realigned_so3.json                   rows in realign_to_slam.json's schema,
                                                   plus up_z (+1 upright, -1 flipped) and `old`
"""
import argparse, json, pickle, sys, zlib
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp5/scripts"); import fit_so3
A = fit_so3.A; apply, normalise, load_ply = A.apply, A.normalise, A.load_ply
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data"); EXP3 = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/outputs")
EXP4 = Path("/home/isha.tariq/viewcrafter_orbits/experiment4/outputs"); OUT = Path("/home/isha.tariq/stable_virtual_camera/exp4/outputs/so3")
OBJ = [r["object"] for r in json.loads(Path("/home/isha.tariq/stable_virtual_camera/exp1/inputs/manifest.json").read_text())]
CONDS = ["real", "gen360", "v3_360", "seva360", "seva360f", "seva360p", "seva360fp"]
TAU, MAX_PTS = 0.02, 120000

def stats(src, tree, diag):
    d, _ = tree.query(src); return {"mean_x100": round(float(d.mean()) / diag * 100, 4), "within_2pct": round(float((d < TAU * diag).mean()), 4)}

def load_set(obj, cond):
    d = EXP3 / f"{obj}__{cond}"; pts = load_ply(d / "points_object.ply")
    if (d / "pi3.npz").exists():
        z = np.load(d / "pi3.npz"); return pts, z["object_colors"], z["camera_poses"][:, :3, 3]
    z = np.load(d / "vggt.npz"); E = z["extrinsic"]
    cams = np.stack([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(E))]); return pts, z["object_colors"], cams

ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); args = ap.parse_args()
conds = args.only or CONDS
prev = {(r["object"], r["condition"]): r for r in json.loads((EXP4 / "realigned_to_slam.json").read_text())}
done = {}
if (OUT / "realigned_so3.json").exists():
    done = {(r["object"], r["condition"]): r for r in json.loads((OUT / "realigned_so3.json").read_text())}
for obj in OBJ:
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb")); slam = np.asarray(d["points_model"].numpy(), np.float64)
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64), faces=np.asarray(d["mesh_faces"]), process=False)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])); tree_slam = cKDTree(slam)
    print(f"\n=== {obj}  ({len(slam)} SLAM points, diag {diag:.2f} m) ===", flush=True)
    for cond in conds:
        if not (EXP3 / f"{obj}__{cond}" / "points_object.ply").exists(): print(f"  {cond}: missing"); continue
        pts, col, cams = load_set(obj, cond)
        ctr = cams.mean(0); n = np.linalg.svd(cams - ctr)[2][-1]; n *= np.sign((ctr - pts.mean(0)) @ n)
        if len(pts) > MAX_PTS:
            sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(pts))).choice(len(pts), MAX_PTS, replace=False); pts = pts[sub]
        M, cost = fit_so3.fit(pts, slam)
        if M is None: print(f"  {cond}: alignment failed"); continue
        R = M[:3, :3] / np.cbrt(abs(np.linalg.det(M[:3, :3]))); up_z = float((R @ n)[2])
        slam_n, slam_c, slam_s = normalise(slam); aligned = apply(M, normalise(pts)[0]) * slam_s + slam_c
        dist, _ = cKDTree(aligned).query(slam)
        row = {"object": obj, "condition": cond, "points": int(len(pts)), "up_z": round(up_z, 3),
               "slam_to_vggt": stats(slam, cKDTree(aligned), diag), "vggt_to_slam": stats(aligned, tree_slam, diag),
               "fitted_scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4), "sym_cost": round(float(cost), 5)}
        p = prev.get((obj, cond))
        if p: row["old"] = {"slam_found": p["slam_to_vggt"]["within_2pct"], "cloud_to_slam": p["vggt_to_slam"]["mean_x100"], "sym_cost": p["sym_cost"], "fitted_scale": p["fitted_scale"]}
        done[(obj, cond)] = row
        np.save(OUT / f"{obj}__{cond}__aligned_to_slam.npy", aligned.astype(np.float32)); np.save(OUT / f"{obj}__{cond}__slam_to_vggt.npy", dist.astype(np.float32))
        (OUT / "realigned_so3.json").write_text(json.dumps(list(done.values()), indent=2))
        old = f"  (old fit: found {p['slam_to_vggt']['within_2pct']:.3f}, cost {p['sym_cost']:.4f})" if p else ""
        print(f"  {cond:10s} found {row['slam_to_vggt']['within_2pct']:.3f}  cloud->SLAM {row['vggt_to_slam']['mean_x100']:.2f}%  cost {cost:.4f}  scale {row['fitted_scale']:.2f}  up->z {up_z:+.2f}{old}", flush=True)
