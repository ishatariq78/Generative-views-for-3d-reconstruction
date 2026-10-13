#!/usr/bin/env python3
"""Clouds against ShapeR's SLAM points, with the current (upright) fitter.

    ~/.conda/envs/shaper/bin/python scripts/score_slam.py

Same loop, subsample seeding and TAU as experiment4/scripts/realign_to_slam.py, and the same
fitter it now uses -- `align_and_score.fit_any`: the reconstructor's own cameras (kv.npz /
pi3.npz / vggt.npz) give the up direction, the search is azimuth x scale from an upright
pre-rotation with a tilt guard (the free two-axis and full-SO(3) searches both left clouds
upside down; see experiment9/scripts/check_upright.py). Numbers here therefore equal the
current layer in experiment4/outputs/realigned_to_slam.json for the shared conditions.
Writes outputs/slam_scores.json and <obj>__<cond>__aligned_to_slam.npz (points, colors).
"""
import json, pickle, sys, zlib
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment3/scripts")
from align_and_score import apply, fit_any, normalise, load_ply  # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data"); EXP3 = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/outputs")
E5 = Path("/home/isha.tariq/stable_virtual_camera/exp5")
OBJ = [r["object"] for r in json.loads((E5.parent / "exp1/inputs/manifest.json").read_text())]
CONDS = [("seva360fk1", "KV-Tracker, kf every frame"), ("seva360fk5", "KV-Tracker, kf every 5"), ("seva360fk10", "KV-Tracker, kf every 10"),
         ("seva360fp", "Pi3 batch, 80 frames"), ("seva360p", "Pi3 batch, 25 frames"), ("seva360f", "VGGT batch, 80 frames")]
TAU, MAX_PTS = 0.02, 120000

def stats(src, tree, diag):
    d, _ = tree.query(src); return {"mean_x100": round(float(d.mean()) / diag * 100, 4), "within_2pct": round(float((d < TAU * diag).mean()), 4)}

def colors_of(d):
    for f, k in (("kv.npz", "object_colors"), ("pi3.npz", "object_colors"), ("vggt.npz", "object_colors")):
        if (d / f).exists(): return np.load(d / f)[k]
    return None

rows = []
for obj in OBJ:
    dd = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb")); slam = np.asarray(dd["points_model"].numpy(), np.float64)
    mesh = trimesh.Trimesh(vertices=np.asarray(dd["mesh_vertices"], np.float64), faces=np.asarray(dd["mesh_faces"]), process=False)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])); tree_slam = cKDTree(slam)
    np.random.seed(0); tree_surf = cKDTree(np.asarray(mesh.sample(80000), np.float64))
    print(f"\n=== {obj}  ({len(slam)} SLAM points, diag {diag:.2f} m) ===", flush=True)
    for cond, label in CONDS:
        d = EXP3 / f"{obj}__{cond}"
        if not (d / "points_object.ply").exists(): print(f"  {cond}: missing"); continue
        pts = load_ply(d / "points_object.ply"); col = colors_of(d)
        if len(pts) > MAX_PTS:
            sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(pts))).choice(len(pts), MAX_PTS, replace=False); pts = pts[sub]; col = None if col is None else col[sub]
        M, cost, up_z, how = fit_any(pts, slam, d)
        if M is None: print(f"  {cond}: alignment failed"); continue
        slam_n, slam_c, slam_s = normalise(slam); aligned = apply(M, normalise(pts)[0]) * slam_s + slam_c
        row = {"object": obj, "condition": cond, "label": label, "points": int(len(pts)), "fit": how, "up_z": None if up_z is None else round(float(up_z), 3),
               "slam_to_vggt": stats(slam, cKDTree(aligned), diag), "vggt_to_slam": stats(aligned, tree_slam, diag), "vggt_to_mesh": stats(aligned, tree_surf, diag),
               "fitted_scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4), "sym_cost": round(float(cost), 5)}
        rows.append(row); np.savez_compressed(E5 / f"outputs/{obj}__{cond}__aligned_to_slam.npz", points=aligned.astype(np.float32), colors=(col if col is not None else np.zeros((0, 3))).astype(np.float32))
        print(f"  {cond:12s} found {row['slam_to_vggt']['within_2pct']:.3f}  cloud->SLAM {row['vggt_to_slam']['mean_x100']:.2f}% (within 2%: {row['vggt_to_slam']['within_2pct']:.3f})  cost {cost:.4f}  scale {row['fitted_scale']:.2f}  up->z {row['up_z']}  [{how}]", flush=True)
(E5 / "outputs/slam_scores.json").write_text(json.dumps(rows, indent=2))
print("\nmeans over objects (upright fit):")
for cond, label in CONDS:
    rs = [r for r in rows if r["condition"] == cond]
    if rs: print(f"  {label:28s} SLAM found {np.mean([r['slam_to_vggt']['within_2pct'] for r in rs]):.3f}   SLAM->cloud {np.mean([r['slam_to_vggt']['mean_x100'] for r in rs]):.2f}%   cloud->SLAM {np.mean([r['vggt_to_slam']['mean_x100'] for r in rs]):.2f}%   within2% {np.mean([r['vggt_to_slam']['within_2pct'] for r in rs]):.3f}")
