#!/usr/bin/env python3
"""Experiment 4's SLAM-point comparison for experiment 10's clouds, with the current
(upright, camera-constrained) fitter -- the same `fit_any` the ViewCrafter-workspace outputs were
last scored with -- written to experiment10/outputs/, nothing touched elsewhere.
Then puts each corrected wooden-table number beside the one it replaces, and the
five-object condition means before and after the substitution.
"""
import json, pickle, sys, zlib
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
E10 = PROJ / "experiment10"
sys.path.insert(0, str(PROJ / "experiment3/scripts")); sys.path.insert(0, str(PROJ / "experiment4/scripts"))
from align_and_score import apply, fit_any, normalise, load_ply   # noqa: E402
from realign_to_slam import stats, MAX_VGGT                        # noqa: E402
sys.path.insert(0, str(PROJ / "shared/scripts")); import objects as C   # noqa: E402
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
OBJS = C.names()

sets = sorted(p for p in (E10 / "vggt").iterdir() if (p / "points_object.ply").exists())
rows, cache = [], {}
for s in sets:
    obj, cond = s.name.rsplit("__", 1)
    if obj not in cache:
        d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64), faces=np.asarray(d["mesh_faces"]), process=False)
        np.random.seed(0)
        cache[obj] = (slam, cKDTree(slam), cKDTree(np.asarray(mesh.sample(80000), np.float64)),
                      float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])))
    slam, tree_slam, tree_surf, diag = cache[obj]
    vggt = load_ply(s / "points_object.ply")
    if len(vggt) > MAX_VGGT:
        sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(vggt)))
        vggt = vggt[sub.choice(len(vggt), MAX_VGGT, replace=False)]
    M, cost, up_z, how = fit_any(vggt, slam, s)
    slam_n, slam_c, slam_s = normalise(slam)
    aligned = apply(M, normalise(vggt)[0]) * slam_s + slam_c
    row = {"object": obj, "condition": cond, "slam_to_vggt": stats(slam, cKDTree(aligned), diag),
           "vggt_to_slam": stats(aligned, tree_slam, diag), "vggt_to_mesh": stats(aligned, tree_surf, diag),
           "fitted_scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4),
           "sym_cost": round(float(cost), 5), "fit": how, "up_z": None if up_z is None else round(up_z, 3)}
    rows.append(row)
    np.save(E10 / "outputs" / f"{s.name}__aligned_to_slam.npy", aligned.astype(np.float32))
    print(f"{s.name:36s} found {row['slam_to_vggt']['within_2pct']:.3f}  cloud->SLAM {row['vggt_to_slam']['mean_x100']:.2f}%  scale {row['fitted_scale']:.2f}  {how} up_z {up_z:+.2f}", flush=True)
(E10 / "outputs" / "realigned_to_slam.json").write_text(json.dumps(rows, indent=2))

# ---- beside the numbers they replace (experiment 4's current, upright-fit table)
cur = {(r["object"], r["condition"]): r for r in json.loads((PROJ / "experiment4/outputs/realigned_to_slam.json").read_text())}
new = {(r["object"], r["condition"]): r for r in rows}
mesh_new = {(r["object"], r["condition"]): r for r in json.loads((E10 / "vggt/scores.json").read_text())}
mesh_cur = {(r["object"], r["condition"]): r for r in json.loads((PROJ / "experiment3/outputs/scores.json").read_text())}
out = []
print(f"\n{'condition':10s} {'table found old->new':>22s} {'cloud->SLAM% old->new':>22s} {'mesh recall old->new':>22s} | {'5-obj mean old->new':>20s}")
for (obj, cond) in sorted(new):
    o, n = cur[(obj, cond)], new[(obj, cond)]
    f_old = [cur[(x, cond)]["slam_to_vggt"]["within_2pct"] for x in OBJS]
    f_new = [n["slam_to_vggt"]["within_2pct"] if x == obj else cur[(x, cond)]["slam_to_vggt"]["within_2pct"] for x in OBJS]
    rec = {"object": obj, "condition": cond,
           "slam_found": [o["slam_to_vggt"]["within_2pct"], n["slam_to_vggt"]["within_2pct"]],
           "cloud_to_slam_x100": [o["vggt_to_slam"]["mean_x100"], n["vggt_to_slam"]["mean_x100"]],
           "mesh_recall": [mesh_cur[(obj, cond)]["recall_at_2pct"], mesh_new[(obj, cond)]["recall_at_2pct"]],
           "mean5_slam_found": [round(float(np.mean(f_old)), 3), round(float(np.mean(f_new)), 3)]}
    out.append(rec)
    print(f"{cond:10s} {rec['slam_found'][0]:10.3f} -> {rec['slam_found'][1]:.3f} {rec['cloud_to_slam_x100'][0]:10.2f} -> {rec['cloud_to_slam_x100'][1]:.2f} "
          f"{rec['mesh_recall'][0]:10.3f} -> {rec['mesh_recall'][1]:.3f} | {rec['mean5_slam_found'][0]:8.3f} -> {rec['mean5_slam_found'][1]:.3f}")
(E10 / "outputs" / "before_after.json").write_text(json.dumps(out, indent=2))
