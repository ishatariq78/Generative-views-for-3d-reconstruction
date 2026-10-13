#!/usr/bin/env python3
"""Place each reconstructor's cloud in the SLAM frame by its cameras, not by ICP.

    ~/.conda/envs/shaper/bin/python scripts/fit_cameras.py

The orbit's 80 target cameras were *specified* (exp1/inputs/<obj>/transforms.json, in the
object's mesh frame = the SLAM frame), and both VGGT and Pi3 estimate a pose for every
one of those frames. So the similarity that maps a reconstructor's camera centres onto
the true camera centres (Umeyama, with scale) places its cloud in the SLAM frame with
no point-cloud ICP at all -- hence no upside-down basin, no slab-at-the-centroid, no
box-from-above ambiguity. Reconstruction error in the cameras shows up as
misplacement, which is fair: it is the reconstructor's own error.

The camera-derived rotation is then the *single seed* for the shared fitter
(experiment3's `fit`: symmetric Chamfer, scale grid, ICP) -- orientation from the
cameras, refinement from the fitter. With one seed the fitter cannot wander to a
flipped basin. Both stages are scored, so the refinement's effect is visible.

Outputs, in the layout scripts/render_slam_reference.py reads:
  outputs/cam/<obj>__<cond>__aligned_to_slam.npy, __slam_to_vggt.npy, realigned_cam.json
"""
import json, pickle, sys, zlib
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment3/scripts")
import align_and_score as A

E1 = Path("/home/isha.tariq/stable_virtual_camera/exp1"); EXP3 = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/outputs")
OUT = Path("/home/isha.tariq/stable_virtual_camera/exp4/outputs/cam"); OUT.mkdir(parents=True, exist_ok=True)
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
OBJ = [r["object"] for r in json.loads((E1 / "inputs/manifest.json").read_text())]
CONDS = ["seva360", "seva360f", "seva360p", "seva360fp"]
TAU, MAX_PTS, N25 = 0.02, 120000, 25
OLD = {(r["object"], r["condition"]): r for r in json.loads(Path("/home/isha.tariq/stable_virtual_camera/exp4/outputs/so3/realigned_so3.json").read_text())}


def load_ply(p):
    with open(p, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"): n = int(line.split()[-1])
            if line.strip() == b"end_header": break
        rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)


def umeyama(src, dst):
    """Sim(3) s,R,t with dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src); U, sig, Vt = np.linalg.svd(cov)
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(d) @ Vt; s = (sig * d).sum() / (S ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def est_cameras(obj, cond):
    d = EXP3 / f"{obj}__{cond}"
    if (d / "pi3.npz").exists():
        return np.load(d / "pi3.npz")["camera_poses"][:, :3, 3].astype(np.float64)
    E = np.load(d / "vggt.npz")["extrinsic"]
    return np.stack([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(E))]).astype(np.float64)


def true_cameras(obj, n):
    tf = json.loads((E1 / "inputs" / obj / "transforms.json").read_text())["frames"]
    n_in = json.loads((E1 / "inputs" / obj / "meta.json").read_text())["views"]
    C = np.array([f["transform_matrix"] for f in tf[n_in:]])[:, :3, 3]        # centres: convention-free
    if n == N25:                                                            # exp3's uniform pick
        C = C[[round(i * (len(C) - 1) / (N25 - 1)) for i in range(N25)]]
    return C


def stats(src, tree, diag):
    d, _ = tree.query(src); return {"mean_x100": round(float(d.mean()) / diag * 100, 4), "within_2pct": round(float((d < TAU * diag).mean()), 4)}


def refine(pts, slam, R_seed):
    """The shared fitter seeded with the camera rotation only. Rotation is unchanged
    by the fitter's centring/scaling normalisation, so the seed carries over."""
    R4 = np.eye(4); R4[:3, :3] = R_seed
    old_rot, old_mid = A.initial_rotations, A.N_MID
    A.initial_rotations, A.N_MID = (lambda *a, **k: [R4]), 1
    try:
        M, cost = A.fit(pts, slam)
    finally:
        A.initial_rotations, A.N_MID = old_rot, old_mid
    slam_n, slam_c, slam_s = A.normalise(slam)
    return A.apply(M, A.normalise(pts)[0]) * slam_s + slam_c, float(cost), float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))


rows = []
for obj in OBJ:
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb")); slam = np.asarray(d["points_model"].numpy(), np.float64)
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64), faces=np.asarray(d["mesh_faces"]), process=False)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])); tree = cKDTree(slam)
    print(f"\n=== {obj} ===", flush=True)
    for cond in CONDS:
        pts = load_ply(EXP3 / f"{obj}__{cond}" / "points_object.ply")
        if len(pts) > MAX_PTS:
            pts = pts[np.random.default_rng(zlib.crc32(np.ascontiguousarray(pts.astype(np.float32)))).choice(len(pts), MAX_PTS, replace=False)]
        est, true = est_cameras(obj, cond), true_cameras(obj, len(est_cameras(obj, cond)))
        s, R, t = umeyama(est, true)
        cam_rms = float(np.sqrt((((s * (R @ est.T)).T + t - true) ** 2).sum(1).mean())) / diag * 100
        placed = (s * (R @ pts.T)).T + t
        refined, cost, scale = refine(pts, slam, R)
        shift = float(np.linalg.norm(refined.mean(0) - placed.mean(0))) / diag * 100
        row = {"object": obj, "condition": cond, "points": int(len(pts)), "camera_rms_x100": round(cam_rms, 2), "camera_scale": round(float(s), 4),
               "cam_only": {"slam_to_vggt": stats(slam, cKDTree(placed), diag), "vggt_to_slam": stats(placed, tree, diag)},
               "slam_to_vggt": stats(slam, cKDTree(refined), diag), "vggt_to_slam": stats(refined, tree, diag),
               "refine_shift_x100": round(shift, 2), "sym_cost": round(cost, 5), "fitted_scale": round(scale, 4), "up_z": 1.0,
               "so3_cost": OLD[(obj, cond)]["sym_cost"] if (obj, cond) in OLD else None,
               "old": {"slam_found": OLD[(obj, cond)]["slam_to_vggt"]["within_2pct"], "cloud_to_slam": OLD[(obj, cond)]["vggt_to_slam"]["mean_x100"]} if (obj, cond) in OLD else None}
        rows.append(row)
        np.save(OUT / f"{obj}__{cond}__aligned_to_slam.npy", refined.astype(np.float32)); np.save(OUT / f"{obj}__{cond}__slam_to_vggt.npy", cKDTree(refined).query(slam)[0].astype(np.float32))
        print(f"  {cond:10s} cameras rms {cam_rms:5.2f}%  cam-only found {row['cam_only']['slam_to_vggt']['within_2pct']:.3f} -> refined {row['slam_to_vggt']['within_2pct']:.3f} "
              f"(SO3 fit gave {row['old']['slam_found']:.3f})   cloud->SLAM {row['vggt_to_slam']['mean_x100']:.2f}%  cost {cost:.4f} vs SO3 {row['so3_cost']:.4f}  moved {shift:.2f}%", flush=True)
(OUT / "realigned_cam.json").write_text(json.dumps(rows, indent=2))
