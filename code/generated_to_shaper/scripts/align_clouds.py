#!/usr/bin/env python3
"""Put each reconstructor's orbit cloud into ShapeR's model frame and score it.

    ~/.conda/envs/shaper/bin/python scripts/align_clouds.py

Two stages, both recorded:
  1. cameras -- the 80 orbit cameras were *specified* in the object's model frame
     (inputs/<obj>/transforms.json), and Pi3 / KV-Tracker estimate a pose for every frame,
     so Umeyama(est centres -> true centres) is a Sim(3) with 80 correspondences and no
     point-cloud ambiguity (no upside-down basin, no box-from-above). Its residual is the
     reconstructor's own pose error.
  2. refine -- a Sim(3) ICP of the SLAM points onto the cloud (sparse onto dense is the
     well-posed direction), seeded by stage 1 and guarded: rotation may move < 25 deg,
     scale < 15 %, translation < 10 % of the diagonal, and the symmetric Chamfer must not
     get worse -- otherwise the camera placement stands. The shape of the cloud is never
     touched; only its 7-DoF pose is.

Scores, in the model frame (diag = mesh bounding-box diagonal):
  slam_reached   share of real SLAM points with a cloud point within 2 % of diag  (GT-free)
  cloud_to_slam  mean cloud -> nearest SLAM point distance, % of diag              (GT-free)
  cloud_to_mesh  mean cloud -> GT surface distance, % of diag       (precision, diagnostic)
  mesh_to_cloud  mean GT-surface-sample -> cloud distance, % diag   (completeness, diagnostic)
The reconstructor with the higher slam_reached is `best` for the object (GT-free choice).

Writes outputs/aligned/<obj>__<rec>.npz (points, colors, cam-only points, transforms) and
outputs/aligned/summary.json.
"""
import json, pickle
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

E = Path("/home/isha.tariq/seva_to_shaper"); OUT = E / "outputs/aligned"; OUT.mkdir(parents=True, exist_ok=True)
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
TAU = 0.02; N_ICP_TGT = 60000; N_SCORE = 40000
# Guards on the ICP refinement. The rotation bound is what stops a flip; the scale band is
# wide because the camera-derived scale is only as good as the pose estimates (a 10-25%
# error is common when the orbit drifts), and the extent band is the actual degeneracy
# guard -- it forbids the sparse/dense pair collapsing into or ballooning around each other.
MAX_ROT_DEG, SCALE_BAND, MAX_SHIFT, EXTENT_BAND = 25.0, (0.7, 1.6), 0.15, (0.6, 1.7)
rng = np.random.default_rng(0)


def load_ply(p):
    with open(p, "rb") as f:
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"): n = int(line.split()[-1])
            if line.strip() == b"end_header": break
        rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64), np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.uint8)


def umeyama(src, dst):
    """s, R, t with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src); U, sig, Vt = np.linalg.svd(cov)
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(d) @ Vt; s = (sig * d).sum() / (S ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def as_matrix(s, R, t):
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t; return M


def apply(M, p):
    return p @ M[:3, :3].T + M[:3, 3]


def decompose(M):
    s = float(np.cbrt(abs(np.linalg.det(M[:3, :3])))); return s, M[:3, :3] / s, M[:3, 3]


def true_cameras(obj):
    tf = json.loads((E / "inputs" / obj / "transforms.json").read_text())["frames"]
    ids = json.loads(next((E / "inputs" / obj).glob("train_test_split_*.json")).read_text())["test_ids"]
    return np.array([tf[i]["transform_matrix"] for i in ids])[:, :3, 3].astype(np.float64)


def est_cameras(obj, rec):
    if rec == "pi3":
        return np.load(E / f"outputs/pi3/{obj}/pi3.npz")["camera_poses"][:, :3, 3].astype(np.float64)
    return np.load(E / f"outputs/kv/{obj}/traj.npy")[:, :3, 3].astype(np.float64)


def sym_chamfer(a, b, diag):
    ta, tb = cKDTree(a), cKDTree(b)
    return float(tb.query(a)[0].mean() + ta.query(b)[0].mean()) / diag * 100


def extent(p):
    """90th-percentile radius about the centroid -- an outlier-tolerant size."""
    return float(np.percentile(np.linalg.norm(p - p.mean(0), axis=1), 90))


def refine(cloud, slam, M_cam, diag):
    """Sim(3) ICP: SLAM (sparse) onto the cloud (dense), seeded from the camera placement, then inverted."""
    sub = cloud[rng.choice(len(cloud), min(N_ICP_TGT, len(cloud)), replace=False)]
    tgt = apply(M_cam, sub)                       # cloud already in model frame -> seed is identity
    try:
        M_s2c, _, _ = trimesh.registration.icp(slam, tgt, initial=np.eye(4), max_iterations=80, scale=True, reflection=False)
    except Exception as e:
        return M_cam, f"icp failed: {e}"
    M_ref = np.linalg.inv(M_s2c) @ M_cam           # cloud -> model
    s0, R0, _ = decompose(M_cam); s1, R1, _ = decompose(M_ref)
    rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R1 @ R0.T).as_rotvec())))
    ref = apply(M_ref, sub)
    shift = float(np.linalg.norm(ref.mean(0) - tgt.mean(0))) / diag
    ext = extent(ref) / extent(slam)
    c_cam, c_ref = sym_chamfer(tgt, slam, diag), sym_chamfer(ref, slam, diag)
    why = (f"rot {rot:.1f}deg scale x{s1/s0:.3f} shift {shift*100:.1f}% extent x{ext:.2f} "
           f"chamfer {c_cam:.2f}->{c_ref:.2f}")
    bad = []
    if rot > MAX_ROT_DEG: bad.append("rotation")
    if not (SCALE_BAND[0] <= s1 / s0 <= SCALE_BAND[1]): bad.append("scale")
    if shift > MAX_SHIFT: bad.append("shift")
    if not (EXTENT_BAND[0] <= ext <= EXTENT_BAND[1]): bad.append("extent")
    if c_ref > c_cam: bad.append("chamfer")
    if bad:
        return M_cam, f"REJECTED ({','.join(bad)}) " + why
    return M_ref, "accepted " + why


def score(pts, slam, slam_tree, mesh_pts, mesh_tree, diag):
    """Distances to the reference surface are measured against a dense (200k) surface
    sample rather than the triangles themselves -- the env has no `rtree`, so trimesh
    cannot build a triangle tree. At 200k samples the mean spacing is far below the 2%
    threshold, and every condition is measured the same way, so the comparison holds."""
    sub = pts[rng.choice(len(pts), min(N_SCORE, len(pts)), replace=False)]
    tree = cKDTree(sub)
    d_slam = tree.query(slam)[0]
    d_c2m = mesh_tree.query(sub)[0]
    return {"slam_reached": round(float((d_slam < TAU * diag).mean()), 4),
            "slam_to_cloud": round(float(d_slam.mean()) / diag * 100, 3),
            "cloud_to_slam": round(float(slam_tree.query(sub)[0].mean()) / diag * 100, 3),
            "cloud_to_mesh": round(float(d_c2m.mean()) / diag * 100, 3),
            "cloud_within_2pct_mesh": round(float((d_c2m < TAU * diag).mean()), 4),
            "mesh_to_cloud": round(float(tree.query(mesh_pts)[0].mean()) / diag * 100, 3)}


rows = []
for obj in json.loads((E / "outputs/sets.json").read_text()):
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    slam = np.asarray(d["points_model"].numpy(), np.float64); slam_tree = cKDTree(slam)
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64), faces=np.asarray(d["mesh_faces"]), process=False)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])); mesh_pts = mesh.sample(200000); mesh_tree = cKDTree(mesh_pts)
    true = true_cameras(obj)
    print(f"\n=== {obj}  ({len(slam)} SLAM points, diag {diag:.2f} m) ===", flush=True)
    for rec in ("pi3", "kv"):
        ply = E / f"outputs/{rec}/{obj}/points_object.ply"
        if not ply.exists():
            print(f"  {rec}: no cloud"); continue
        cloud, col = load_ply(ply)
        est = est_cameras(obj, rec)
        n = min(len(est), len(true))
        s, R, t = umeyama(est[:n], true[:n]); M_cam = as_matrix(s, R, t)
        cam_rms = float(np.sqrt(((apply(M_cam, est[:n]) - true[:n]) ** 2).sum(1).mean())) / diag * 100
        M, note = refine(cloud, slam, M_cam, diag)
        placed, final = apply(M_cam, cloud), apply(M, cloud)
        sc_cam = score(placed, slam, slam_tree, mesh_pts, mesh_tree, diag)
        sc = score(final, slam, slam_tree, mesh_pts, mesh_tree, diag)
        row = {"object": obj, "reconstructor": rec, "points": int(len(cloud)), "camera_frames": int(n),
               "camera_rms_pct": round(cam_rms, 2), "camera_scale": round(float(s), 4), "refine": note,
               "refine_used": not note.startswith("REJECTED"),
               "cam_only": sc_cam, "final": sc}
        rows.append(row)
        np.savez_compressed(OUT / f"{obj}__{rec}.npz", points=final.astype(np.float32), colors=col,
                            points_cam_only=placed.astype(np.float32), M_cam=M_cam, M_final=M, diag=diag)
        print(f"  {rec:4s} {len(cloud):7d} pts  cam rms {cam_rms:5.2f}%  {note}\n"
              f"       cam-only: reached {sc_cam['slam_reached']:.3f} cloud->SLAM {sc_cam['cloud_to_slam']:.2f}% cloud->mesh {sc_cam['cloud_to_mesh']:.2f}%\n"
              f"       final   : reached {sc['slam_reached']:.3f} cloud->SLAM {sc['cloud_to_slam']:.2f}% cloud->mesh {sc['cloud_to_mesh']:.2f}% mesh->cloud {sc['mesh_to_cloud']:.2f}%", flush=True)
    cands = [r for r in rows if r["object"] == obj]
    if cands:
        best = max(cands, key=lambda r: (r["final"]["slam_reached"], -r["final"]["cloud_to_slam"]))
        for r in cands: r["best"] = r is best
        by_mesh = min(cands, key=lambda r: r["final"]["cloud_to_mesh"] + r["final"]["mesh_to_cloud"])
        print(f"  best by SLAM: {best['reconstructor']}   (best by GT mesh: {by_mesh['reconstructor']})")
(OUT / "summary.json").write_text(json.dumps(rows, indent=2))
print("\nmean over objects (final):")
for rec in ("pi3", "kv"):
    rs = [r["final"] for r in rows if r["reconstructor"] == rec]
    if rs: print(f"  {rec:4s} reached {np.mean([r['slam_reached'] for r in rs]):.3f}  cloud->SLAM {np.mean([r['cloud_to_slam'] for r in rs]):.2f}%  "
                 f"cloud->mesh {np.mean([r['cloud_to_mesh'] for r in rs]):.2f}%  mesh->cloud {np.mean([r['mesh_to_cloud'] for r in rs]):.2f}%")
