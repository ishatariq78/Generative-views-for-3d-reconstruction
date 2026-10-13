#!/usr/bin/env python3
"""Score VGGT's point cloud against the object's true shape.

VGGT returns geometry in an arbitrary frame at an arbitrary scale, so "is this the
right object?" cannot be answered by comparing coordinates directly. The cloud is
first aligned to the object's reference mesh by a similarity fit -- rotation,
translation and one global scale -- and only then measured. Reflection is
disallowed: a mirrored chair is a wrong chair, and letting Procrustes flip would
hide exactly the failure mode worth catching.

Alignment is a coarse-to-fine search because ICP is only a local method: from a
single starting guess it will happily settle into a confidently wrong pose. A grid
of initial rotations *and scales* is scored first, then the best few are refined
with ICP, then the best of those on the full clouds.

Candidates are ranked by **symmetric** Chamfer throughout, never by ICP's own
cost. This matters more than it sounds. ICP's cost is one-directional -- mean
distance from each source point to its nearest target -- and with free scale the
cheapest way to minimise it is to shrink the source into a dense blob somewhere
inside the target. That is not a hypothetical: the first version of this script
did exactly that, returning fitted scales of 0.07-0.49, precision near 1.0 and
recall near 0.05 for every condition including the control. A symmetric cost has
no such minimum, and fits outside a sane scale band are rejected outright.

Metrics, all scale-free (divided by the object's own diagonal) so a bed and a
chair are comparable, and reported x100 to match ShapeR's convention:

    accuracy       mean distance from a predicted point to the true surface
    completeness   mean distance from the true surface to the nearest prediction
    chamfer        the mean of the two
    recall@2%      share of the true surface within 2% of the diagonal of a point

`accuracy` alone rewards a cloud that covers a fraction of the object very well;
`completeness` alone rewards a cloud that sprays points everywhere. Both are
reported because either on its own is easy to game.

    python scripts/align_and_score.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment3")
SHAPER_MESHES = Path("/home/isha.tariq/shapeR_testing/outputs/eval_balance/per_object")
N_TARGET = 60000
N_COARSE_SRC, N_COARSE_TGT = 1500, 8000
N_FINE_SRC = 12000
N_MID = 48           # grid candidates given a short ICP (24 before the SO(3) grid)
N_KEEP = 6           # of those, refined at full resolution
SCALE_GRID = (0.7, 0.85, 1.0, 1.18, 1.4)   # searched around the RMS-matched scale
SCALE_BAND = (0.55, 1.9)                   # anything outside is a collapsed fit
RECALL_TAU = 0.02    # fraction of the object diagonal


def load_ply(path):
    """Minimal reader for the binary PLY this experiment writes."""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        n = int([l for l in head.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        rec = np.frombuffer(f.read(n * 15), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                   ("r", "u1"), ("g", "u1"), ("b", "u1")], count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)


def normalise(p):
    c = p.mean(0)
    q = p - c
    s = np.sqrt((q ** 2).sum(1)).mean()
    return q / max(s, 1e-9), c, s


def initial_rotations(step_deg=30):
    """A grid over all of SO(3): azimuth x tilt x roll at `step_deg`, tilt through
    180 degrees (744 starts at 30 deg).

    CORRECTED 2026-09-16. The original grid was Rz(az) * Rx(tilt) with tilt in
    {0, 60, 120} deg -- two of the three rotational degrees of freedom and nothing
    within 60 deg of the -z pole -- and ICP then settled *upside down* on most of the
    generated-orbit clouds (VGGT and Pi3 alike; see ~/stable_virtual_camera/exp5/
    scripts/check_upright.py). Every number scored through the old grid is kept in
    outputs/oldfit/; the old function is align_and_score.py.oldfit. Same cost,
    same ICP, same scale grid as before: only the starts changed, plus N_MID.
    """
    out = []
    s = np.radians(step_deg)
    for tilt in np.arange(0, np.pi + 1e-9, s):
        n_az = 1 if tilt < 1e-9 or abs(tilt - np.pi) < 1e-9 else int(round(2 * np.pi / s))
        for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            for roll in np.arange(0, 2 * np.pi - 1e-9, s):
                out.append(trimesh.transformations.rotation_matrix(az, [0, 0, 1])
                           @ trimesh.transformations.rotation_matrix(tilt, [1, 0, 0])
                           @ trimesh.transformations.rotation_matrix(roll, [0, 0, 1]))
    return out


def sym_chamfer(src, tgt_pts, tgt_tree):
    """Mean of both nearest-neighbour directions. Symmetric by construction, so
    shrinking the source cannot buy a lower score."""
    d1, _ = tgt_tree.query(src)
    d2, _ = cKDTree(src).query(tgt_pts)
    return float(0.5 * (d1.mean() + d2.mean()))


def similarity(R, s):
    M = np.eye(4)
    M[:3, :3] = R[:3, :3] * s
    return M


def fit(source, target):
    """Similarity alignment of source onto target. Returns (matrix, symmetric cost).

    Both clouds are first put on a common footing (centroid at the origin, unit
    RMS radius), which makes 1.0 the expected scale -- but only expected, not
    guaranteed, because VGGT recovers just the surfaces it saw while the reference
    mesh is a complete object. So scale is searched over a band around 1.0 rather
    than fixed, and any fit that leaves SCALE_BAND is discarded."""
    src_c, _, _ = normalise(source)
    tgt_c, _, _ = normalise(target)
    rng = np.random.default_rng(0)
    s_small = src_c[rng.choice(len(src_c), min(N_COARSE_SRC, len(src_c)), replace=False)]
    t_small = tgt_c[rng.choice(len(tgt_c), min(N_COARSE_TGT, len(tgt_c)), replace=False)]
    tree_small = cKDTree(t_small)

    # 1. Score the raw grid of orientations x scales. No ICP yet -- this is only
    #    to find which basins are worth the iteration budget.
    grid = []
    for R in initial_rotations():
        for s0 in SCALE_GRID:
            M = similarity(R, s0)
            grid.append((sym_chamfer(apply(M, s_small), t_small, tree_small), M))
    grid.sort(key=lambda x: x[0])

    # 2. Short ICP from each surviving basin, re-ranked by the symmetric cost.
    mid = []
    for _, M in grid[:N_MID]:
        try:
            M2, _, _ = trimesh.registration.icp(s_small, t_small, initial=M,
                                                max_iterations=20, scale=True,
                                                reflection=False)
        except Exception:
            continue
        sc = float(np.cbrt(abs(np.linalg.det(M2[:3, :3]))))
        if not (SCALE_BAND[0] <= sc <= SCALE_BAND[1]):
            continue
        mid.append((sym_chamfer(apply(M2, s_small), t_small, tree_small), M2))
    mid += grid[:N_KEEP]          # keep the un-refined bests as a floor
    mid.sort(key=lambda x: x[0])

    # 3. Long ICP on the full clouds for the finalists.
    s_big = src_c[rng.choice(len(src_c), min(N_FINE_SRC, len(src_c)), replace=False)]
    tree_big = cKDTree(tgt_c)
    best = (np.inf, None)
    for _, M in mid[:N_KEEP]:
        cand = [M]
        try:
            M2, _, _ = trimesh.registration.icp(s_big, tgt_c, initial=M,
                                                max_iterations=60, scale=True,
                                                reflection=False)
            sc = float(np.cbrt(abs(np.linalg.det(M2[:3, :3]))))
            if SCALE_BAND[0] <= sc <= SCALE_BAND[1]:
                cand.append(M2)
        except Exception:
            pass
        for C in cand:
            c = sym_chamfer(apply(C, s_big), tgt_c, tree_big)
            if c < best[0]:
                best = (c, C)
    return best[1], best[0]


def apply(M, p):
    return (M[:3, :3] @ p.T).T + M[:3, 3]


# ---------------------------------------------------------------- upright fit
# CORRECTED AGAIN 2026-09-16, after the SO(3) grid. The wider grid finds the
# upright pose but does not prefer it: on a partial cloud the mirrored pose often
# has the lower symmetric Chamfer, and the free search took it on 21 of 60 sets
# (experiment9/scripts/check_upright.py). No cost can settle this, because nothing
# in a point cloud says which way is up. The reconstructor does say, though: it
# returns the camera poses, and every capture here -- a generated orbit, the
# wearer's arc, ViewCrafter's interpolated path -- has its cameras *above* the
# object. So the normal of the camera ring, pointed away from the cloud, is the
# cloud's own up, and the reference frame is z-up. The fit below uses that as a
# constraint: pre-rotate up onto +z, search azimuth and scale only, refine with
# ICP, and throw away any refinement that tilts more than UPRIGHT_MAX_TILT away.
# The free SO(3) fit remains as the fallback when a set has no camera poses.

UPRIGHT_N_AZ = 36
UPRIGHT_MAX_TILT = 60.0     # degrees; a ring is not exactly level on the tilted paths


def camera_up(set_dir):
    """The cloud's own up, from whichever reconstructor wrote this set.

    Returns (unit vector, source name) or (None, None). The sign is fixed by
    pointing the ring normal from the cloud's centroid towards the cameras."""
    set_dir = Path(set_dir)
    pts = None
    if (set_dir / "vggt.npz").exists():
        z = np.load(set_dir / "vggt.npz")
        E = z["extrinsic"]                                   # world-to-camera, 3x4
        cams = np.stack([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(E))])
        pts, src = z["object_points"], "vggt"
    elif (set_dir / "pi3.npz").exists():
        z = np.load(set_dir / "pi3.npz")
        cams = z["camera_poses"][:, :3, 3]                   # camera-to-world
        pts, src = z["object_points"], "pi3"
    elif (set_dir / "kv.npz").exists():
        z = np.load(set_dir / "kv.npz")
        cams = z["traj"][:, :3, 3]
        pts, src = z["object_points"], "kvtracker"
    if pts is None or len(cams) < 3:
        return None, None
    ctr = cams.mean(0)
    n = np.linalg.svd(cams - ctr)[2][-1]
    n = n / max(np.linalg.norm(n), 1e-12)
    n *= np.sign((ctr - np.asarray(pts, np.float64).mean(0)) @ n)
    return n, src


def rot_to_z(v):
    """Rotation taking unit vector v onto +z (Rodrigues)."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    a = np.cross(v, z); s = np.linalg.norm(a); c = float(v @ z)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


def _tilt_ok(M):
    R = M[:3, :3] / np.cbrt(abs(np.linalg.det(M[:3, :3])))
    return float(R[2, 2]) >= np.cos(np.radians(UPRIGHT_MAX_TILT))


def fit_upright(source, target, up):
    """Similarity alignment with the source's known up held near +z.

    Same cost, same ICP, same scale grid and band as `fit`; the differences are
    the start set (azimuth x scale only, from an upright pre-rotation) and a tilt
    guard on every ICP result. Returns (matrix acting on normalise(source)[0],
    symmetric cost, up_z of the result)."""
    R0 = np.eye(4); R0[:3, :3] = rot_to_z(up)
    src_c, _, _ = normalise(source)
    src_c = apply(R0, src_c)                     # rotation commutes with normalise
    tgt_c, _, _ = normalise(target)
    rng = np.random.default_rng(0)
    s_small = src_c[rng.choice(len(src_c), min(N_COARSE_SRC, len(src_c)), replace=False)]
    t_small = tgt_c[rng.choice(len(tgt_c), min(N_COARSE_TGT, len(tgt_c)), replace=False)]
    tree_small = cKDTree(t_small)

    grid = []
    for az in np.linspace(0, 2 * np.pi, UPRIGHT_N_AZ, endpoint=False):
        Rz = trimesh.transformations.rotation_matrix(az, [0, 0, 1])
        for s0 in SCALE_GRID:
            M = similarity(Rz, s0)
            grid.append((sym_chamfer(apply(M, s_small), t_small, tree_small), M))
    grid.sort(key=lambda x: x[0])

    mid = []
    for _, M in grid[:N_MID]:
        try:
            M2, _, _ = trimesh.registration.icp(s_small, t_small, initial=M,
                                                max_iterations=20, scale=True,
                                                reflection=False)
        except Exception:
            continue
        sc = float(np.cbrt(abs(np.linalg.det(M2[:3, :3]))))
        if not (SCALE_BAND[0] <= sc <= SCALE_BAND[1]) or not _tilt_ok(M2):
            continue
        mid.append((sym_chamfer(apply(M2, s_small), t_small, tree_small), M2))
    mid += grid[:N_KEEP]
    mid.sort(key=lambda x: x[0])

    s_big = src_c[rng.choice(len(src_c), min(N_FINE_SRC, len(src_c)), replace=False)]
    tree_big = cKDTree(tgt_c)
    best = (np.inf, None)
    for _, M in mid[:N_KEEP]:
        cand = [M]
        try:
            M2, _, _ = trimesh.registration.icp(s_big, tgt_c, initial=M,
                                                max_iterations=60, scale=True,
                                                reflection=False)
            sc = float(np.cbrt(abs(np.linalg.det(M2[:3, :3]))))
            if SCALE_BAND[0] <= sc <= SCALE_BAND[1] and _tilt_ok(M2):
                cand.append(M2)
        except Exception:
            pass
        for C in cand:
            c = sym_chamfer(apply(C, s_big), tgt_c, tree_big)
            if c < best[0]:
                best = (c, C)
    if best[1] is None:
        return None, np.inf, None
    M = best[1] @ R0                             # so it acts on the un-rotated source
    R = M[:3, :3] / np.cbrt(abs(np.linalg.det(M[:3, :3])))
    return M, best[0], float((R @ (up / np.linalg.norm(up)))[2])


def fit_any(source, target, set_dir):
    """Upright fit when the set carries camera poses, free SO(3) fit otherwise."""
    up, src = camera_up(set_dir)
    if up is None:
        M, cost = fit(source, target)
        return M, cost, None, "so3"
    M, cost, up_z = fit_upright(source, target, up)
    if M is None:
        M, cost = fit(source, target)
        return M, cost, None, "so3"
    return M, cost, up_z, "upright"


def score(pred, gt_pts, diag):
    tree_gt, tree_pr = cKDTree(gt_pts), cKDTree(pred)
    d_pred, _ = tree_gt.query(pred)     # prediction -> truth
    d_gt, _ = tree_pr.query(gt_pts)     # truth -> prediction
    acc, comp = float(d_pred.mean()) / diag, float(d_gt.mean()) / diag
    return {
        "accuracy_x100": round(acc * 100, 4),
        "completeness_x100": round(comp * 100, 4),
        "chamfer_x100": round((acc + comp) / 2 * 100, 4),
        "recall_at_2pct": round(float((d_gt < RECALL_TAU * diag).mean()), 4),
        "precision_at_2pct": round(float((d_pred < RECALL_TAU * diag).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    args = ap.parse_args()

    sets = sorted(p for p in args.out_dir.iterdir()
                  if p.is_dir() and (p / "points_object.ply").exists())
    if not sets:
        raise SystemExit(f"no VGGT outputs under {args.out_dir}")

    gt_cache, results = {}, []
    for s in sets:
        obj, cond = s.name.rsplit("__", 1)
        if obj not in gt_cache:
            mesh = trimesh.load(SHAPER_MESHES / f"GT__{obj}.glb", force="mesh")
            np.random.seed(0)   # trimesh samples with the global RNG
            pts = np.asarray(mesh.sample(N_TARGET), np.float64)
            diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
            gt_cache[obj] = (pts, diag)
        gt_pts, diag = gt_cache[obj]

        pred = load_ply(s / "points_object.ply")
        if len(pred) < 500:
            print(f"{s.name:28s} only {len(pred)} points -- skipping")
            continue
        rng = np.random.default_rng(0)
        if len(pred) > 200000:
            pred = pred[rng.choice(len(pred), 200000, replace=False)]

        M, cost, up_z, how = fit_any(pred, gt_pts, s)
        if M is None:
            print(f"{s.name:28s} alignment failed")
            continue
        # Bring the prediction into the mesh's own normalised frame, then into
        # metric space, so distances are in the object's real units.
        gt_n, gt_c, gt_s = normalise(gt_pts)
        pred_n = apply(M, normalise(pred)[0])
        pred_metric = pred_n * gt_s + gt_c
        m = score(pred_metric, gt_pts, diag)
        # Keep the aligned cloud: it is what any visual check has to look at, and
        # re-deriving it later would mean re-running the whole ICP search.
        np.save(s / "points_aligned.npy", pred_metric.astype(np.float32))

        scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        row = {"object": obj, "condition": cond, "points": int(len(pred)),
               "icp_cost": round(float(cost), 5), "fitted_scale": round(scale, 4),
               "fit": how, "up_z": None if up_z is None else round(up_z, 3), **m}
        (s / "score.json").write_text(json.dumps(row, indent=2))
        results.append(row)
        print(f"{obj:18s} {cond:9s} {len(pred):7d} pts   "
              f"CD {m['chamfer_x100']:7.3f}   acc {m['accuracy_x100']:7.3f}   "
              f"comp {m['completeness_x100']:7.3f}   recall@2% {m['recall_at_2pct']:.3f}   "
              f"{how} up->z {'  --' if up_z is None else f'{up_z:+.2f}'}", flush=True)

    (args.out_dir / "scores.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} scores to {args.out_dir/'scores.json'}")


if __name__ == "__main__":
    main()
