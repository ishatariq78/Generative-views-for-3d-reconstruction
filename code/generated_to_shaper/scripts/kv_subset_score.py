#!/usr/bin/env python3
"""Put each KV-Tracker orbit cloud in the model frame, swap it in for the SLAM points, score.

    ~/.conda/envs/shaper/bin/python scripts/kv_subset_score.py

Placement is the same two-stage method the earlier study used and validated: the 80 orbit
cameras are known in the object's model frame and the tracker estimates one per frame, so
Umeyama(estimated centres -> true centres) fixes the pose with no point-cloud ambiguity;
then a Sim(3) ICP of the SLAM points onto the cloud refines it, accepted only if rotation
moves < 25 deg, scale stays in [0.7, 1.6], the centre moves < 15% of the diagonal, the
extent ratio stays sane, and the symmetric Chamfer improves. Only the 7-DoF pose is touched.

The cloud then replaces `points_model` (with its per-point std arrays zeroed, as the
benchmark pkls have them); everything else in the sample is the benchmark's own bytes, so
the conditioning is the only difference from the baseline.
"""
import json, os, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

E = Path("/home/isha.tariq/seva_to_shaper")
DATA = Path("/home/isha.tariq/shapeR_testing/data")
sys.path.insert(0, str(E / "scripts"))
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
SUBJ_SAT, SUBJ_VAL = 0.15, 0.70
CAP, NUM_BINS, BOX_MARGIN = 4096, 256, 1.1
MAX_ROT, SCALE_BAND, MAX_SHIFT, EXTENT_BAND = 25.0, (0.7, 1.6), 0.15, (0.6, 1.7)
rng = np.random.default_rng(0)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src); U, sig, Vt = np.linalg.svd(cov)
    d = np.ones(3); d[-1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag(d) @ Vt; s = (sig * d).sum() / (S ** 2).sum(1).mean()
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = mu_d - s * R @ mu_s
    return M


def apply(M, p):
    return p @ M[:3, :3].T + M[:3, 3]


def decomp(M):
    s = float(np.cbrt(abs(np.linalg.det(M[:3, :3])))); return s, M[:3, :3] / s


def sym_chamfer(a, b, diag):
    return float(cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / diag * 100


def voxel_unique(pts, bounds):
    scale = 0.9 / float(np.max(bounds)); q = pts * scale
    c = np.clip(np.round((q + 1.0) / 2.0 * NUM_BINS).astype(np.int64), 0, NUM_BINS - 1)
    _, inv = np.unique(c, axis=0, return_inverse=True); inv = inv.reshape(-1)
    sums = np.zeros((inv.max() + 1, 3)); np.add.at(sums, inv, pts)
    return sums / np.bincount(inv, minlength=len(sums)).astype(np.float64)[:, None]


def true_cams(obj):
    tf = json.loads((E / "inputs_full" / obj / "transforms.json").read_text())["frames"]
    ids = json.loads(next((E / "inputs_full" / obj).glob("train_test_split_*.json")).read_text())["test_ids"]
    return np.array([tf[i]["transform_matrix"] for i in ids])[:, :3, 3].astype(np.float64)


def main():
    names = [r["name"] for r in json.loads((E / "objects_kvtest.json").read_text())]
    pkl_dir = E / "outputs_sub/kvpkls"; pkl_dir.mkdir(parents=True, exist_ok=True)
    out_root = E / "outputs_sub/kvseeded"
    rows = []
    for i, obj in enumerate(names, 1):
        if all((E / f"outputs_sub/kvseeded/s{s_}/per_object/{obj}__{c}.json").exists()
               for s_ in (0, 1, 2) for c in ("slam", "kvpts")):
            print(f"[{i}/{len(names)}] {obj}: already scored", flush=True); continue
        d = E / f"outputs_sub/kv/{obj}"
        # scripts/kv_ply_to_npz.py converts the tracker's open3d PLY in the kvtracker env,
        # because open3d is not installed in the scoring env.
        npz = d / "cloud.npz"
        if not npz.exists() or not (d / "traj.npy").exists():
            print(f"[{i}/{len(names)}] {obj}: no tracker output"); continue
        base = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
        slam = base["points_model"].numpy().astype(np.float64)
        verts = np.asarray(base["mesh_vertices"], np.float64)
        diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        bounds = base["bounds"].numpy()

        z = np.load(npz)
        pts = z["points"].astype(np.float64); col = z["colors"].astype(np.float64)
        mx, mn = col.max(1), col.min(1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        pts = pts[(sat > SUBJ_SAT) | (mx < SUBJ_VAL)]                  # object, not backdrop
        est = np.load(d / "traj.npy").astype(np.float64)[:, :3, 3]
        true = true_cams(obj); n = min(len(est), len(true))
        M_cam = umeyama(est[:n], true[:n])
        cam_rms = float(np.sqrt(((apply(M_cam, est[:n]) - true[:n]) ** 2).sum(1).mean())) / diag * 100

        sub = pts[rng.choice(len(pts), min(60000, len(pts)), replace=False)]
        tgt = apply(M_cam, sub)
        note, M = "cameras only", M_cam
        try:
            import trimesh
            M_s2c, _, _ = trimesh.registration.icp(slam, tgt, initial=np.eye(4),
                                                   max_iterations=80, scale=True, reflection=False)
            M_ref = np.linalg.inv(M_s2c) @ M_cam
            s0, R0 = decomp(M_cam); s1, R1 = decomp(M_ref)
            rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R1 @ R0.T).as_rotvec())))
            ref = apply(M_ref, sub)
            shift = float(np.linalg.norm(ref.mean(0) - tgt.mean(0))) / diag
            ext = float(np.percentile(np.linalg.norm(ref - ref.mean(0), axis=1), 90) /
                        np.percentile(np.linalg.norm(slam - slam.mean(0), axis=1), 90))
            ok = (rot <= MAX_ROT and SCALE_BAND[0] <= s1 / s0 <= SCALE_BAND[1]
                  and shift <= MAX_SHIFT and EXTENT_BAND[0] <= ext <= EXTENT_BAND[1]
                  and sym_chamfer(ref, slam, diag) <= sym_chamfer(tgt, slam, diag))
            if ok:
                M, note = M_ref, f"refined (rot {rot:.1f}deg scale x{s1/s0:.2f})"
            else:
                note = f"refinement rejected (rot {rot:.1f}deg scale x{s1/s0:.2f})"
        except Exception as e:
            note = f"icp failed: {e}"

        placed = apply(M, pts)
        inside = np.all(np.abs(placed) <= BOX_MARGIN * bounds, axis=1)
        if inside.sum() < 50:
            # The whole cloud landed outside the object box: the placement failed, and an
            # empty cloud is not a condition worth running. Record it as a failure instead.
            rows.append({"object": obj, "camera_rms_pct": round(cam_rms, 2),
                         "placement": note + " | PLACEMENT FAILED: no points inside the box",
                         "raw_points": int(len(pts)), "points_used": 0,
                         "slam_points": int(len(slam)), "slam_reached": 0.0})
            print(f"[{i}/{len(names)}] {obj:30s} cam rms {cam_rms:6.1f}%  "
                  f"PLACEMENT FAILED - 0 of {len(pts)} points inside the object box", flush=True)
            continue
        cloud = voxel_unique(placed[inside], bounds)
        if len(cloud) > CAP:
            cloud = cloud[rng.choice(len(cloud), CAP, replace=False)]
        reach = float((cKDTree(cloud).query(slam)[0] < 0.02 * diag).mean()) if len(cloud) else 0.0

        import torch
        s = dict(base)
        s["points_model"] = torch.from_numpy(np.ascontiguousarray(cloud, dtype=np.float32))
        s["inv_dist_std"] = torch.zeros(len(cloud), dtype=torch.float32)
        s["dist_std"] = torch.zeros(len(cloud), dtype=torch.float32)
        p = pkl_dir / f"{obj}__kvpts.pkl"
        with open(p, "wb") as fh:
            pickle.dump(s, fh)
        rows.append({"object": obj, "camera_rms_pct": round(cam_rms, 2), "placement": note,
                     "raw_points": int(len(pts)), "points_used": int(len(cloud)),
                     "slam_points": int(len(slam)), "slam_reached": round(reach, 3)})
        print(f"[{i}/{len(names)}] {obj:30s} cam rms {cam_rms:6.1f}%  {note:42s} "
              f"{len(cloud):5d} pts (SLAM {len(slam)})  reach {reach:.2f}", flush=True)

        for seed in (0, 1, 2):
            subprocess.run([sys.executable, str(E / "scripts/run_shaper_nviews.py"),
                            "--input_pkls", str(DATA / f"{obj}.pkl"), str(p),
                            "--out_dir", str(out_root / f"s{seed}"), "--seed", str(seed),
                            "--num_images", "24", "--skip_done"],
                           cwd="/home/isha.tariq/shapeR_testing/exp_object_aware/scripts")
            src = out_root / f"s{seed}/per_object/{obj}.json"
            dst = out_root / f"s{seed}/per_object/{obj}__slam.json"
            if src.exists() and not dst.exists():
                src.rename(dst)
        p.unlink(missing_ok=True)
    (E / "outputs_sub/kvtest_placement.json").write_text(json.dumps(rows, indent=2))
    print("### done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
