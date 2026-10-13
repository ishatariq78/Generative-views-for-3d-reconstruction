#!/usr/bin/env python3
"""Compare VGGT's recovered points against ShapeR's own SLAM points.

Experiment 3 scored VGGT against the *reference mesh* -- a complete, watertight
surface. This scores it against the thing ShapeR actually ingests: the sparse
semi-dense SLAM cloud the Aria capture produced. That is a different question and
a fairer one, because both sides are now point clouds of observed surface rather
than one estimate against an idealised ground truth.

It also lets the SLAM points themselves be put on trial. They are treated as
"ground truth" by the pipeline, but they are an estimate too -- roughly 2-3k
points, wherever the tracker happened to find features -- so this measures how far
they sit from the reference mesh as well. Without that baseline, "VGGT disagrees
with SLAM by X" has no scale to be read against.

Frames line up for free. The exported GT__<obj>.glb turns out to be bit-identical
to the pickle's mesh_vertices, so experiment 3's aligned clouds are already in
model coordinates -- the same frame points_model lives in. Nothing is re-registered
here, which also means this comparison inherits experiment 3's mesh alignment: it
measures agreement *given* the best available registration, not registration.

Distances are divided by the object's diagonal and reported x100, matching the
convention used throughout this project.

    python scripts/compare_slam_points.py
"""
import argparse
import json
import sys
import pickle
import zlib
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as OBJECTS_CFG  # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
EXP3 = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/outputs")
OBJECTS = OBJECTS_CFG.names()
# `sparse` is experiment 6: ViewCrafter given many views instead of one.
# `mv360` is experiment 7: the same 360 degree path as gen360, but rendered
# from an eight-view cloud instead of a one-view one.
# `v3_360` is experiment 8: three views — ViewCrafter's designed operating
# point — on the same 360 degree path, at the elevation the views were taken from.
# `p3_*` is experiment 9: experiment 5's six camera paths re-run on the
# three-view cloud, to ask whether the route matters once the geometry is
# better. `p3_level` repeats `v3_360` and should reproduce it.
CONDITIONS = ["gen360", "gen45", "real", "sparse", "mv360", "v3_360", "v3_360g", "seva360", "seva360f", "seva360p", "seva360fp", "seva360fk5", "seva360fk10", "seva360fk1"] + \
             ["p3_" + p for p in ("level", "high", "low", "helix", "wobble", "dolly")]
N_SURFACE = 80000
TAU = 0.02          # 2% of the object diagonal, as in experiment 3
MAX_VGGT = 200000


def nn_stats(src, tgt_tree, diag):
    d, _ = tgt_tree.query(src)
    return {
        "mean_x100": round(float(d.mean()) / diag * 100, 4),
        "median_x100": round(float(np.median(d)) / diag * 100, 4),
        "within_2pct": round(float((d < TAU * diag).mean()), 4),
    }, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/viewcrafter_orbits/experiment4/outputs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results, baselines = [], []
    for obj in OBJECTS:
        d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        np.random.seed(0)   # see realign_to_slam.py: trimesh uses the global RNG
        surf = np.asarray(mesh.sample(N_SURFACE), np.float64)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        tree_slam, tree_surf = cKDTree(slam), cKDTree(surf)

        # How good is the "ground truth" itself? Both directions, because the SLAM
        # cloud can be accurate (every point on the surface) and still cover almost
        # none of it.
        s_acc, _ = nn_stats(slam, tree_surf, diag)
        s_cov, _ = nn_stats(surf, tree_slam, diag)
        baselines.append({
            "object": obj, "slam_points": int(len(slam)),
            "object_diagonal_m": round(diag, 4),
            "slam_to_mesh": s_acc, "mesh_to_slam": s_cov,
            "slam_density_per_m2": round(len(slam) / max(float(mesh.area), 1e-9), 1),
        })
        print(f"\n=== {obj}  ({len(slam)} SLAM points, diagonal {diag:.3f} m) ===")
        print(f"  SLAM -> mesh   mean {s_acc['mean_x100']:6.3f}   within 2%: {s_acc['within_2pct']:.3f}"
              f"      (how accurate the SLAM points are)")
        print(f"  mesh -> SLAM   mean {s_cov['mean_x100']:6.3f}   within 2%: {s_cov['within_2pct']:.3f}"
              f"      (how much of the object they cover)")

        for cond in CONDITIONS:
            f = EXP3 / f"{obj}__{cond}" / "points_aligned.npy"
            if not f.exists():
                print(f"  {cond}: missing {f}")
                continue
            vggt = np.load(f).astype(np.float64)
            if len(vggt) > MAX_VGGT:
                # Seeded from the cloud's own bytes, not its name or its position in
                # the loop. Two identical clouds then take the identical
                # subsample, which is what makes `p3_level` vs `v3_360` a real
                # end-to-end reproducibility check rather than a near-miss.
                sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(vggt)))
                vggt = vggt[sub.choice(len(vggt), MAX_VGGT, replace=False)]
            tree_vggt = cKDTree(vggt)

            v2s, _ = nn_stats(vggt, tree_slam, diag)      # VGGT -> SLAM
            s2v, d_s2v = nn_stats(slam, tree_vggt, diag)  # SLAM -> VGGT
            v2m, _ = nn_stats(vggt, tree_surf, diag)      # VGGT -> mesh
            m2v, _ = nn_stats(surf, tree_vggt, diag)      # mesh -> VGGT

            row = {
                "object": obj, "condition": cond,
                "vggt_points": int(len(vggt)), "slam_points": int(len(slam)),
                "density_ratio": round(len(vggt) / len(slam), 1),
                "vggt_to_slam": v2s, "slam_to_vggt": s2v,
                "vggt_to_mesh": v2m, "mesh_to_vggt": m2v,
                "symmetric_vggt_slam_x100": round((v2s["mean_x100"] + s2v["mean_x100"]) / 2, 4),
            }
            results.append(row)
            # Per-SLAM-point distance to the nearest VGGT point: this is what the
            # figures colour by, and re-deriving it means rebuilding the KD-tree.
            np.save(args.out_dir / f"{obj}__{cond}__slam_to_vggt.npy",
                    (d_s2v / diag).astype(np.float32))
            print(f"  {cond:7s} {len(vggt):7d} pts ({row['density_ratio']:6.1f}x SLAM)   "
                  f"SLAM->VGGT {s2v['mean_x100']:6.3f} (found {s2v['within_2pct']:.3f})   "
                  f"VGGT->SLAM {v2s['mean_x100']:6.3f}   VGGT->mesh {v2m['mean_x100']:6.3f}")

    (args.out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    (args.out_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
