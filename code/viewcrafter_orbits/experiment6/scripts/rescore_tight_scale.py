#!/usr/bin/env python3
"""Test whether the sparse results are alignment-limited rather than real.

The change from 1 view to 8 tracks the *fitted alignment scale* at r = +0.98,
and the three worst objects all sit at 0.58-0.64 — close to the 0.55 floor below
which the scorer already rejects a fit as collapsed. That is the signature of the
degeneracy this project has hit before: a similarity fit with free scale shrinks a
cloud into the target rather than matching it.

Sparse-mode clouds are the case most exposed to it. They cover only the observed
arc, so the RMS normalisation that seeds the scale search is computed over a
*partial* object and comes out too small -- the same caveat already flagged for
the +/-45 degree condition in experiment 4, but stronger here.

This re-fits every sparse cloud with the scale band tightened to (0.80, 1.45),
which forbids the shrink, and reports both sets of numbers. If the sparse results
rise and the correlation with fitted scale breaks, the original numbers were
measuring alignment. If they barely move, they were measuring reconstruction.

    python scripts/rescore_tight_scale.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "experiment3" / "scripts"))
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import align_and_score as A  # noqa: E402
import objects as C          # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
TAU = 0.02
TIGHT = (0.80, 1.45)


def main():
    A.SCALE_BAND = TIGHT
    A.SCALE_GRID = (0.9, 1.0, 1.1, 1.25, 1.4)
    old = {f"{r['object']}__{r['condition']}": r
           for r in json.loads((PROJ / "experiment4/outputs/realigned_to_slam.json").read_text())}
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'object':14s} {'old found':>9} {'new found':>9} {'old scale':>9} {'new scale':>9}")
    for name in C.names():
        d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        surf = np.asarray(mesh.sample(80000), np.float64)

        pred = A.load_ply(PROJ / f"experiment3/outputs/{name}__sparse/points_object.ply")
        if len(pred) > 120000:
            pred = pred[rng.choice(len(pred), 120000, replace=False)]
        M, cost = A.fit(pred, slam)
        if M is None:
            print(f"{name:14s} alignment failed"); continue
        slam_n, slam_c, slam_s = A.normalise(slam)
        aligned = A.apply(M, A.normalise(pred)[0]) * slam_s + slam_c
        scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))

        d_s, _ = cKDTree(aligned).query(slam)
        d_m, _ = cKDTree(surf).query(aligned)
        o = old[f"{name}__sparse"]
        row = {"object": name,
               "found_old": o["slam_to_vggt"]["within_2pct"],
               "found_tight": round(float((d_s < TAU * diag).mean()), 4),
               "vggt_to_mesh_old": o["vggt_to_mesh"]["mean_x100"],
               "vggt_to_mesh_tight": round(float(d_m.mean()) / diag * 100, 4),
               "scale_old": o["fitted_scale"], "scale_tight": round(scale, 3)}
        rows.append(row)
        print(f"{name.split('__')[1]:14s} {row['found_old']:9.3f} {row['found_tight']:9.3f} "
              f"{row['scale_old']:9.2f} {row['scale_tight']:9.2f}")

    (PROJ / "experiment6/outputs/rescore_tight_scale.json").write_text(json.dumps(rows, indent=2))
    a = np.array([r["found_old"] for r in rows]); b = np.array([r["found_tight"] for r in rows])
    sa = np.array([r["scale_old"] for r in rows]); sb = np.array([r["scale_tight"] for r in rows])
    print(f"\nmean found: {a.mean():.3f} -> {b.mean():.3f}")
    print(f"correlation of 'found' with fitted scale: {np.corrcoef(sa, a)[0,1]:+.2f} "
          f"-> {np.corrcoef(sb, b)[0,1]:+.2f}")


if __name__ == "__main__":
    main()
