#!/usr/bin/env python3
"""Re-align VGGT's cloud to the SLAM points directly, as a control.

The first comparison reused experiment 3's registration, which was fitted to the
reference *mesh*. That leaves an ambiguity: when VGGT and SLAM disagree, is the
geometry genuinely different, or is the cloud just sitting a few centimetres off
because it was aligned to something else? A sparse target punishes a small
registration error much harder than a dense one, so the question is not idle.

This re-runs the same symmetric-Chamfer similarity fit from experiment 3, but with
the SLAM points as the target instead of the mesh, and reports the same numbers.
If agreement improves a lot, the earlier gap was registration. If it barely moves,
the disagreement is real.

    python scripts/realign_to_slam.py
"""
import argparse
import json
import pickle
import zlib
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment3/scripts")
from align_and_score import apply, fit_any, normalise  # noqa: E402
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
TAU = 0.02
MAX_VGGT = 120000


def stats(src, tree, diag):
    d, _ = tree.query(src)
    return {"mean_x100": round(float(d.mean()) / diag * 100, 4),
            "within_2pct": round(float((d < TAU * diag).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/viewcrafter_orbits/experiment4/outputs"))
    args = ap.parse_args()
    prev = {(r["object"], r["condition"]): r
            for r in json.loads((args.out_dir / "comparison.json").read_text())}

    rows = []
    for obj in OBJECTS:
        d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        tree_slam = cKDTree(slam)
        # trimesh samples with numpy's global RNG, so without this the "ground
        # truth" surface differs between runs and every number moves a little.
        np.random.seed(0)
        surf = np.asarray(mesh.sample(80000), np.float64)
        tree_surf = cKDTree(surf)
        print(f"\n=== {obj} ===")

        for cond in CONDITIONS:
            f = EXP3 / f"{obj}__{cond}" / "points_object.ply"
            if not f.exists():
                continue
            # Start from the raw cloud, not experiment 3's aligned one, so the fit
            # is genuinely independent of the mesh registration.
            from align_and_score import load_ply
            vggt = load_ply(f)
            if len(vggt) > MAX_VGGT:
                # Seeded per set, not once per run. A single shared stream makes
                # the subsample depend on loop position, so two identical clouds
                # scored in the same pass get different subsamples -- worth up to
                # 0.05 of "SLAM found" on one object, which is the size of the
                # effects being measured.
                # Seeded from the cloud's own bytes, not its name or its position in
                # the loop. Two identical clouds then take the identical
                # subsample, which is what makes `p3_level` vs `v3_360` a real
                # end-to-end reproducibility check rather than a near-miss.
                sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(vggt)))
                vggt = vggt[sub.choice(len(vggt), MAX_VGGT, replace=False)]

            M, cost, up_z, how = fit_any(vggt, slam, EXP3 / f"{obj}__{cond}")
            if M is None:
                print(f"  {cond}: alignment failed")
                continue
            slam_n, slam_c, slam_s = normalise(slam)
            aligned = apply(M, normalise(vggt)[0]) * slam_s + slam_c
            tree_v = cKDTree(aligned)

            row = {"object": obj, "condition": cond,
                   "slam_to_vggt": stats(slam, tree_v, diag),
                   "vggt_to_slam": stats(aligned, tree_slam, diag),
                   "vggt_to_mesh": stats(aligned, tree_surf, diag),
                   "fitted_scale": round(float(np.cbrt(abs(np.linalg.det(M[:3, :3])))), 4),
                   "sym_cost": round(float(cost), 5),
                   "fit": how, "up_z": None if up_z is None else round(up_z, 3)}
            p = prev.get((obj, cond))
            row["slam_found_before"] = p["slam_to_vggt"]["within_2pct"] if p else None
            rows.append(row)
            np.save(args.out_dir / f"{obj}__{cond}__aligned_to_slam.npy",
                    aligned.astype(np.float32))
            was = "  --" if row["slam_found_before"] is None else f"{row['slam_found_before']:.3f}"
            print(f"  {cond:9s} SLAM->VGGT {row['slam_to_vggt']['mean_x100']:6.3f} "
                  f"(found {row['slam_to_vggt']['within_2pct']:.3f}, was {was})   "
                  f"VGGT->mesh {row['vggt_to_mesh']['mean_x100']:6.3f}   "
                  f"scale {row['fitted_scale']:.2f}   {how} up->z "
                  f"{'  --' if up_z is None else f'{up_z:+.2f}'}", flush=True)

    (args.out_dir / "realigned_to_slam.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {len(rows)} re-alignments to {args.out_dir}")


if __name__ == "__main__":
    main()
