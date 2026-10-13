#!/usr/bin/env python3
"""Pick each object's output by how well it agrees with the capture's own SLAM points.

    ~/.conda/envs/shaper/bin/python scripts/select_by_slam.py

No condition wins on every object -- clean generated views rescue an occluded capture,
the point mixes help a clean one -- so running several and *choosing* is the thing that
can beat the baseline everywhere. The choice has to be made without the reference mesh,
and it can be: the SLAM points are an input ShapeR was given, not ground truth, and a
correct mesh has to pass close to real measurements of the object's surface. So each
candidate is scored by the mean distance from the SLAM points to its surface, and the
lowest wins. The reference mesh is used only afterwards, to report what the choice cost
or gained.

Reports: the baseline, the oracle (best possible pick), and this selector.
"""
import json, pickle
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial import cKDTree

E = Path("/home/isha.tariq/seva_to_shaper"); SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
SEED = 0
CONDS = ["slam", "gap1", "views", "vonly", "vg80", "v80", "vgap"]
N_SURF = 80000


def surface(mesh, n=N_SURF):
    return np.asarray(mesh.sample(n), np.float64)


rows, scores = [], {}
for obj in json.loads((E / "outputs/sets.json").read_text()):
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    # `InferenceDataset.rescale_back` multiplies the mesh by 1/scale on export, so the saved
    # GLB is already in model units -- the same frame as `points_model`. Scaling the points
    # here as well was a bug: it shrank them and made every agreement figure meaningless.
    slam = d["points_model"].numpy().astype(np.float64)
    verts = np.asarray(d["mesh_vertices"], np.float64)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    cand = {}
    for c in CONDS:
        p = E / f"outputs/seeded/s{SEED}/per_object/{obj}__{c}.glb"
        j = E / f"outputs/seeded/s{SEED}/per_object/{obj}__{c}.json"
        if not (p.exists() and j.exists()): continue
        m = trimesh.load(p, force="mesh")
        if len(m.vertices) < 10: continue
        tree = cKDTree(surface(m))
        dist = tree.query(slam)[0]
        cand[c] = {"slam_to_mesh_pct": float(dist.mean()) / diag * 100,
                   "cd": json.load(open(j))["metrics"]["chamfer"] * 100,
                   "f1": json.load(open(j))["metrics"]["f-score"]}
    if "slam" not in cand: continue
    chosen = min(cand, key=lambda c: cand[c]["slam_to_mesh_pct"])
    best = min(cand, key=lambda c: cand[c]["cd"])
    rows.append({"object": obj, "chosen": chosen, "oracle": best,
                 "baseline_cd": cand["slam"]["cd"], "chosen_cd": cand[chosen]["cd"], "oracle_cd": cand[best]["cd"],
                 "baseline_f1": cand["slam"]["f1"], "chosen_f1": cand[chosen]["f1"],
                 "candidates": cand})
    scores[obj] = cand
    print(f"{obj:24s} picked {chosen:6s} (agreement {cand[chosen]['slam_to_mesh_pct']:.2f}%)  "
          f"CD {cand['slam']['cd']:5.2f} -> {cand[chosen]['cd']:5.2f}   F1 {cand['slam']['f1']:.3f} -> {cand[chosen]['f1']:.3f}"
          f"   [oracle {best}: {cand[best]['cd']:.2f}]")

b = np.array([r["baseline_cd"] for r in rows]); c = np.array([r["chosen_cd"] for r in rows]); o = np.array([r["oracle_cd"] for r in rows])
bf = np.array([r["baseline_f1"] for r in rows]); cf = np.array([r["chosen_f1"] for r in rows])
print(f"\nmean CD  baseline {b.mean():.3f}   selected {c.mean():.3f} ({c.mean()-b.mean():+.3f})   oracle {o.mean():.3f} ({o.mean()-b.mean():+.3f})")
print(f"mean F1  baseline {bf.mean():.3f}   selected {cf.mean():.3f} ({cf.mean()-bf.mean():+.3f})")
print(f"objects improved: CD {(c<b).sum()}/{len(rows)}   F1 {(cf>bf).sum()}/{len(rows)}")
(E / "outputs/selected.json").write_text(json.dumps(rows, indent=2))
