#!/usr/bin/env python3
"""The gap-fill condition: every SLAM point kept, generated points added only where
there are none.

    ~/.conda/envs/shaper/bin/python scripts/make_pkls_gapfill.py

`union` adds the whole generated cloud on top of the SLAM points, so it dilutes precise
real points with less precise generated ones in regions that were already covered. This
adds a generated point only where it is farther than TAU from every SLAM point -- i.e.
only the parts of the object the capture never saw. It strictly adds information and
removes none, which is the one way a generated cloud can help an object whose real cloud
is already good.

The source cloud is chosen GT-free by **mean cloud->SLAM distance** (precision against the
real points), not by the share of SLAM points reached. Reach rewards a hazy cloud for
covering more, and on this data it picked the mesh-worse cloud on 2 of 6 objects; the
precision criterion agrees with the reference mesh on 6 of 6.

On a sparse capture almost the whole generated cloud is "new" (89-96% of it here), so
gap-filling alone is nearly the same thing as `union`. What actually distinguishes them is
the **mixing ratio**: `union` puts 4096 generated points beside 300 real ones, so 93% of
what ShapeR sees is generated and the precise points are drowned. So the added points are
capped as a multiple of the object's own SLAM count:

  gap1   slam + 1x n_slam gap-filled points   (half real)
  gap4   slam + 4x n_slam, capped at CAP      (mostly generated, ~= union)

Writes outputs/pkls/<obj>__gap{1,4}.pkl and outputs/pkls/manifest_gapfill.json.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / "make_pkls.py").read_text().split("\nsummary = ")[0]
ns = {}; exec(compile(src, "make_pkls.py", "exec"), ns)
voxel_unique, subsample, write = ns["voxel_unique"], ns["subsample"], ns["write"]
E, SHAPER_DATA, BOX_MARGIN, CAP = ns["E"], ns["SHAPER_DATA"], ns["BOX_MARGIN"], ns["CAP"]

TAU = 0.02          # a generated point counts as new if no SLAM point is within 2% of the diagonal
aligned = {(r["object"], r["reconstructor"]): r for r in json.loads((E / "outputs/aligned/summary.json").read_text())}

rows = []
for obj in json.loads((E / "outputs/sets.json").read_text()):
    base = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    bounds = base["bounds"].numpy(); slam = base["points_model"].numpy()
    diag = float(np.linalg.norm(np.asarray(base["mesh_vertices"]).max(0) - np.asarray(base["mesh_vertices"]).min(0)))
    cands = {r: aligned[(obj, r)] for r in ("pi3", "kv") if (obj, r) in aligned}
    pick = min(cands, key=lambda r: cands[r]["final"]["cloud_to_slam"])
    p = np.load(E / f"outputs/aligned/{obj}__{pick}.npz")["points"].astype(np.float64)
    p = voxel_unique(p[np.all(np.abs(p) <= BOX_MARGIN * bounds, axis=1)], bounds)
    far = cKDTree(slam).query(p)[0] > TAU * diag
    new = p[far]
    counts = {}
    for name, mult in (("gap1", 1), ("gap4", 4)):
        added = subsample(new, min(mult * len(slam), CAP))
        counts[name] = write(obj, name, base, np.concatenate([slam, added]) if len(added) else slam)
    rows.append({"object": obj, "source": pick, "slam": int(len(slam)),
                 "candidate_far_fraction": round(float(far.mean()), 3), "counts": counts,
                 "cloud_to_slam": {r: cands[r]["final"]["cloud_to_slam"] for r in cands}})
    print(f"{obj:26s} source={pick:3s}  slam={len(slam):5d}  {far.mean():.0%} of the cloud uncovered  "
          + "  ".join(f"{k}={v}" for k, v in counts.items()))
(E / "outputs/pkls/manifest_gapfill.json").write_text(json.dumps(rows, indent=2))
