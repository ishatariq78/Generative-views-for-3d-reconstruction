#!/usr/bin/env python3
"""Write one ShapeR input pkl per (object, condition), changing only the 3D points.

    ~/.conda/envs/shaper/bin/python scripts/make_pkls.py

Everything else in the sample -- images, masks (which come from the precomputed
`object_point_projections`, not from `points_model`), cameras, caption, bounds and the
reference mesh -- is byte-identical to the benchmark pkl, so any change in ShapeR's
output is due to the point conditioning alone. The per-point uncertainties
`inv_dist_std` / `dist_std` are zeros for the generated points (the benchmark pkls
already carry zeros for the SLAM points, so both pass the loader's threshold).

Conditions (outputs/pkls/<obj>__<cond>.pkl):
  slam      the original SLAM points, untouched  (baseline, re-run for a fair seed draw)
  pi3       Pi3 orbit cloud, aligned, voxel-deduplicated on ShapeR's own 256^3 grid, <= CAP points
  kv        KV-Tracker orbit cloud, same treatment
  best_n    the better of the two (by SLAM reach, GT-free), subsampled to the SLAM point count
  union     SLAM points + the `best` cloud (CAP points)
The generated clouds are clipped to 1.1x the object box first (the loader itself drops
anything beyond 1/0.9 of the box), which removes backdrop haze.
"""
import json, pickle
from pathlib import Path
import numpy as np, torch

E = Path("/home/isha.tariq/seva_to_shaper"); SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
OUT = E / "outputs/pkls"; OUT.mkdir(parents=True, exist_ok=True)
CAP, NUM_BINS, BOX_MARGIN = 4096, 256, 1.1
rng = np.random.default_rng(0)


def voxel_unique(pts, bounds):
    """Average points per cell of the grid ShapeR discretises on (scale 0.9/max(bounds), 256 bins over [-1,1])."""
    scale = 0.9 / float(np.max(bounds)); q = pts * scale
    coords = np.clip(np.round((q + 1.0) / 2.0 * NUM_BINS).astype(np.int64), 0, NUM_BINS - 1)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.zeros((inv.max() + 1, 3)); np.add.at(sums, inv, pts)
    cnt = np.bincount(inv, minlength=len(sums)).astype(np.float64)
    return sums / cnt[:, None]


def subsample(pts, n):
    return pts if len(pts) <= n else pts[rng.choice(len(pts), n, replace=False)]


def write(obj, cond, base, pts):
    d = dict(base)
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    d["points_model"] = torch.from_numpy(pts)
    d["inv_dist_std"] = torch.zeros(len(pts), dtype=torch.float32)
    d["dist_std"] = torch.zeros(len(pts), dtype=torch.float32)
    d["s2s_condition"] = cond
    with open(OUT / f"{obj}__{cond}.pkl", "wb") as f:
        pickle.dump(d, f)
    return len(pts)


summary = json.loads((E / "outputs/aligned/summary.json").read_text())
manifest = []
for obj in json.loads((E / "outputs/sets.json").read_text()):
    base = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    bounds = base["bounds"].numpy(); slam = base["points_model"].numpy()
    rows = {r["reconstructor"]: r for r in summary if r["object"] == obj}
    if not rows:
        print(f"!! {obj}: no aligned clouds"); continue
    best = next(r["reconstructor"] for r in rows.values() if r.get("best"))
    clouds = {}
    for rec in rows:
        p = np.load(E / f"outputs/aligned/{obj}__{rec}.npz")["points"].astype(np.float64)
        inside = np.all(np.abs(p) <= BOX_MARGIN * bounds, axis=1)
        clouds[rec] = voxel_unique(p[inside], bounds)
    counts = {"slam": write(obj, "slam", base, slam)}
    for rec in rows:
        counts[rec] = write(obj, rec, base, subsample(clouds[rec], CAP))
    counts["best_n"] = write(obj, "best_n", base, subsample(clouds[best], len(slam)))
    counts["union"] = write(obj, "union", base, np.concatenate([slam, subsample(clouds[best], CAP)]))
    rec_info = {rec: {"raw": rows[rec]["points"], "in_box_voxels": int(len(clouds[rec]))} for rec in rows}
    manifest.append({"object": obj, "best": best, "counts": counts, "clouds": rec_info})
    print(f"{obj:26s} best={best:3s}  " + "  ".join(f"{k}={v}" for k, v in counts.items()) +
          "   voxels: " + " ".join(f"{k}={v['in_box_voxels']}" for k, v in rec_info.items()))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
