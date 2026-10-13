#!/usr/bin/env python3
"""Density-matched controls: each reconstructor's cloud at exactly the SLAM point count.

    ~/.conda/envs/shaper/bin/python scripts/make_pkls_density.py

`pi3`/`kv` hand ShapeR 4096 points where the benchmark gave it 300-7628, so a change could
be the points' *content* or merely their number and spread. `pi3_n` / `kv_n` are the same
clouds subsampled to the object's own SLAM count, which isolates content. Written with the
same code path as scripts/make_pkls.py (imported, not copied).
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / "make_pkls.py").read_text().split("\nsummary = ")[0]
ns = {}; exec(compile(src, "make_pkls.py", "exec"), ns)
voxel_unique, subsample, write, E, SHAPER_DATA, BOX_MARGIN = (
    ns["voxel_unique"], ns["subsample"], ns["write"], ns["E"], ns["SHAPER_DATA"], ns["BOX_MARGIN"])

rows = []
for obj in json.loads((E / "outputs/sets.json").read_text()):
    base = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    bounds = base["bounds"].numpy(); n_slam = len(base["points_model"].numpy())
    counts = {}
    for rec in ("pi3", "kv"):
        f = E / f"outputs/aligned/{obj}__{rec}.npz"
        if not f.exists(): continue
        p = np.load(f)["points"].astype(np.float64)
        p = p[np.all(np.abs(p) <= BOX_MARGIN * bounds, axis=1)]
        counts[f"{rec}_n"] = write(obj, f"{rec}n", base, subsample(voxel_unique(p, bounds), n_slam))
    rows.append({"object": obj, "n_slam": n_slam, "counts": counts})
    print(f"{obj:26s} " + "  ".join(f"{k}={v}" for k, v in counts.items()))
(E / "outputs/pkls/manifest_density.json").write_text(json.dumps(rows, indent=2))
