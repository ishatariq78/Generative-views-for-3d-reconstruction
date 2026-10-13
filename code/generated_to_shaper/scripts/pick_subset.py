#!/usr/bin/env python3
"""Choose the objects where a generated orbit is *mechanistically* expected to help.

    ~/.conda/envs/shaper/bin/python scripts/pick_subset.py --n 40

The selection must not look at ShapeR's own error, or picking the worst objects would
flatter any intervention through regression to the mean. It uses only properties of the
capture, all readable from the pkl before anything is run:

  coverage   fraction of 24 azimuth bins around the object that any camera occupies --
             how one-sided the walk-through was
  size       the object's bounding-box diagonal; the paper's own failure mode is large
             furniture, which a casual capture never gets all of
  seen       mean fraction of the object's points visible per view -- low means the object
             is usually clipped or blocked

Ranked by one-sidedness first, with a size floor so the set is furniture rather than mugs.
"""
import argparse, glob, json, os, pickle
import numpy as np

DATA = "/home/isha.tariq/shapeR_testing/data"


def stats(p):
    d = pickle.load(open(p, "rb"))
    v = np.asarray(d["mesh_vertices"], np.float64)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    centre = (v.max(0) + v.min(0)) / 2
    T = d["Ts_rgbCamera_model"].numpy() if "Ts_rgbCamera_model" in d else d["Ts_camera_model"].numpy()
    pos = np.stack([-T[i, :3, :3].T @ T[i, :3, 3] for i in range(len(T))])
    az = np.degrees(np.arctan2(*(pos - centre)[:, [1, 0]].T)) % 360
    cov = len(set((az // 15).astype(int))) / 24.0
    vis = d.get("rgb_visible_points_model", d.get("visible_points_model", []))
    n_pts = max(len(d["points_model"].numpy()), 1)
    seen = float(np.mean([len(x) for x in vis]) / n_pts) if len(vis) else 0.0
    return {"name": os.path.basename(p)[:-4], "coverage": round(cov, 3), "diag": round(diag, 2),
            "seen": round(seen, 3), "n_points": int(n_pts), "n_views": int(len(T))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min_diag", type=float, default=0.9, help="metres; furniture, not tabletop objects")
    a = ap.parse_args()
    rows = [stats(p) for p in sorted(glob.glob(f"{DATA}/*.pkl"))]
    big = [r for r in rows if r["diag"] >= a.min_diag]
    big.sort(key=lambda r: (r["coverage"], -r["diag"]))
    pick = big[:a.n]
    json.dump([{"name": r["name"]} for r in pick], open("objects_subset.json", "w"), indent=1)
    json.dump(rows, open("outputs/capture_stats.json", "w"), indent=1)
    print(f"{len(rows)} objects, {len(big)} at diag >= {a.min_diag} m, picked {len(pick)}")
    print(f"picked: coverage {np.mean([r['coverage'] for r in pick]):.2f} vs rest {np.mean([r['coverage'] for r in big[a.n:]]):.2f}"
          f" | diag {np.mean([r['diag'] for r in pick]):.2f} m")
    for r in pick[:12]:
        print(f"  {r['name']:28s} cov {r['coverage']:.2f}  diag {r['diag']:.2f}  seen {r['seen']:.3f}  {r['n_views']} views")
