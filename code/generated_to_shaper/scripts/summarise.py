#!/usr/bin/env python3
"""Collect the ShapeR repeats into one table: per object x condition, mean +- std over repeats.

    python scripts/summarise.py     -> outputs/summary.json, outputs/summary.md (printed too)
"""
import glob, json
from pathlib import Path
import numpy as np

E = Path("/home/isha.tariq/seva_to_shaper")
CONDS = ["slam", "pi3", "kv", "pi3n", "kvn", "best_n", "union"]
bench = {}
for f in glob.glob("/home/isha.tariq/shapeR_testing/outputs/eval_balance/*.json"):
    if "summary" in f: continue
    for r in json.load(open(f))["per_sample"]: bench[r["name"]] = r["metrics"]

runs = {}
for f in sorted(glob.glob(str(E / "outputs/shaper/rep*/eval_results*.json"))):
    for r in json.load(open(f))["per_sample"]:
        obj, cond = r["name"].rsplit("__", 1)
        runs.setdefault((obj, cond), []).append(r["metrics"])
objs = sorted({o for o, _ in runs}); reps = max(len(v) for v in runs.values())
pk = {m["object"]: m for m in json.loads((E / "outputs/pkls/manifest.json").read_text())}
al = {(r["object"], r["reconstructor"]): r for r in json.loads((E / "outputs/aligned/summary.json").read_text())}


def agg(ms, k):
    v = np.array([m[k] for m in ms]); return float(v.mean()), float(v.std())


table = {}
lines = [f"# ShapeR with SEVA-orbit points instead of SLAM points ({reps} repeats per cell)\n",
         "CD x10^2 lower is better; NC / F1 higher. `bench` = the 178-object benchmark run (one draw).\n"]
for obj in objs:
    best = pk[obj]["best"]; n = pk[obj]["counts"]
    lines.append(f"\n## {obj}   (best cloud by SLAM reach: {best}; points slam={n['slam']} pi3={n.get('pi3','-')} kv={n.get('kv','-')})\n")
    lines.append("| condition | CD x10^2 | NC | F1 | n |\n|---|---:|---:|---:|---:|")
    b = bench.get(obj)
    if b: lines.append(f"| bench (slam) | {b['chamfer']*100:.2f} | {b['normals']:.3f} | {b['f-score']:.3f} | 1 |")
    table[obj] = {"best": best, "counts": n, "bench": b, "conditions": {}}
    for cond in CONDS:
        ms = runs.get((obj, cond))
        if not ms: continue
        cd, cds = agg(ms, "chamfer"); nc, ncs = agg(ms, "normals"); f1, f1s = agg(ms, "f-score")
        table[obj]["conditions"][cond] = {"chamfer_x100": [cd * 100, cds * 100], "normals": [nc, ncs], "f1": [f1, f1s], "reps": len(ms)}
        lines.append(f"| {cond} | {cd*100:.2f} ± {cds*100:.2f} | {nc:.3f} ± {ncs:.3f} | {f1:.3f} ± {f1s:.3f} | {len(ms)} |")
    for rec in ("pi3", "kv"):
        if (obj, rec) in al:
            f = al[(obj, rec)]["final"]
            lines.append(f"\n{rec}: SLAM reached {f['slam_reached']:.3f}, cloud->SLAM {f['cloud_to_slam']:.2f}%, cloud->mesh {f['cloud_to_mesh']:.2f}%, mesh->cloud {f['mesh_to_cloud']:.2f}%, {al[(obj, rec)]['refine']}")
lines.append("\n## Mean over objects\n\n| condition | CD x10^2 | NC | F1 | objects |\n|---|---:|---:|---:|---:|")
means = {}
for cond in CONDS:
    per = [table[o]["conditions"][cond] for o in objs if cond in table[o]["conditions"]]
    if not per: continue
    cd = np.mean([p["chamfer_x100"][0] for p in per]); nc = np.mean([p["normals"][0] for p in per]); f1 = np.mean([p["f1"][0] for p in per])
    means[cond] = {"chamfer_x100": float(cd), "normals": float(nc), "f1": float(f1), "objects": len(per)}
    lines.append(f"| {cond} | {cd:.2f} | {nc:.3f} | {f1:.3f} | {len(per)} |")
# paired: how often does each condition beat slam on the same object (mean over repeats)
lines.append("\n## Paired against `slam` (same object, mean over repeats)\n\n| condition | CD better | NC better | F1 better |\n|---|---:|---:|---:|")
for cond in CONDS[1:]:
    w = {"cd": 0, "nc": 0, "f1": 0}; n = 0
    for o in objs:
        c = table[o]["conditions"]
        if cond not in c or "slam" not in c: continue
        n += 1; w["cd"] += c[cond]["chamfer_x100"][0] < c["slam"]["chamfer_x100"][0]
        w["nc"] += c[cond]["normals"][0] > c["slam"]["normals"][0]; w["f1"] += c[cond]["f1"][0] > c["slam"]["f1"][0]
    lines.append(f"| {cond} | {w['cd']}/{n} | {w['nc']}/{n} | {w['f1']}/{n} |")
md = "\n".join(lines); print(md)
(E / "outputs/summary.md").write_text(md + "\n")
(E / "outputs/summary.json").write_text(json.dumps({"reps": reps, "per_object": table, "means": means}, indent=2))
