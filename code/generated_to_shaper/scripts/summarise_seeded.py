#!/usr/bin/env python3
"""The paired comparison: per object x condition, and per condition the paired difference
against `slam` on the *same object and the same seed*.

    python scripts/summarise_seeded.py   -> outputs/summary_seeded.{md,json}

Because the noise is identical across conditions, the per-seed difference is a paired
observation, so the spread reported here is the spread of the *effect*, not of the sampler.
"""
import glob, json
from pathlib import Path
import numpy as np

E = Path("/home/isha.tariq/seva_to_shaper")
CONDS = ["slam", "pi3", "kv", "union", "gap1", "gap4", "views", "vonly", "vg80", "v80", "vgap"]
runs = {}
for f in glob.glob(str(E / "outputs/seeded/s*/per_object/*.json")):
    r = json.load(open(f)); obj, cond = r["name"].rsplit("__", 1)
    runs.setdefault((obj, cond), {})[r["seed"]] = r["metrics"]
objs = sorted({o for o, _ in runs})
seeds = sorted({s for v in runs.values() for s in v})

L = [f"# Paired comparison, seeds {seeds} (same noise per object across conditions)\n",
     "CD x10^2 lower is better, F1 higher. Mean over seeds.\n",
     "| object | n | " + " | ".join(CONDS) + " |",
     "|---|---:|" + "---|" * len(CONDS)]
table = {}
for o in objs:
    cells, row = [], {}
    n = None
    for c in CONDS:
        m = runs.get((o, c))
        if not m: cells.append("-"); continue
        cd = np.mean([x["chamfer"] for x in m.values()]) * 100
        f1 = np.mean([x["f-score"] for x in m.values()])
        nc = np.mean([x["normals"] for x in m.values()])
        row[c] = {"chamfer_x100": float(cd), "f1": float(f1), "normals": float(nc), "seeds": len(m)}
        cells.append(f"{cd:.2f} / {f1:.3f}")
    table[o] = row
    L.append(f"| {o} | {n or ''} | " + " | ".join(cells) + " |")

L += ["\n## Paired difference against `slam` (negative CD = better)\n",
      "| condition | dCD mean | dCD std | wins | dF1 mean | F1 wins | n pairs |", "|---|---:|---:|---:|---:|---:|---:|"]
diffs = {}
for c in CONDS[1:]:
    dcd, df1 = [], []
    for o in objs:
        a, b = runs.get((o, c)), runs.get((o, "slam"))
        if not a or not b: continue
        for s in set(a) & set(b):
            dcd.append((a[s]["chamfer"] - b[s]["chamfer"]) * 100)
            df1.append(a[s]["f-score"] - b[s]["f-score"])
    if not dcd: continue
    dcd, df1 = np.array(dcd), np.array(df1)
    diffs[c] = {"dCD_mean": float(dcd.mean()), "dCD_std": float(dcd.std()), "cd_wins": int((dcd < 0).sum()),
                "dF1_mean": float(df1.mean()), "f1_wins": int((df1 > 0).sum()), "pairs": len(dcd)}
    L.append(f"| {c} | {dcd.mean():+.3f} | {dcd.std():.3f} | {(dcd<0).sum()}/{len(dcd)} | "
             f"{df1.mean():+.3f} | {(df1>0).sum()}/{len(df1)} | {len(dcd)} |")

L += ["\n## Per object, paired dCD against `slam` (mean over seeds)\n",
      "| object | " + " | ".join(CONDS[1:]) + " |", "|---|" + "---:|" * len(CONDS[1:])]
for o in objs:
    cells = []
    for c in CONDS[1:]:
        a, b = runs.get((o, c)), runs.get((o, "slam"))
        if not a or not b: cells.append("-"); continue
        d = np.mean([(a[s]["chamfer"] - b[s]["chamfer"]) * 100 for s in set(a) & set(b)])
        cells.append(f"**{d:+.2f}**" if d < 0 else f"{d:+.2f}")
    L.append(f"| {o} | " + " | ".join(cells) + " |")

md = "\n".join(L); print(md)
(E / "outputs/summary_seeded.md").write_text(md + "\n")
(E / "outputs/summary_seeded.json").write_text(json.dumps({"seeds": seeds, "per_object": table, "paired": diffs}, indent=2))
