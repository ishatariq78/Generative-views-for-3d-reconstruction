#!/usr/bin/env python3
"""Compare the selection conditions, aggregating at the object level.

    ~/.conda/envs/shaper/bin/python scripts/sel_eval.py --run sel/pilot

The primary comparison is D (reliability) vs B (uniform); D vs C says whether informed
selection beats random, D vs E whether the reliability signals add anything beyond viewpoint
spread. A is the real-only reference and is NOT budget-matched -- it carries 24 real frames
where B-E carry 15 real + 9 generated -- so it is reported, not treated as a control.

Seed-level differences are averaged WITHIN each object before any aggregate statistic, so
the unit of analysis is the object and the three seeds do not inflate the sample size. The
same audit is applied to the previously reported figure, which pooled 465 seed-level
differences as if independent.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

CONDS = ["A", "B", "C", "D", "E"]
NAMES = {"A": "real only (24 real)", "B": "uniform", "C": "random", "D": "reliability",
         "E": "diversity only"}


def load(run):
    r = defaultdict(dict)
    for f in glob.glob(f"{run}/s*/per_object/*.json"):
        stem = os.path.basename(f)[:-5]
        if "__" not in stem:
            continue
        obj, c = stem.rsplit("__", 1)
        if c not in CONDS:
            continue
        x = json.load(open(f))
        r[(obj, c)][x["seed"]] = x["metrics"]
    return r


def obj_mean(r, obj, c, key):
    m = r.get((obj, c), {})
    return float(np.mean([v[key] for v in m.values()])) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sel/pilot")
    ap.add_argument("--ref", default=None, help="optional: audit a previous run's pooling")
    a = ap.parse_args()
    r = load(a.run)
    objs = sorted({o for o, _ in r})
    full = [o for o in objs if all((o, c) in r and len(r[(o, c)]) == 3 for c in CONDS)]
    print(f"{len(objs)} objects present, {len(full)} complete across all 5 conditions x 3 seeds\n")
    if not full:
        return

    print(f"{'condition':22s}{'CD':>8s}{'F1':>8s}")
    for c in CONDS:
        cd = np.mean([obj_mean(r, o, c, "chamfer") for o in full]) * 100
        f1 = np.mean([obj_mean(r, o, c, "f-score") for o in full])
        print(f"  {NAMES[c]:20s}{cd:8.3f}{f1:8.3f}")

    def paired(x, y):
        d = np.array([obj_mean(r, o, x, "chamfer") - obj_mean(r, o, y, "chamfer") for o in full]) * 100
        df = np.array([obj_mean(r, o, x, "f-score") - obj_mean(r, o, y, "f-score") for o in full])
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d, df, se

    print(f"\n{'comparison':18s}{'dCD mean':>10s}{'median':>9s}{'se':>8s}{'t':>7s}"
          f"{'better':>8s}{'worse':>7s}{'dF1':>8s}")
    for x, y in [("D", "B"), ("D", "C"), ("D", "E"), ("B", "A"), ("D", "A"), ("C", "B"), ("E", "B")]:
        d, df, se = paired(x, y)
        t = d.mean() / se if se > 0 else float("nan")
        print(f"  {x} vs {y:14s}{d.mean():+10.3f}{np.median(d):+9.3f}{se:8.3f}{t:7.2f}"
              f"{int((d < -1e-9).sum()):8d}{int((d > 1e-9).sum()):7d}{df.mean():+8.3f}")

    d, _, _ = paired("D", "B")
    worst = sorted(zip(full, d), key=lambda z: -z[1])[:5]
    print("\nlargest regressions, D vs B:")
    for o, v in worst:
        print(f"  {o[:40]:40s}{v:+7.2f}")

    rows = [{"object": o, **{c: {"cd": obj_mean(r, o, c, "chamfer") * 100,
                                "f1": obj_mean(r, o, c, "f-score")} for c in CONDS}} for o in full]
    json.dump({"objects": rows, "n_objects": len(full)},
              open(f"{a.run}/summary.json", "w"), indent=1)

    if a.ref:
        rr = {}
        for f in glob.glob(f"{a.ref}/s*/per_object/*.json"):
            stem = os.path.basename(f)[:-5]
            c = "views" if stem.endswith("__views") else ("slam" if stem.endswith("__slam") else None)
            if not c:
                continue
            o = stem.rsplit("__", 1)[0]
            rr.setdefault((o, c), {})[json.load(open(f))["seed"]] = json.load(open(f))["metrics"]
        objs2 = sorted({o for o, c in rr if (o, "slam") in rr and (o, "views") in rr
                        and len(rr[(o, "slam")]) == 3 and len(rr[(o, "views")]) == 3})
        pooled = np.array([(rr[(o, "views")][s]["chamfer"] - rr[(o, "slam")][s]["chamfer"]) * 100
                           for o in objs2 for s in (0, 1, 2)])
        byobj = np.array([np.mean([(rr[(o, "views")][s]["chamfer"] - rr[(o, "slam")][s]["chamfer"]) * 100
                                   for s in (0, 1, 2)]) for o in objs2])
        print(f"\nAUDIT of the previously reported figure ({len(objs2)} objects):")
        print(f"  pooling 465 seed-level diffs : {pooled.mean():+.3f} se {pooled.std(ddof=1)/np.sqrt(len(pooled)):.3f} "
              f"t {pooled.mean()/(pooled.std(ddof=1)/np.sqrt(len(pooled))):.2f}")
        print(f"  object-level (correct)       : {byobj.mean():+.3f} se {byobj.std(ddof=1)/np.sqrt(len(byobj)):.3f} "
              f"t {byobj.mean()/(byobj.std(ddof=1)/np.sqrt(len(byobj))):.2f}")


if __name__ == "__main__":
    main()
