#!/usr/bin/env python3
"""Build pkls with an EXACT, controlled view set: N_REAL real + N_GEN generated, no more.

    ~/.conda/envs/shaper/bin/python scripts/sel_build.py --object <obj> --cond B --seed 0

ShapeR picks its own views by k-means over camera centres, so in the existing experiment the
real/generated split was whatever fell out -- 1 of 24 on one object, 9 of 24 on another. A
selection study cannot sit on that. Supplying exactly `num_views` images makes the selector
keep all of them (it clusters with min(k, n) clusters), so the view set is frozen and the
only thing that varies between conditions is WHICH generated frames are in it.

Conditions (all share the same real frames, the same cached orbit, the same totals):
  A  real only                      -- N_REAL + N_GEN real frames, no generated
  B  uniform                        -- N_GEN generated frames spaced evenly round the orbit
  C  random                         -- N_GEN drawn at random (seed logged)
  D  reliability                    -- N_GEN chosen by scripts/sel_score.py
  E  diversity-only                 -- N_GEN chosen for viewpoint spread alone (control for D)

Real frames are chosen once per object by ShapeR's own ranking and then reused by every
condition, so no condition gets a better set of real views than another.
"""
import argparse, io, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch

E = Path("/home/isha.tariq/seva_to_shaper")
DATA = Path("/home/isha.tariq/shapeR_testing/data")
REPO = "/home/isha.tariq/shapeR_testing/ShapeR"
sys.path.insert(0, str(E / "scripts"))


def real_frame_order(obj, n_want):
    """The real frames ShapeR itself would rank highest, computed once and cached."""
    cache = E / f"sel/real_order/{obj}.json"
    if cache.exists():
        return json.loads(cache.read_text())[:n_want]
    cwd = os.getcwd(); os.chdir(REPO); sys.path.insert(0, REPO)
    from dataset.image_processor import cluster_and_select_images
    d = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
    scale = 0.9 / float(np.max(d["bounds"].numpy()))
    sel = cluster_and_select_images(d, n_want, scale, False)
    order = [int(x[0]) for x in sorted(sel, key=lambda x: -x[5])]
    os.chdir(cwd)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(order))
    return order[:n_want]


def take(seq, idx):
    return [seq[i] for i in idx]


def build(obj, cond, n_real, n_gen, seed, out_dir):
    import make_pkls_views as M                      # the verified camera conversion
    base = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
    real_idx = real_frame_order(obj, n_real if cond != "A" else n_real + n_gen)
    views = [v for v in M.orbit_views(obj, 80) if v["n_dots"] >= M.DOT_MIN]
    if cond != "A" and len(views) < n_gen:
        return None, f"only {len(views)} usable generated frames"

    if cond == "A":
        pick = []
    elif cond == "B":
        pick = [round(i * (len(views) - 1) / (n_gen - 1)) for i in range(n_gen)]
    elif cond == "C":
        pick = sorted(np.random.default_rng(seed).choice(len(views), n_gen, replace=False).tolist())
    else:
        sc = json.loads((E / f"sel/scores/{obj}.json").read_text())
        key = {"D": "reliability_order", "E": "diversity_order"}[cond]
        pick = sorted(sc[key][:n_gen])
    gen = take(views, pick)

    d = dict(base)
    d["image_data"] = take(list(base["image_data"]), real_idx) + [M.encode(v["image"]) for v in gen]
    d["camera_params"] = torch.cat([base["camera_params"][real_idx]] +
                                   ([torch.from_numpy(np.stack([v["params"] for v in gen]))] if gen else []))
    d["Ts_camera_model"] = torch.cat([base["Ts_camera_model"][real_idx]] +
                                     ([torch.from_numpy(np.stack([v["T"] for v in gen]))] if gen else []))
    d["visible_points_model"] = take(list(base["visible_points_model"]), real_idx) + \
                                [torch.from_numpy(v["pts"]) for v in gen]
    d["object_point_projections"] = take(list(base["object_point_projections"]), real_idx) + \
                                    [torch.from_numpy(v["uv"]) for v in gen]
    d["s2s_condition"] = cond
    out = out_dir / f"{obj}__{cond}.pkl"
    with open(out, "wb") as f:
        pickle.dump(d, f)
    man = {"object": obj, "condition": cond, "seed": seed if cond == "C" else None,
           "n_real": len(real_idx), "n_generated": len(gen), "total": len(d["image_data"]),
           "real_frame_indices": [int(i) for i in real_idx],
           "generated_orbit_frames": [int(views[i]["frame"]) for i in pick]}
    return man, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", required=True)
    ap.add_argument("--cond", required=True, choices=list("ABCDE"))
    ap.add_argument("--n_real", type=int, default=15)
    ap.add_argument("--n_gen", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=E / "sel/pkls")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    man, err = build(a.object, a.cond, a.n_real, a.n_gen, a.seed, a.out_dir)
    if err:
        print(f"!! {a.object} {a.cond}: {err}"); sys.exit(1)
    md = E / "sel/manifests"; md.mkdir(parents=True, exist_ok=True)
    (md / f"{a.object}__{a.cond}.json").write_text(json.dumps(man, indent=2))
    print(f"{a.object} {a.cond}: {man['n_real']} real + {man['n_generated']} generated "
          f"= {man['total']} views")
