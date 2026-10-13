#!/usr/bin/env python3
"""Run the selection conditions on a set of objects: build, score, infer, clean up.

    ~/.conda/envs/shaper/bin/python scripts/sel_run.py --objects_json sel/dev_objects.json

Per object the expensive camera conversion runs ONCE and every condition reuses it, so the
only difference between conditions is which generated frames are in the 24-view set. Each
object's pkls are deleted after it is scored -- five conditions x ~120 MB would not fit.
"""
import argparse, json, os, pickle, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

E = Path("/home/isha.tariq/seva_to_shaper")
DATA = Path("/home/isha.tariq/shapeR_testing/data")
sys.path.insert(0, str(E / "scripts"))
import sel_build as B
import make_pkls_views as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects_json", default="sel/dev_objects.json")
    ap.add_argument("--conds", nargs="*", default=["A", "B", "C", "D", "E"])
    ap.add_argument("--n_real", type=int, default=15)
    ap.add_argument("--n_gen", type=int, default=9)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--rand_seed", type=int, default=7)
    ap.add_argument("--out", default="sel/runs")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    a = ap.parse_args()

    names = [r["name"] for r in json.loads((E / a.objects_json).read_text())]
    names = [n for i, n in enumerate(names) if i % a.n_shards == a.shard]
    pkl_dir = E / "sel/pkls"; pkl_dir.mkdir(parents=True, exist_ok=True)
    out_root = E / a.out
    (E / "sel/manifests").mkdir(parents=True, exist_ok=True)
    print(f"shard {a.shard}/{a.n_shards}: {len(names)} objects x {len(a.conds)} conditions", flush=True)

    for j, obj in enumerate(names, 1):
        if all((out_root / f"s{s}/per_object/{obj}__{c}.json").exists()
               for s in a.seeds for c in a.conds):
            print(f"[{j}/{len(names)}] {obj}: already done", flush=True); continue
        t0 = time.time()
        try:
            views = [v for v in M.orbit_views(obj, 80) if v["n_dots"] >= M.DOT_MIN]
        except Exception as e:
            print(f"[{j}/{len(names)}] {obj}: orbit conversion failed: {e}", flush=True); continue
        if len(views) < a.n_gen:
            print(f"[{j}/{len(names)}] {obj}: only {len(views)} usable generated frames", flush=True); continue
        sc = None
        sp = E / f"sel/scores/{obj}.json"
        if sp.exists():
            sc = json.loads(sp.read_text())
        base = pickle.load(open(DATA / f"{obj}.pkl", "rb"))

        made = []
        for cond in a.conds:
            n_real = a.n_real + (a.n_gen if cond == "A" else 0)
            real_idx = B.real_frame_order(obj, n_real)
            if cond == "A":
                pick = []
            elif cond == "B":
                pick = [round(i * (len(views) - 1) / (a.n_gen - 1)) for i in range(a.n_gen)]
            elif cond == "C":
                pick = sorted(np.random.default_rng(a.rand_seed).choice(len(views), a.n_gen, replace=False).tolist())
            elif sc is None:
                print(f"   {cond}: no scores for {obj}, skipped", flush=True); continue
            else:
                key = "reliability_order" if cond == "D" else "diversity_order"
                pick = sorted(sc[key][:a.n_gen])
            gen = [views[i] for i in pick]
            d = dict(base)
            d["image_data"] = [base["image_data"][i] for i in real_idx] + [M.encode(v["image"]) for v in gen]
            d["camera_params"] = torch.cat([base["camera_params"][real_idx]] +
                ([torch.from_numpy(np.stack([v["params"] for v in gen]))] if gen else []))
            d["Ts_camera_model"] = torch.cat([base["Ts_camera_model"][real_idx]] +
                ([torch.from_numpy(np.stack([v["T"] for v in gen]))] if gen else []))
            d["visible_points_model"] = [base["visible_points_model"][i] for i in real_idx] + \
                [torch.from_numpy(v["pts"]) for v in gen]
            d["object_point_projections"] = [base["object_point_projections"][i] for i in real_idx] + \
                [torch.from_numpy(v["uv"]) for v in gen]
            p = pkl_dir / f"{obj}__{cond}.pkl"
            with open(p, "wb") as f:
                pickle.dump(d, f)
            made.append(str(p))
            (E / f"sel/manifests/{obj}__{cond}.json").write_text(json.dumps(
                {"object": obj, "condition": cond, "rand_seed": a.rand_seed if cond == "C" else None,
                 "n_real": len(real_idx), "n_generated": len(gen), "total": len(d["image_data"]),
                 "real_frame_indices": [int(i) for i in real_idx],
                 "generated_orbit_frames": [int(views[i]["frame"]) for i in pick]}, indent=2))
        if not made:
            continue
        for s in a.seeds:
            subprocess.run([sys.executable, str(E / "scripts/run_shaper_nviews.py"),
                            "--input_pkls", *made, "--out_dir", str(out_root / f"s{s}"),
                            "--seed", str(s), "--num_images", "24", "--skip_done"],
                           cwd="/home/isha.tariq/shapeR_testing/exp_object_aware/scripts")
        for p in made:
            Path(p).unlink(missing_ok=True)
        print(f"[{j}/{len(names)}] {obj}: {len(made)} conditions in {time.time()-t0:.0f}s", flush=True)
    print("### shard finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
