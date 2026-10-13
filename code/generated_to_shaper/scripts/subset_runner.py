#!/usr/bin/env python3
"""Build views pkls for the subset, score them against the baseline, delete, repeat.

    python scripts/subset_runner.py --shard 0 --n_shards 2 --num_images 24

40 objects x ~120 MB of pkl will not fit on a /home at 98%, so each object is built,
scored in both conditions at the chosen view budget, and its pkl removed before the next.
Seeds are reset per object by the seeded runner, so `slam` and `views` see identical noise
and the difference is the imagery alone. Results land one JSON per (object, condition, seed)
in outputs_sub/seeded/s<seed>/per_object, the same layout the 6-object study uses.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

E = Path("/home/isha.tariq/seva_to_shaper")
PY = sys.executable
DATA = Path("/home/isha.tariq/shapeR_testing/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--num_images", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--objects", default="objects_subset_built.json")
    a = ap.parse_args()

    names = [r["name"] for r in json.loads((E / a.objects).read_text())]
    names = [n for i, n in enumerate(names) if i % a.n_shards == a.shard]
    pkl_dir = Path(os.environ["S2S_PKLS"])
    out_root = E / "outputs_sub/seeded"
    print(f"shard {a.shard}/{a.n_shards}: {len(names)} objects, budget {a.num_images}", flush=True)

    for j, obj in enumerate(names, 1):
        done = all((out_root / f"s{s}/per_object/{obj}__{c}.json").exists()
                   for s in a.seeds for c in ("slam", "views"))
        if done:
            print(f"[{j}/{len(names)}] {obj}: already scored", flush=True); continue
        t0 = time.time()
        r = subprocess.run([PY, str(E / "scripts/make_pkls_views.py"),
                            "--only", obj, "--conds", "views"], cwd=E,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   pkl stderr: {(r.stderr or '')[-200:].strip()}", flush=True)
        views_pkl = pkl_dir / f"{obj}__views.pkl"
        if r.returncode != 0 or not views_pkl.exists():
            print(f"[{j}/{len(names)}] {obj}: views pkl FAILED", flush=True); continue
        # The baseline is the benchmark pkl itself -- no copy, so nothing can drift.
        pkls = [str(DATA / f"{obj}.pkl"), str(views_pkl)]
        for s in a.seeds:
            subprocess.run([PY, str(E / "scripts/run_shaper_nviews.py"),
                            "--input_pkls", *pkls, "--out_dir", str(out_root / f"s{s}"),
                            "--seed", str(s), "--num_images", str(a.num_images), "--skip_done"],
                           cwd="/home/isha.tariq/shapeR_testing/exp_object_aware/scripts")
        # `<obj>.json` is the baseline's name (the benchmark pkl); keep it as <obj>__slam.json
        for s in a.seeds:
            src = out_root / f"s{s}/per_object/{obj}.json"
            dst = out_root / f"s{s}/per_object/{obj}__slam.json"
            if src.exists() and not dst.exists():
                src.rename(dst)
        views_pkl.unlink(missing_ok=True)
        print(f"[{j}/{len(names)}] {obj}: done in {time.time()-t0:.0f}s", flush=True)
    print("### shard finished", flush=True)


if __name__ == "__main__":
    sys.exit(main())
