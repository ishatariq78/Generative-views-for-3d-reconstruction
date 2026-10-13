#!/usr/bin/env python3
"""Experiment 5's six camera paths, re-run from a three-view cloud.

Experiment 5 asked whether the *route* around the object matters, and answered no:
six full-turn trajectories all landed between 0.23 and 0.36 worst coverage, a
spread smaller than the gap to simply not going all the way round. But that was
asked of a cloud built from **one** photograph, where roughly 70% of the far side
was never observed. With that much missing, no route can differ much from any
other -- they are all flying over the same hole.

Experiment 8 changed the geometry underneath: three views, and an orbit plane
measured from the input cameras rather than guessed. That is the strongest
configuration found so far. This asks experiment 5's question again on top of it.
If the path is genuinely not the lever, the six should stay bunched. If experiment
5's null was really a floor effect -- everything equally bad because everything was
equally unobserved -- the paths should start to separate now.

Held fixed against experiment 8: the same five objects, the same three views per
object, the same measured elevation, the same seed, the same 25-frame / 50-step
budget, the same masked multi-view cloud and the middle-view anchor. The
trajectory files are copied byte-for-byte from experiment 5.

`level` is that copy of experiment 5's baseline, and its trajectory file is
byte-identical to experiment 8's `orbit360`. So the level run is also a
reproducibility check: it should land on experiment 8's `v3_360` numbers.

30 runs (5 objects x 6 paths) sharing one loaded model, ~75 min. DUSt3R runs once
per object, not once per run.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

VC = Path("/home/isha.tariq/view_crafter/ViewCrafter")
sys.path.insert(0, str(VC))
os.chdir(VC)

import torch                                                # noqa: E402
from configs.infer_config import get_parser                 # noqa: E402
from pytorch_lightning import seed_everything               # noqa: E402
from viewcrafter import ViewCrafter                         # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment9")
VIEWS = Path("/home/isha.tariq/viewcrafter_orbits/experiment8/inputs")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment7/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
from run_multiview_orbits import orbit_from_multiview       # noqa: E402
import objects as C                                         # noqa: E402

# Measured from the three input cameras in experiment 8 -- the elevation the
# object was actually photographed from. Every path's theta is a delta on top of
# this, so `level` turns on the capture plane and `high` rides 25 degrees above it.
MEASURED = json.loads((VIEWS / "elevations.json").read_text())
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]


def base_args(image_dir, traj, elevation, seed):
    return [
        "--image_dir", str(image_dir),
        "--traj_txt", str(PROJ / "trajs" / f"{traj}.txt"),
        "--mode", "single_view_txt",
        "--out_dir", str(PROJ / "outputs"),
        "--center_scale", "1.",
        "--elevation", str(elevation),
        "--seed", str(seed),
        "--bg_trd", "0.2",
        "--dpt_trd", "0.9",
        "--ckpt_path", "./checkpoints/model.ckpt",
        "--config", "configs/inference_pvd_1024.yaml",
        "--model_path", "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "--ddim_steps", "50",
        "--video_length", "25",
        "--device", "cuda:0",
        "--height", "576", "--width", "1024",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--paths", nargs="*", default=PATHS)
    args = ap.parse_args()

    names = [n for n in C.names(args.only) if (VIEWS / n).is_dir()]
    # Object-major so DUSt3R's cloud is built once and reused across the six paths.
    plan = [(n, p) for n in names for p in args.paths]
    if not plan:
        raise SystemExit("nothing to run")

    parser = get_parser()
    opts = parser.parse_args(base_args(VIEWS / plan[0][0], plan[0][1],
                                       MEASURED[plan[0][0]], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{plan[0][0]}__{plan[0][1]}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter for {len(plan)} runs ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log, last = [], None
    for name, path in plan:
        exp = f"{name}__{path}"
        save_dir = PROJ / "outputs" / exp
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(VIEWS / name, path, MEASURED[name], args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o

        t = time.time()
        print(f"\n=== {exp}  (3 views, measured elevation {MEASURED[name]}°) ===", flush=True)
        if last != name:
            torch.cuda.empty_cache()
            pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(VIEWS / name))
            pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
            last = name
        seed_everything(o.seed)
        orbit_from_multiview(pvd, o.traj_txt)
        dt = time.time() - t
        print(f"=== {exp} done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": exp, "object": name, "path": path, "views": 3,
                    "elevation": MEASURED[name], "elevation_source": "measured",
                    "seed": args.seed, "seconds": round(dt, 1),
                    "output_dir": str(save_dir)})
        # Merged, not overwritten: a partial re-run must not erase earlier records.
        rec = PROJ / "outputs" / "runs.json"
        prev = json.loads(rec.read_text()) if rec.exists() else []
        merged = {r["experiment"]: r for r in prev}
        merged.update({r["experiment"]: r for r in log})
        rec.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["experiment"]), indent=2))

    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
