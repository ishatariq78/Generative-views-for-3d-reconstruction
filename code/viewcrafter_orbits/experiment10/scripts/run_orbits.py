#!/usr/bin/env python3
"""Experiments 7, 8 and 9 again, on the corrected input order (wooden table only).

One loaded model, same seed / budget / checkpoint / anchor rule as the originals:
  mv360     8 views (inputs8), anchor = file 004, orbit360 at the object's guessed elevation   (exp 7)
  v3_360    3 views (inputs3), anchor = file 001, orbit360 at the measured elevation           (exp 8)
  v3_360g   3 views, orbit360 at the guessed elevation                                        (exp 8 control)
  p3_<path> 3 views, measured elevation, the six experiment-5 paths                           (exp 9)
p3_level is byte-identical to v3_360 in trajectory, so it is the reproducibility check.

    python scripts/run_orbits.py [--only ADT1292__woodentable] [--conds mv360 v3_360 ...]
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

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment10")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment7/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
from run_multiview_orbits import orbit_from_multiview       # noqa: E402
import objects as C                                         # noqa: E402

GUESSED = dict(C.load())
MEASURED = json.loads((PROJ / "inputs3" / "elevations.json").read_text())
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]
# condition -> (view set, trajectory, elevation table)
PLAN = {"mv360": ("inputs8", "orbit360", GUESSED),
        "v3_360": ("inputs3", "orbit360", MEASURED),
        "v3_360g": ("inputs3", "orbit360", GUESSED)}
PLAN.update({f"p3_{p}": ("inputs3", p, MEASURED) for p in PATHS})


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
    ap.add_argument("--only", nargs="*", default=["ADT1292__woodentable"])
    ap.add_argument("--conds", nargs="*", default=list(PLAN))
    args = ap.parse_args()

    runs = [(n, c) for n in C.names(args.only) for c in args.conds]
    # group by view set so DUSt3R runs once per (object, view set)
    runs.sort(key=lambda r: (r[0], PLAN[r[1]][0]))

    def spec(n, c):
        views, traj, elev = PLAN[c]
        return PROJ / views / n, traj, elev[n]

    parser = get_parser()
    d0, t0_, e0 = spec(*runs[0])
    opts = parser.parse_args(base_args(d0, t0_, e0, args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{runs[0][0]}__{runs[0][1]}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter for {len(runs)} runs ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log, last = [], None
    for name, cond in runs:
        views_dir, traj, elev = spec(name, cond)
        exp = f"{name}__{cond}"
        save_dir = PROJ / "outputs" / exp
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(views_dir, traj, elev, args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o
        t = time.time()
        n_views = len(list(views_dir.glob("*.png")))
        print(f"\n=== {exp}  ({n_views} views, {traj}, elevation {elev}°) ===", flush=True)
        if last != views_dir:
            torch.cuda.empty_cache()
            pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(views_dir))
            pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
            last = views_dir
        seed_everything(o.seed)
        orbit_from_multiview(pvd, o.traj_txt)
        dt = time.time() - t
        print(f"=== {exp} done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": exp, "object": name, "condition": cond, "views": n_views,
                    "view_set": views_dir.name, "anchor_index": n_views // 2,
                    "trajectory": traj, "elevation": elev, "seed": args.seed,
                    "seconds": round(dt, 1), "output_dir": str(save_dir)})
        rec = PROJ / "outputs" / "runs.json"
        prev = json.loads(rec.read_text()) if rec.exists() else []
        merged = {r["experiment"]: r for r in prev}
        merged.update({r["experiment"]: r for r in log})
        rec.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["experiment"]), indent=2))

    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
