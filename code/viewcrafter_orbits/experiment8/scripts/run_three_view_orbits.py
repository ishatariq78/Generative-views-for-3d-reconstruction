#!/usr/bin/env python3
"""360-degree orbits from three views, at the elevation those views were taken from.

Three views is ViewCrafter's designed operating point (its sparse examples ship
2-3, and the code special-cases `len(images) == 2`). This runs the same chosen-path
machinery as experiment 7 -- multi-view DUSt3R cloud, trajectory of our choosing --
but with three views instead of eight.

The one substantive change: **elevation is measured, not guessed.** Every earlier
experiment passed a per-object elevation read off the reference image by eye.
Here it is the mean elevation of the three input cameras above the object's own
ground plane, so the turn happens on the level the object was actually
photographed from.
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

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment8")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment7/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
from run_multiview_orbits import orbit_from_multiview       # noqa: E402
import objects as C                                         # noqa: E402

MEASURED = json.loads((PROJ / "inputs" / "elevations.json").read_text())


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
    ap.add_argument("--trajs", nargs="*", default=["orbit360", "orbit45"])
    # The measured-elevation run changes two things at once against the one-view
    # baseline: three views instead of one, and an elevation taken from the cameras
    # rather than guessed. This flag runs the same three views at the old guessed
    # elevation, so the two can be told apart.
    ap.add_argument("--guessed_elevation", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    ELEV = dict(C.load()) if args.guessed_elevation else MEASURED
    names = [n for n in C.names(args.only) if (PROJ / "inputs" / n).is_dir()]
    plan = [(n, t) for n in names for t in args.trajs]
    if not plan:
        raise SystemExit("nothing to run")

    parser = get_parser()
    opts = parser.parse_args(base_args(PROJ / "inputs" / plan[0][0], plan[0][1],
                                       ELEV[plan[0][0]], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{plan[0][0]}__{plan[0][1]}{args.suffix}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter for {len(plan)} runs ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log, last = [], None
    for name, traj in plan:
        exp = f"{name}__{traj}{args.suffix}"
        save_dir = PROJ / "outputs" / exp
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(PROJ / "inputs" / name, traj, ELEV[name], args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o

        t = time.time()
        kind = "guessed" if args.guessed_elevation else "measured"
        print(f"\n=== {exp}  (3 views, {kind} elevation {ELEV[name]}°) ===", flush=True)
        if last != name:
            torch.cuda.empty_cache()
            pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(PROJ / "inputs" / name))
            pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
            last = name
        seed_everything(o.seed)
        orbit_from_multiview(pvd, o.traj_txt)
        dt = time.time() - t
        print(f"=== {exp} done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": exp, "object": name, "trajectory": traj, "views": 3,
                    "elevation": ELEV[name], "elevation_source": kind, "seed": args.seed,
                    "seconds": round(dt, 1), "output_dir": str(save_dir)})
        rec = PROJ / "outputs" / "runs.json"
        prev = json.loads(rec.read_text()) if rec.exists() else []
        merged = {r["experiment"]: r for r in prev}
        merged.update({r["experiment"]: r for r in log})
        rec.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["experiment"]), indent=2))

    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
