#!/usr/bin/env python3
"""Orbit each object six different ways, changing only the camera path.

Experiment 2 established one full-turn trajectory: azimuth 0->360 at a fixed
elevation. That is one choice out of many, and it was never tested against the
alternatives. This holds everything else still -- same three objects, the same
experiment 2 cutout reference images, same seed, same 25-frame / 50-step budget,
same assumed reference elevation -- and varies only how the camera is carried
around the object.

18 runs (3 objects x 6 paths) sharing one loaded model. Output goes to
outputs/<object>__<path>/ with the same four files experiments 1 and 2 produced,
so the measurements carry straight over.

The reference images come from experiment 2 rather than experiment 1 on purpose:
experiment 2 showed the object only survives a full turn once the scene is cut
out of the frame, so testing camera paths on the scene-context images would be
measuring the wrong failure.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

VC = Path("/home/isha.tariq/view_crafter/ViewCrafter")
sys.path.insert(0, str(VC))
os.chdir(VC)  # ViewCrafter resolves its config and checkpoint paths relative to cwd

from configs.infer_config import get_parser          # noqa: E402
from pytorch_lightning import seed_everything        # noqa: E402
from viewcrafter import ViewCrafter                  # noqa: E402
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as OBJECTS_CFG  # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment5")
EXP2_INPUTS = Path("/home/isha.tariq/viewcrafter_orbits/experiment2/inputs")

# Elevations are experiment 2's, unchanged. This is ViewCrafter's estimate of how
# far above the object the *reference camera* sits -- a separate knob from the
# trajectory, and holding it fixed is what keeps the path the only variable.
OBJECTS = OBJECTS_CFG.load()
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]


def base_args(image, traj, elevation, seed):
    return [
        "--image_dir", str(image),
        "--traj_txt", str(PROJ / "trajs" / f"{traj}.txt"),
        "--mode", "single_view_txt",
        "--out_dir", str(PROJ / "outputs"),
        "--center_scale", "1.",
        "--elevation", str(elevation),
        "--seed", str(seed),
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

    objs = [(n, e) for n, e in OBJECTS if args.only is None or n in args.only]
    plan = [(n, e, p) for n, e in objs for p in args.paths]
    if not plan:
        raise SystemExit("nothing to run")

    parser = get_parser()
    first = EXP2_INPUTS / f"{plan[0][0]}__cutout.png"
    opts = parser.parse_args(base_args(first, plan[0][2], plan[0][1], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{plan[0][0]}__{plan[0][2]}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter for {len(plan)} runs (loads ~12 GB) ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log = []
    for name, elevation, path in plan:
        image = EXP2_INPUTS / f"{name}__cutout.png"
        if not image.exists():
            print(f"!!! missing {image}, skipping", flush=True)
            continue
        exp = f"{name}__{path}"
        save_dir = PROJ / "outputs" / exp
        save_dir.mkdir(parents=True, exist_ok=True)

        o = parser.parse_args(base_args(image, path, elevation, args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o

        t = time.time()
        print(f"\n=== {exp}  (elevation {elevation}, seed {args.seed}) ===", flush=True)
        pvd.images, pvd.img_ori = pvd.load_initial_images(image_dir=str(image))
        pvd.run_dust3r(input_images=pvd.images)
        seed_everything(o.seed)
        pvd.nvs_single_view()
        dt = time.time() - t
        print(f"=== {exp} done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": exp, "object": name, "path": path,
                    "elevation": elevation, "seed": args.seed,
                    "seconds": round(dt, 1), "output_dir": str(save_dir)})

    (PROJ / "outputs" / "runs.json").write_text(json.dumps(log, indent=2))
    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
