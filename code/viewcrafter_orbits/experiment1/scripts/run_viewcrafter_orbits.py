#!/usr/bin/env python3
"""Orbit each ShapeR object with ViewCrafter, loading the model once.

Six runs: three objects x two trajectories. Building a ViewCrafter costs ~12 GB of
checkpoint reads (9.8 GB video diffusion model + 2.2 GB DUSt3R), so the model is
constructed once and only the per-run pieces -- reference image, DUSt3R point
cloud, camera path -- are rebuilt each time. This mirrors the pattern already used
in ~/view_crafter/ViewCrafter/experiments.py.

Each run writes, into outputs/<object>__<traj>/:
    render0.mp4     the point cloud rendered along the orbit, before diffusion
    diffusion0.mp4  the final video -- the deliverable
    pcd0.ply        the DUSt3R point cloud the orbit was rendered from
    viz_traj.mp4    the camera path
Keeping render0 next to diffusion0 is the point of the comparison: it separates
"DUSt3R never recovered the geometry" from "the diffusion model could not fill in
what the render left empty".
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

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment1")

# elevation is ViewCrafter's estimate of how far *above* the object the reference
# camera sits; it defines the orbit's tilt plane. These are read off the reference
# images -- all three are head-height Aria captures looking down at furniture.
RUNS = [
    ("ADT1292__chair", 25.0),
    ("BNB1045__sofa",  25.0),
    ("BNB1723__bed",   30.0),
]
TRAJS = ["orbit360", "orbit45"]


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
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these object names")
    args = ap.parse_args()

    runs = [(n, e) for n, e in RUNS if args.only is None or n in args.only]
    first_img = PROJ / "inputs" / f"{runs[0][0]}.png"

    parser = get_parser()
    opts = parser.parse_args(base_args(first_img, TRAJS[0], runs[0][1], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{runs[0][0]}__{TRAJS[0]}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print("=== building ViewCrafter (loads ~12 GB of checkpoints) ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log = []
    for name, elevation in runs:
        image = PROJ / "inputs" / f"{name}.png"
        for traj in TRAJS:
            exp = f"{name}__{traj}"
            save_dir = PROJ / "outputs" / exp
            save_dir.mkdir(parents=True, exist_ok=True)

            o = parser.parse_args(base_args(image, traj, elevation, args.seed))
            o.save_dir = str(save_dir)
            pvd.opts = o

            t = time.time()
            print(f"\n=== {exp}  (elevation {elevation}, seed {args.seed}) ===", flush=True)
            # Re-run DUSt3R per reference image: the point cloud the orbit renders
            # from is derived from this image alone, not carried over.
            pvd.images, pvd.img_ori = pvd.load_initial_images(image_dir=str(image))
            pvd.run_dust3r(input_images=pvd.images)
            seed_everything(o.seed)
            pvd.nvs_single_view()
            dt = time.time() - t
            print(f"=== {exp} done in {dt:.0f}s -> {save_dir} ===", flush=True)
            log.append({"experiment": exp, "object": name, "trajectory": traj,
                        "elevation": elevation, "seed": args.seed,
                        "seconds": round(dt, 1), "output_dir": str(save_dir)})

    (PROJ / "outputs" / "runs.json").write_text(json.dumps(log, indent=2))
    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
