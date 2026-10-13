#!/usr/bin/env python3
"""Run ViewCrafter in sparse-view mode: many views of one object, not one.

Three things differ from every earlier run, and all three follow from the mode
rather than from a choice:

  * `--image_dir` is a **directory**. ViewCrafter switches loader on
    `os.path.isdir`, and requires the files to be named as integers because it
    sorts them numerically.
  * A **different checkpoint** -- `model_sparse.ckpt` -- which is what upstream's
    own `run_sparse.sh` uses for this mode.
  * **No trajectory file.** `generate_traj_interp` interpolates between the input
    views' own estimated poses, so the camera follows the path the wearer walked.
    A 360° orbit cannot be requested here; the views decide.

The mode renders one 25-frame clip per consecutive pair and concatenates them, so
16 views produce a 375-frame video. Output lands in outputs/<object>/ as
`diffusion.mp4`, `render.mp4` and `pcd.ply` -- note the names differ from the
single-view mode's `diffusion0.mp4`.
"""
import argparse
import json
import os
import sys
import time

import torch
from pathlib import Path

VC = Path("/home/isha.tariq/view_crafter/ViewCrafter")
sys.path.insert(0, str(VC))
os.chdir(VC)

from configs.infer_config import get_parser          # noqa: E402
from pytorch_lightning import seed_everything        # noqa: E402
from viewcrafter import ViewCrafter                  # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment6")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as C                                  # noqa: E402


def base_args(image_dir, seed):
    return [
        "--image_dir", str(image_dir),
        "--mode", "sparse_view_interp",
        "--out_dir", str(PROJ / "outputs"),
        "--bg_trd", "0.2",
        # Clip the farthest depths out of the fused cloud. On these inputs that is
        # the studio backdrop, which is exactly what we do not want rendered -- so
        # the memory saving and the intent point the same way.
        "--dpt_trd", "0.8",
        "--seed", str(seed),
        "--ckpt_path", "./checkpoints/model_sparse.ckpt",
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
    # 16 views exhausts a 40 GB A100 inside pytorch3d's point rasteriser: the fused
    # cloud from that many views, padded across a 375-frame trajectory, asks for a
    # single 9.5 GB allocation. Upstream's own sparse examples ship 2-3 views. 8 is
    # the compromise -- several times upstream, and it fits.
    ap.add_argument("--views_dir", default="inputs8",
                    help="which prepared view set to use (inputs = 16, inputs8 = 8)")
    args = ap.parse_args()

    root = PROJ / args.views_dir
    names = [n for n in C.names(args.only) if (root / n).is_dir()]
    if not names:
        raise SystemExit(f"no multi-view sets under {PROJ / 'inputs'}")

    parser = get_parser()
    opts = parser.parse_args(base_args(root / names[0], args.seed))
    opts.save_dir = str(PROJ / "outputs" / names[0])
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter (sparse checkpoint) for {len(names)} objects ===", flush=True)
    pvd = ViewCrafter(opts)          # loads DUSt3R + diffusion, and the first object's cloud
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log = []
    for name in names:
        src = root / name
        save_dir = PROJ / "outputs" / name
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(src, args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o

        # The fused cloud and its render are not freed between objects, and by the
        # fifth the card is 38 GB deep and 286 MB short. Reclaim between runs.
        torch.cuda.empty_cache()
        n_views = len(list(src.glob("*.png")))
        t = time.time()
        print(f"\n=== {name}  ({n_views} views -> {n_views - 1} clips) ===", flush=True)
        pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(src))
        pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
        seed_everything(o.seed)
        pvd.nvs_sparse_view_interp()
        dt = time.time() - t
        print(f"=== {name} done in {dt:.0f}s ===", flush=True)
        log.append({"object": name, "views": n_views, "clips": n_views - 1,
                    "seed": args.seed, "seconds": round(dt, 1),
                    "output_dir": str(save_dir)})
        (PROJ / "outputs" / "runs.json").write_text(json.dumps(log, indent=2))

    print(f"\n=== {len(log)} objects, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
