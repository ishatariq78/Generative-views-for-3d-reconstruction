#!/usr/bin/env python3
"""Experiment 6 again -- ViewCrafter sparse-view interpolation -- on the corrected
input order (wooden table only). Same seed, checkpoint, thresholds and memory
settings as experiment6/scripts/run_sparse_viewcrafter.py; only PROJ differs.

    python scripts/run_sparse.py [--only ADT1292__woodentable]
"""
import argparse, json, os, sys, time
from pathlib import Path

VC = Path("/home/isha.tariq/view_crafter/ViewCrafter")
sys.path.insert(0, str(VC)); os.chdir(VC)
import torch                                          # noqa: E402
from configs.infer_config import get_parser           # noqa: E402
from pytorch_lightning import seed_everything         # noqa: E402
from viewcrafter import ViewCrafter                   # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment10")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as C                                   # noqa: E402


def base_args(image_dir, seed):
    return ["--image_dir", str(image_dir), "--mode", "sparse_view_interp",
            "--out_dir", str(PROJ / "outputs"), "--bg_trd", "0.2", "--dpt_trd", "0.8",
            "--seed", str(seed), "--ckpt_path", "./checkpoints/model_sparse.ckpt",
            "--config", "configs/inference_pvd_1024.yaml",
            "--model_path", "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
            "--ddim_steps", "50", "--video_length", "25", "--device", "cuda:0",
            "--height", "576", "--width", "1024"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--only", nargs="*", default=["ADT1292__woodentable"])
    args = ap.parse_args()
    root = PROJ / "inputs8"
    names = [n for n in C.names(args.only) if (root / n).is_dir()]
    parser = get_parser()
    opts = parser.parse_args(base_args(root / names[0], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{names[0]}__sparse")
    os.makedirs(opts.save_dir, exist_ok=True)
    t0 = time.time()
    print(f"=== building ViewCrafter (sparse checkpoint) for {len(names)} objects ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)
    log = []
    for name in names:
        src, save_dir = root / name, PROJ / "outputs" / f"{name}__sparse"
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(src, args.seed)); o.save_dir = str(save_dir); pvd.opts = o
        torch.cuda.empty_cache()
        n_views = len(list(src.glob("*.png")))
        t = time.time()
        print(f"\n=== {name}__sparse  ({n_views} views -> {n_views - 1} clips) ===", flush=True)
        pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(src))
        pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
        seed_everything(o.seed)
        pvd.nvs_sparse_view_interp()
        dt = time.time() - t
        print(f"=== {name}__sparse done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": f"{name}__sparse", "object": name, "condition": "sparse",
                    "views": n_views, "clips": n_views - 1, "seed": args.seed,
                    "seconds": round(dt, 1), "output_dir": str(save_dir)})
        rec = PROJ / "outputs" / "runs.json"
        prev = json.loads(rec.read_text()) if rec.exists() else []
        merged = {r["experiment"]: r for r in prev}; merged.update({r["experiment"]: r for r in log})
        rec.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["experiment"]), indent=2))
    print(f"\n=== {len(log)} objects, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
