#!/usr/bin/env python3
"""Drive SEVA over the subset from one process: group, launch, log, skip what is done.

    python scripts/run_seva_subset.py --shard 0 --n_shards 2

The shell version of this went through `$(...)` into `set -e` and died silently three
different ways (a group helper whose exit status killed the assignment, an f-string
backslash, a `[ test ] && {}` returning non-zero). There is no reason for a shell to be
in the loop at all: this computes the groups, calls demo.py per group with its own
--num_inputs, and reports each group's outcome instead of taking the job down with it.

A one-sided capture often yields fewer than 8 usable views, and build_scenes.py names the
split file after the count it kept, so --num_inputs must match per scene.
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

E = Path("/home/isha.tariq/seva_to_shaper")
P = Path("/home/isha.tariq/stable_virtual_camera")
PY = "/home/isha.tariq/.conda/envs/svcam/bin/python"


def groups(objects_file, shard, n_shards):
    names = [r["name"] for r in json.loads((E / objects_file).read_text())]
    names = [n for i, n in enumerate(names) if i % n_shards == shard]
    per = {}
    for n in names:
        if (E / f"outputs_full/demo/img2trajvid/{n}/samples-rgb/079.png").exists():
            continue
        fs = glob.glob(str(E / f"inputs_full/{n}/train_test_split_*.json"))
        if not fs:
            print(f"  skip {n}: no split file", flush=True)
            continue
        k = int(os.path.basename(fs[0]).split("_")[-1][:-5])
        per.setdefault(k, []).append(n)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--objects", default="objects_full_built.json")
    ap.add_argument("--chunk", type=int, default=6, help="objects per demo.py call")
    a = ap.parse_args()

    run_dir = E / "run_full"
    run_dir.mkdir(exist_ok=True)
    (E / "outputs_full").mkdir(exist_ok=True)
    link = run_dir / "work_dirs"                       # demo.py resolves WORK_DIR from the cwd
    if not link.is_symlink():
        link.symlink_to(E / "outputs_full")

    env = dict(os.environ, TMPDIR="/tmp",
               HF_HOME="/home/isha.tariq/.cache/huggingface",
               SEVA_VAE_PATH=str(P / "assets/sd21base"))
    per = groups(a.objects, a.shard, a.n_shards)
    total = sum(len(v) for v in per.values())
    print(f"shard {a.shard}/{a.n_shards}: {total} objects left, "
          f"groups " + ", ".join(f"{k}->{len(v)}" for k, v in sorted(per.items())), flush=True)
    if not total:
        return 0

    ok = bad = 0
    for k, names in sorted(per.items()):
        for i in range(0, len(names), a.chunk):       # chunked so one bad object loses less
            batch = names[i:i + a.chunk]
            print(f"\n### num_inputs={k}  {len(batch)} objects: {','.join(batch)}", flush=True)
            cmd = [PY, str(P / "third_party/stable-virtual-camera/demo.py"),
                   "--data_path", str(E / "inputs_full"), "--data_items", ",".join(batch),
                   "--task", "img2trajvid", "--num_inputs", str(k), "--cfg", "3.0,2.0",
                   "--use_traj_prior", "True", "--chunk_strategy", "interp-gt",
                   "--video_save_fps", "20"]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=run_dir, env=env)
            done = [n for n in batch if (E / f"outputs_full/demo/img2trajvid/{n}/samples-rgb/079.png").exists()]
            ok += len(done); bad += len(batch) - len(done)
            print(f"### group done rc={r.returncode} in {time.time()-t0:.0f}s: "
                  f"{len(done)}/{len(batch)} orbits complete", flush=True)
    print(f"\n### shard finished: {ok} orbits complete, {bad} missing", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
