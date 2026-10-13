#!/usr/bin/env python3
"""Print the SEVA work left, grouped by how many input views each scene actually kept.

    python scripts/seva_groups.py --shard 0 --n_shards 2
    -> "5:objA,objB 8:objC,objD"

A one-sided capture often yields fewer than 8 usable views, and build_scenes.py names the
split file after the count it kept, so every group needs its own --num_inputs. Scenes whose
orbit is already complete are dropped, which makes the job resumable.
"""
import argparse, glob, json, os
from pathlib import Path

E = Path("/home/isha.tariq/seva_to_shaper")
ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n_shards", type=int, default=1)
ap.add_argument("--objects", default="objects_subset_built.json")
a = ap.parse_args()

names = [r["name"] for r in json.loads((E / a.objects).read_text())]
names = [n for i, n in enumerate(names) if i % a.n_shards == a.shard]
per = {}
for n in names:
    if (E / f"outputs_full/demo/img2trajvid/{n}/samples-rgb/079.png").exists():
        continue
    fs = glob.glob(str(E / f"inputs_full/{n}/train_test_split_*.json"))
    if not fs:
        continue
    k = int(os.path.basename(fs[0]).split("_")[-1][:-5])
    per.setdefault(k, []).append(n)
print(" ".join(f"{k}:" + ",".join(v) for k, v in sorted(per.items())))
