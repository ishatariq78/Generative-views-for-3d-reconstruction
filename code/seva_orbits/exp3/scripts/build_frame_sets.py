#!/usr/bin/env python3
"""Hand SEVA's exp1 orbits to the ViewCrafter-workspace VGGT pipeline as new conditions.

    python scripts/build_frame_sets.py

Writes into ~/viewcrafter_orbits/experiment3/inputs/, where run_vggt.py,
align_and_score.py and experiment 4's SLAM comparison already pick up any
<object>__<condition> folder:

  <object>__seva360    25 frames sampled uniformly from the 80-frame orbit --
                       the same count as every ViewCrafter condition (gen360,
                       mv360, v3_360, p3_*), so the comparison is like for like.
  <object>__seva360f   all 80 frames, to see whether VGGT gains from more views.

Frames are SEVA's own output: object on the exp2 studio backdrop, which is the
same backdrop run_vggt.py's "saturated or dark = object" mask was written for.
"""
import json
import shutil
from pathlib import Path

E1 = Path("/home/isha.tariq/stable_virtual_camera/exp1")
DST = Path("/home/isha.tariq/viewcrafter_orbits/experiment3/inputs")
N25 = 25

objs = [r["object"] for r in json.loads((E1 / "inputs/manifest.json").read_text())]
sets = []
for o in objs:
    src = E1 / "outputs/demo/img2trajvid" / o / "samples-rgb"
    frames = sorted(src.glob("*.png"))
    assert len(frames) == 80, (o, len(frames))
    pick = [frames[round(i * (len(frames) - 1) / (N25 - 1))] for i in range(N25)]
    for cond, sel in [("seva360", pick), ("seva360f", frames)]:
        d = DST / f"{o}__{cond}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for i, f in enumerate(sel):
            shutil.copyfile(f, d / f"{i:03d}.png")
        sets.append(d.name)
        print(f"{d.name:34s} {len(sel)} frames")
Path("/home/isha.tariq/stable_virtual_camera/exp3/sets.json").write_text(json.dumps(sets, indent=2))
