#!/usr/bin/env python3
"""SEVA's 80 orbit frames per object -> the two reconstructors' input layouts.

    python scripts/make_frame_sets.py            # any env with numpy

  outputs/frames/<obj>/000..079.png       Pi3 batch input (run_pi3.py reads a flat folder)
  outputs/kv_inputs/<obj>/rgb/*.png       KV-Tracker's TUMLoader reads <scene_dir>/rgb/*.png
  outputs/kv_inputs/<obj>.yaml            KV-Tracker config for that scene
"""
import json, shutil
from pathlib import Path

E = Path("/home/isha.tariq/seva_to_shaper")
manifest = json.loads((E / "inputs/manifest.json").read_text())
sets = []
for r in manifest:
    o = r["object"]
    src = E / "outputs/demo/img2trajvid" / o / "samples-rgb"
    frames = sorted(src.glob("*.png"))
    if len(frames) != 80:
        print(f"!! {o}: {len(frames)} frames, expected 80 -- skipped"); continue
    fd = E / "outputs/frames" / o
    if fd.exists(): shutil.rmtree(fd)
    fd.mkdir(parents=True)
    for i, f in enumerate(frames):
        shutil.copyfile(f, fd / f"{i:03d}.png")
    kd = E / "outputs/kv_inputs" / o / "rgb"
    if kd.parent.exists(): shutil.rmtree(kd.parent)
    kd.mkdir(parents=True)
    for f in sorted(fd.glob("*.png")):
        (kd / f.name).symlink_to(f)
    (E / "outputs/kv_inputs" / f"{o}.yaml").write_text(
        f"# SEVA 80-frame orbit of {o} fed to KV-Tracker (keyframe every frame; KVT_KEEP_LAST_PCD=1, KVT_MAX_KF=80)\n"
        "datasource: TUMLoader\nque_size: -1\noffset: 0\nuse_bbox: -1\nexport_masked_imgs: False\n"
        f"results_path: {E}/outputs/kv/{o}\nscene_dir: {E}/outputs/kv_inputs/{o}\n")
    sets.append(o); print(f"{o:26s} 80 frames")
(E / "outputs/sets.json").write_text(json.dumps(sets, indent=2))
