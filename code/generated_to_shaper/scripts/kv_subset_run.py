#!/usr/bin/env python3
"""Play the subset orbits through KV-Tracker, so its cloud can be used in place of SLAM points.

    ~/.conda/envs/kvtracker/bin/python scripts/kv_subset_run.py

The tracker's TUM loader reads <scene_dir>/rgb/*.png, so each object's orbit frames are
symlinked into that shape. `--kf_auto 1` (a keyframe on every frame) is the setting that
scored best in the earlier KV work; it needs the two env-gated local patches in main.py
(KVT_MAX_KF lifts the upstream 20-keyframe cap, KVT_KEEP_LAST_PCD keeps only the newest
cumulative cloud).
"""
import json, os, subprocess, sys, time
from pathlib import Path

E = Path("/home/isha.tariq/seva_to_shaper")
REPO = Path("/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
PY = "/home/isha.tariq/.conda/envs/kvtracker/bin/python"
ORBITS = E / "outputs_full/demo/img2trajvid"


def main():
    names = [r["name"] for r in json.loads((E / "objects_kvtest.json").read_text())]
    root = E / "outputs_sub/kv"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR="/tmp", HF_HOME="/home/isha.tariq/.cache/huggingface",
               KVT_KEEP_LAST_PCD="1", KVT_MAX_KF="80")
    ok = 0
    for i, obj in enumerate(names, 1):
        out = root / obj
        if list(out.glob("pcd_*.ply")) and (out / "traj.npy").exists():
            print(f"[{i}/{len(names)}] {obj}: already tracked", flush=True); ok += 1; continue
        frames = sorted((ORBITS / obj / "samples-rgb").glob("*.png"))
        if len(frames) != 80:
            print(f"[{i}/{len(names)}] {obj}: {len(frames)} frames, skipped", flush=True); continue
        scene = root / "_scenes" / obj / "rgb"
        scene.mkdir(parents=True, exist_ok=True)
        for f in scene.glob("*.png"):
            f.unlink()
        for f in frames:
            (scene / f.name).symlink_to(f)
        cfg = root / "_scenes" / f"{obj}.yaml"
        cfg.write_text("datasource: TUMLoader\nque_size: -1\noffset: 0\nuse_bbox: -1\n"
                       "export_masked_imgs: False\n"
                       f"results_path: {out}\nscene_dir: {scene.parent}\n")
        t0 = time.time()
        r = subprocess.run([PY, "main.py", str(cfg), "--cam_only", "--export_pcd",
                            "--resize_dim", "518", "--kf_auto", "1"],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=2400)
        got = bool(list(out.glob("pcd_*.ply"))) and (out / "traj.npy").exists()
        ok += got
        print(f"[{i}/{len(names)}] {obj}: rc={r.returncode} {'OK' if got else 'NO OUTPUT'} "
              f"in {time.time()-t0:.0f}s", flush=True)
        if not got:
            print("   tail:", (r.stderr or r.stdout)[-300:].replace("\n", " "), flush=True)
    print(f"### tracked {ok}/{len(names)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
