#!/usr/bin/env python3
"""Do the generated pixels actually follow the camera SEVA was asked for?

    ~/.conda/envs/shaper/bin/python scripts/sel_check_corr.py --objects <obj> ...

A reprojection-based reliability score is meaningless if the generated frame does not sit on
the camera it was requested from. This measures that directly, with no reference mesh: the
object's real SLAM points are projected into each requested orbit camera, and compared with
where the object actually is in the generated frame (its silhouette against the plain
backdrop). If the projected points do not land inside the rendered object, the trajectory
was not followed and reprojection scores must not be used.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

E = Path("/home/isha.tariq/seva_to_shaper")
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def subject_mask(rgb):
    a = np.asarray(rgb, np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > 0.15) | (mx < 0.70)


def run(obj):
    import pickle
    d = pickle.load(open(f"/home/isha.tariq/shapeR_testing/data/{obj}.pkl", "rb"))
    slam = d["points_model"].numpy().astype(np.float64)
    tf = json.loads((E / "inputs_full" / obj / "transforms.json").read_text())["frames"]
    ids = json.loads(next((E / "inputs_full" / obj).glob("train_test_split_*.json")).read_text())["test_ids"]
    frames = sorted((E / "outputs_full/demo/img2trajvid" / obj / "samples-rgb").glob("*.png"))
    inside, cover = [], []
    for k in range(0, len(frames), 8):
        fr = tf[ids[k]]; w, h = int(fr["w"]), int(fr["h"])
        f, cx, cy = fr["fl_x"] * w, fr["cx"] * w, fr["cy"] * h
        c2w = np.asarray(fr["transform_matrix"], np.float64) @ GL2CV
        P = (slam - c2w[:3, 3]) @ c2w[:3, :3]
        front = P[:, 2] > 1e-6
        u = cx + f * P[:, 0] / np.maximum(P[:, 2], 1e-9)
        v = cy + f * P[:, 1] / np.maximum(P[:, 2], 1e-9)
        ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() < 20:
            continue
        m = subject_mask(Image.open(frames[k]).convert("RGB"))
        hit = m[v[ok].astype(int), u[ok].astype(int)].mean()
        inside.append(float(hit)); cover.append(float(m.mean()))
    return (float(np.mean(inside)) if inside else float("nan"),
            float(np.min(inside)) if inside else float("nan"),
            float(np.mean(cover)) if cover else float("nan"), len(inside))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--objects", nargs="*", required=True)
    a = ap.parse_args()
    out = {}
    print(f"{'object':32s}{'mean hit':>10s}{'min hit':>9s}{'obj area':>10s}{'frames':>8s}")
    for o in a.objects:
        try:
            mh, mn, cv, n = run(o)
            out[o] = {"mean_hit": mh, "min_hit": mn, "object_area": cv, "frames": n}
            print(f"{o[:32]:32s}{mh:10.3f}{mn:9.3f}{cv:10.3f}{n:8d}")
        except Exception as e:
            print(f"{o[:32]:32s}  failed: {type(e).__name__}: {e}")
    (E / "sel/camera_correspondence.json").parent.mkdir(parents=True, exist_ok=True)
    (E / "sel/camera_correspondence.json").write_text(json.dumps(out, indent=2))
    if out:
        v = [r["mean_hit"] for r in out.values()]
        print(f"\nmean over objects: {np.mean(v):.3f}  (1.0 = every projected SLAM point lands on the object)")
