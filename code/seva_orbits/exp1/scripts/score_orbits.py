#!/usr/bin/env python3
"""Score each generated orbit against the object's true shape, angle by angle.

    /home/isha.tariq/.conda/envs/shaper/bin/python scripts/score_orbits.py

This is the measurement the ViewCrafter experiments could not make. There, the
target cameras were whatever the model inferred, so "is the object the right
shape from behind?" had no ground truth to compare against. Here we *chose* the
target cameras and we have ShapeR's reference mesh, so for every one of the 80
frames the true silhouette can be rasterised from exactly the camera that frame
was asked for, and compared with what came back.

Segmenting the generated object is the only soft step, so it is calibrated rather
than assumed. The backdrop is the known `studio_backdrop` ramp, so a pixel is
called object when it departs from that ramp -- except along the ramp's own
colour direction, which is what the contact shadow does. The same masker is then
run over the *input* images, whose true masks we cut ourselves, and the resulting
IoU is reported as `masker_iou_on_inputs`. A low number there means the orbit
scores below are not to be trusted.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment1/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment2/scripts")

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pickle  # noqa: E402

import trimesh  # noqa: E402

from extract_reference_views import SHAPER_DATA  # noqa: E402
from build_object_only_views import render_silhouette, studio_backdrop  # noqa: E402

ROOT = Path("/home/isha.tariq/stable_virtual_camera/exp1")
OUT = 576
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
DIFF_T = 26.0        # how far off the backdrop ramp a pixel must sit
CHROMA_T = 0.030     # tolerance for "same hue as the backdrop, only darker"


def object_mask(img):
    """Everything that is not the studio backdrop or its contact shadow.

    The shadow is the backdrop scaled toward black, so it keeps the backdrop's
    colour direction; a pixel that is merely darker along that direction is
    background, one that moves off it is object.
    """
    a = np.asarray(img, np.float32)
    bg = studio_backdrop(a.shape[1], a.shape[0]).astype(np.float32)
    diff = np.linalg.norm(a - bg, axis=-1)

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)

    chroma = np.linalg.norm(unit(a) - unit(bg), axis=-1)
    darker = (a.sum(-1) <= bg.sum(-1) + 12.0)
    shadowish = (chroma < CHROMA_T) & darker
    return (diff > DIFF_T) & ~shadowish


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score(name):
    scene = ROOT / "inputs" / name
    gen = ROOT / "outputs" / "demo" / "img2trajvid" / name / "samples-rgb"
    if not gen.exists():
        return None
    meta = json.loads((scene / "meta.json").read_text())
    tf = json.loads((scene / "transforms.json").read_text())["frames"]
    n_in = meta["views"]

    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)

    def K_of(fr):
        K = np.eye(3)
        K[0, 0], K[1, 1] = fr["fl_x"] * fr["w"], fr["fl_y"] * fr["h"]
        K[0, 2], K[1, 2] = fr["cx"] * fr["w"], fr["cy"] * fr["h"]
        return K

    # Calibrate the masker on the inputs, whose true masks we produced ourselves.
    cal = []
    for i, fr in enumerate(tf[:n_in]):
        img = Image.open(scene / f"images/{i:03d}.png").convert("RGB")
        truth = render_silhouette(mesh, K_of(fr), OUT, OUT,
                                  np.array(fr["transform_matrix"]) @ CV2GL) > 0.5
        cal.append(iou(object_mask(img), truth))

    # Score the orbit: true silhouette from the very camera each frame was asked
    # for, against what the model drew.
    per, areas = [], []
    for j, fr in enumerate(tf[n_in:]):
        f = gen / f"{j:03d}.png"
        if not f.exists():
            continue
        img = Image.open(f).convert("RGB")
        truth = render_silhouette(mesh, K_of(fr), OUT, OUT,
                                  np.array(fr["transform_matrix"]) @ CV2GL) > 0.5
        got = object_mask(img)
        per.append(iou(got, truth))
        areas.append(float(got.sum()) / max(truth.sum(), 1))

    per, areas = np.array(per), np.array(areas)
    rec = {
        "object": name,
        "input_arc_deg": meta["azimuth_span_deg"],
        "masker_iou_on_inputs": round(float(np.mean(cal)), 4),
        "orbit_iou_mean": round(float(per.mean()), 4),
        "orbit_iou_min": round(float(per.min()), 4),
        "orbit_iou_worst_frame": int(per.argmin()),
        "area_ratio_mean": round(float(areas.mean()), 3),
        "area_ratio_max": round(float(areas.max()), 3),
        "frames_over_1_5x": int((areas > 1.5).sum()),
        "per_frame_iou": [round(float(v), 4) for v in per],
    }
    print(f"{name:22s} arc {meta['azimuth_span_deg']:5.1f}°  "
          f"masker {rec['masker_iou_on_inputs']:.3f}  "
          f"orbit IoU mean {per.mean():.3f} min {per.min():.3f} "
          f"(frame {per.argmin()*360//len(per):3d}°)  "
          f"area x{areas.mean():.2f}  >1.5x in {rec['frames_over_1_5x']}/{len(areas)}")
    return rec


if __name__ == "__main__":
    names = sys.argv[1:] or [r["object"] for r in
                             json.loads((ROOT / "inputs/manifest.json").read_text())]
    recs = [r for r in (score(n) for n in names) if r]
    p = ROOT / "outputs" / "scores.json"
    old = {r["object"]: r for r in json.loads(p.read_text())} if p.exists() else {}
    old.update({r["object"]: r for r in recs})          # merge, never overwrite
    p.write_text(json.dumps(list(old.values()), indent=2))
    print(f"\nscored {len(recs)} objects -> {p}")
