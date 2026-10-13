#!/usr/bin/env python3
"""Cut an object-centric reference image out of a ShapeR evaluation sample.

ViewCrafter cannot be *told* which object in a photo to orbit -- it takes no text
prompt for content and no segmentation input. The only channel that carries that
instruction is the reference image itself, so the object has to become the
picture: rectified, upright, centred, filling the frame. That is what this script
builds, and it is the whole reason the experiment is object-centric rather than a
room fly-through.

ShapeR's pickles make it possible without a detector. Alongside the SLAM stream
they carry the Aria RGB stream (`rgb_image_data`), the per-frame camera
(`rgb_camera_params`, Fisheye624), the pose of each frame relative to the object
(`Ts_rgbCamera_model`), the object's own 3D points (`points_model`), and
`rgb_object_point_projections` -- those points already projected into each frame.
The projections answer "where is this object in this frame" exactly and for free.

Three steps that matter, in order:

  1. Rank frames on how well they show *this* object -- support, apparent size,
     centrality, clearance from the border. A clipped object gives ViewCrafter's
     DUSt3R stage a truncated shape to orbit around.
  2. Rectify the winner from Fisheye624 to a pinhole projection, reusing ShapeR's
     own `preprocessing.helper.rectify_images`. Not cosmetic: DUSt3R assumes a
     pinhole camera, and Aria's fisheye bends straight edges hard enough to
     corrupt the point cloud the orbit is rendered from.
  3. Locate the object in the rectified frame by painting `rgb_object_point_projections`
     into a mask and pushing that mask through the *same* rectifier, then take a
     16:9 crop around it at 1024x576 -- ViewCrafter's native input size.
     Warping the mask rather than re-projecting the 3D points is deliberate: it
     needs no assumption about which direction `Ts_rgbCamera_model` transforms
     in, and getting that backwards silently puts the crop on the wrong piece of
     furniture -- which is exactly what happened on the first attempt.

Aria Gen1 RGB frames are stored rotated a quarter turn. ShapeR rotates its SLAM
frames CCW; the RGB stream needs the other direction -- verified by rendering
both and looking -- so this rotates CW and rotates the intrinsics to match.

    python scripts/extract_reference_views.py ADT1292__stool BNB1045__sofa
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
from preprocessing.helper import rectify_images  # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
OUT_W, OUT_H = 1024, 576     # ViewCrafter's native resolution
CROP_MARGIN = 1.45           # object extent -> crop width; room to sit in, no more
N_CANDIDATES = 40            # frames rectified before the final pick (rectify is slow)


# --------------------------------------------------------------------------- scoring
def rank_raw_frames(projs, W, H):
    """Cheap pre-ranking on the fisheye projections, to choose what to rectify."""
    out = []
    for i, p in enumerate(projs):
        pts = np.asarray(p.numpy() if hasattr(p, "numpy") else p, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 40:
            continue
        # 2nd/98th percentile, not min/max: a few stray points on a neighbouring
        # surface would otherwise blow the box up to the whole room.
        x0, y0 = np.percentile(pts, 2, axis=0)
        x1, y1 = np.percentile(pts, 98, axis=0)
        w, h = x1 - x0, y1 - y0
        if w < 8 or h < 8:
            continue
        size = np.sqrt(w * h) / np.sqrt(W * H)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        off = np.hypot(cx - W / 2, cy - H / 2) / np.hypot(W / 2, H / 2)
        border = min(x0, y0, W - x1, H - y1) / min(W, H)
        if border < 0.0:
            continue
        support = min(pts.shape[0] / 500.0, 1.0)
        out.append((4.0 * size + 0.7 * support - 0.9 * off + min(border, .06) / .06 * 0.5,
                    i, (float(x0), float(y0), float(x1), float(y1)), int(pts.shape[0])))
    out.sort(reverse=True)
    return out


def score_rect(box, W, H):
    """Final score in rectified pixel space. Apparent size dominates -- a small
    object upscaled to 1024x576 is mush, and mush is what DUSt3R fails on -- but
    a whole object beats a big one: a frame where the extent touches the border
    is rejected outright, however large it looks. ViewCrafter orbits whatever
    geometry DUSt3R recovers, and half an armchair orbits as half an armchair."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return -1.0
    pad = 0.015 * min(W, H)
    if x0 < pad or y0 < pad or x1 > W - pad or y1 > H - pad:
        return -1.0
    size = np.sqrt(w * h) / np.sqrt(W * H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    off = np.hypot(cx - W / 2, cy - H / 2) / np.hypot(W / 2, H / 2)
    return 5.0 * size - 1.0 * off


# --------------------------------------------------------------------------- geometry
def bbox_of_mask(mask, W, H):
    """Extent of the warped object mask. 2nd/98th percentile of the lit pixels, so
    a few stragglers from rectification resampling cannot stretch the box."""
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size < 40:
        return None
    x0, y0 = np.percentile(xs, 2), np.percentile(ys, 2)
    x1, y1 = np.percentile(xs, 98), np.percentile(ys, 98)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def rotate_intrinsics_cw90(K, old_H):
    """Intrinsics for a 90-degree clockwise image rotation: (x, y) -> (old_H - y, x),
    so the focal lengths swap and the principal point moves with them."""
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, 0], out[1, 1] = K[1, 1], K[0, 0]
    out[0, 2] = old_H - K[1, 2]
    out[1, 2] = K[0, 2]
    return out


def crop_window(box, W, H):
    """16:9 window centred on the object extent, clamped inside the image."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ow, oh = x1 - x0, y1 - y0
    cw = max(ow * CROP_MARGIN, oh * CROP_MARGIN * OUT_W / OUT_H)
    ch = cw * OUT_H / OUT_W
    if cw > W:
        cw, ch = W, W * OUT_H / OUT_W
    if ch > H:
        ch, cw = H, H * OUT_W / OUT_H
    cx = min(max(cx, cw / 2), W - cw / 2)
    cy = min(max(cy, ch / 2), H - ch / 2)
    return (round(cx - cw / 2), round(cy - ch / 2), round(cx + cw / 2), round(cy + ch / 2))


def point_mask(pts, W, H, radius=6):
    """Paint the object's projected points into a binary plane, thickened enough
    that the rectifier's nearest-neighbour resampling cannot erase them."""
    m = np.zeros((H, W), dtype=np.float32)
    x = np.clip(pts[:, 0].astype(int), 0, W - 1)
    y = np.clip(pts[:, 1].astype(int), 0, H - 1)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            m[np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)] = 1.0
    return m


def rectify_one(rgb, mask, cam_param):
    """Fisheye624 -> pinhole via ShapeR's own rectifier, image and object mask
    together so both land in the same rectified pixel grid."""
    img = np.asarray(rgb, dtype=np.uint8)
    H, W = img.shape[:2]
    rect, rect_mask, rect_cam = rectify_images(
        [torch.from_numpy(img.copy())],
        [torch.from_numpy(mask[..., None].copy())],
        [torch.as_tensor(np.asarray(cam_param))],
    )
    return rect[0], np.asarray(rect_mask[0], dtype=np.float32) / 255.0, np.asarray(rect_cam[0], dtype=np.float64)


# --------------------------------------------------------------------------- main
def process(name, out_dir, rotate_cw=None):
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
    cams = d["rgb_camera_params"]
    if not frames:
        raise SystemExit(f"{name}: sample carries no RGB stream")

    W0, H0 = Image.open(io.BytesIO(frames[0])).size
    # Gen1 (1408x1408) frames are stored a quarter turn off; Gen2 (2016x1512) upright.
    if rotate_cw is None:
        rotate_cw = not bool(d["is_ariagen2"])

    candidates = rank_raw_frames(projs, W0, H0)
    if not candidates:
        raise SystemExit(f"{name}: no RGB frame shows the object unclipped")

    best = None
    for _, idx, _, npts in candidates[:N_CANDIDATES]:
        rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
        pts = np.asarray(projs[idx].numpy() if hasattr(projs[idx], "numpy") else projs[idx],
                         dtype=np.float32).reshape(-1, 2)
        rect, rmask, K = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
        if rotate_cw:
            K = rotate_intrinsics_cw90(K, rect.shape[0])
            rect = np.rot90(rect, 3).copy()
            rmask = np.rot90(rmask, 3).copy()
        H, W = rect.shape[:2]
        box = bbox_of_mask(rmask, W, H)
        if box is None:
            continue
        s = score_rect(box, W, H)
        if s < 0:
            continue
        if best is None or s > best[0]:
            best = (s, idx, rect, box, npts, (W, H))

    if best is None:
        raise SystemExit(f"{name}: no candidate frame survived rectification")
    s, idx, rect, box, npts, (W, H) = best

    win = crop_window(box, W, H)
    full = Image.fromarray(rect)
    ref = full.crop(win).resize((OUT_W, OUT_H), Image.LANCZOS)

    out_dir.mkdir(parents=True, exist_ok=True)
    ref.save(out_dir / f"{name}.png")

    sheet = full.copy()
    dr = ImageDraw.Draw(sheet)
    dr.rectangle(box, outline=(255, 190, 0), width=max(3, W // 300))
    dr.rectangle(win, outline=(60, 200, 90), width=max(3, W // 400))
    sheet.thumbnail((900, 900), Image.LANCZOS)
    sheet.save(out_dir / f"CONTEXT__{name}.jpg", quality=88)

    meta = {
        "sample": name, "caption": d["caption"], "category": d["category"],
        "is_ariagen2": bool(d["is_ariagen2"]), "rotated_cw90": bool(rotate_cw),
        "rgb_frame_index": int(idx), "rgb_frames_available": len(frames),
        "raw_resolution": [W0, H0], "rectified_resolution": [W, H],
        "projected_points": int(npts),
        "object_bbox_rectified_px": [round(v, 1) for v in box],
        "object_px_extent": [round(box[2] - box[0]), round(box[3] - box[1])],
        "crop_window_px": list(win), "frame_score": round(float(s), 4),
        "reference_image": str(out_dir / f"{name}.png"),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"{name:24s} frame {idx:5d}/{len(frames):<5d} obj "
          f"{meta['object_px_extent'][0]:4d}x{meta['object_px_extent'][1]:<4d}px  "
          f"score {s:.3f}  crop {win}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--out_dir", type=Path, default=Path("/home/isha.tariq/viewcrafter_orbits/experiment1/inputs"))
    args = ap.parse_args()
    metas = [process(s, args.out_dir) for s in args.samples]
    (args.out_dir / "manifest.json").write_text(json.dumps(metas, indent=2))
    print(f"\nwrote {len(metas)} reference views to {args.out_dir}")
