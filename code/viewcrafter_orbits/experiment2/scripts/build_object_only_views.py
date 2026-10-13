#!/usr/bin/env python3
"""Build reference views that contain the target object and nothing else.

Experiment 1 handed ViewCrafter a crop of the room with the object in the middle
of it, and got back a video of the room: the sofa run drifts into a different
sofa, the chair run re-invents the whole dining table. ViewCrafter has no notion
of a target, so a scene in the frame is a scene in the output.

This removes the scene. The object's reference mesh is rasterised into the
reference camera to get a pixel-exact silhouette, and everything outside it is
replaced. What ViewCrafter receives is a product shot: one object, centred,
against a plain studio backdrop, with nothing else for the orbit to latch onto.

Two treatments are produced from the same frame and the same silhouette:

  cutout    background replaced by a neutral gradient with a contact shadow.
            The literal request -- only the object survives.
  context   background kept but darkened, desaturated and blurred. A hedge:
            DUSt3R needs texture to estimate depth, and a hard cutout leaves it
            almost none, so this keeps low-frequency anchors while still making
            the object unambiguously the subject.

Where the silhouette comes from, and how it is checked
------------------------------------------------------
The pickle carries the object's reference mesh (`mesh_vertices`/`mesh_faces`, in
model coordinates) and `Ts_rgbCamera_model`, the per-frame pose. Rasterising the
mesh through the rectified intrinsics gives an exact mask. The catch is that the
pose's *direction* is not documented, and guessing it wrong does not error -- it
silently masks empty floor. So both directions are rendered and scored against
the warped point-projection mask (the localisation experiment 1 already trusts),
and the better one wins. Anything below a hard overlap floor aborts the sample
rather than producing a plausible-looking wrong crop.

Needs a GPU for pyrender's EGL context; run it through slurm/build_views.sbatch.
"""
import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment1/scripts")

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402
import trimesh   # noqa: E402

# The frame ranking, rectifier wrapper and point-mask painter are experiment 1's;
# reusing them keeps both experiments looking at the same frame of the same
# capture, so the only thing that changes between them is the masking.
from extract_reference_views import (  # noqa: E402
    SHAPER_DATA, bbox_of_mask, point_mask, rank_raw_frames, rectify_one,
)

OUT_W, OUT_H = 1024, 576
FILL = 0.66          # object's larger side, as a fraction of the crop's short side
N_CANDIDATES = 40
MIN_OVERLAP = 0.35   # mesh mask must cover this fraction of the projected points
# Reject frames taken from closer than this multiple of the object's own diagonal
# -- a guard against the wearer walking right into a large object. It is a weak
# filter and deliberately so: a view can be geometrically fine at 0.5 diagonals
# (the bed is) and useless anyway if the object is an untextured white panel in a
# dark room, which is a photometric problem no geometric test catches. The fix for
# that case is to pick a different object, not a cleverer threshold.
MIN_CAM_DIST = 0.5
FEATHER_PX = 2.5

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV camera -> pyrender/OpenGL camera


# ------------------------------------------------------------------ silhouette
def rigid(T):
    """pyrender rejects a pose whose bottom row is not exactly [0,0,0,1], and
    np.linalg.inv leaves float dust there. Re-seat it."""
    out = np.array(T, dtype=np.float64, copy=True)
    out[3, :] = [0.0, 0.0, 0.0, 1.0]
    return out


_RENDERERS = {}


def _renderer(W, H):
    """One renderer per resolution -- building an EGL context for each of the
    ~240 candidate renders costs more than everything else in this script."""
    if (W, H) not in _RENDERERS:
        _RENDERERS[(W, H)] = pyrender.OffscreenRenderer(W, H)
    return _RENDERERS[(W, H)]


def render_silhouette(mesh, K, W, H, pose_world_cam):
    """Rasterise the mesh into a camera with these intrinsics and pose."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.IntrinsicsCamera(fx=float(K[0, 0]), fy=float(K[1, 1]),
                                    cx=float(K[0, 2]), cy=float(K[1, 2]),
                                    znear=0.01, zfar=100.0)
    scene.add(cam, pose=rigid(pose_world_cam @ CV2GL))
    # DEPTH_ONLY returns the depth buffer alone, not a (colour, depth) pair.
    depth = _renderer(W, H).render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    return (np.asarray(depth) > 0).astype(np.float32)


def silhouette_for_frame(mesh, T_pair, K, W, H, ref_points_mask):
    """Try both readings of Ts_rgbCamera_model, keep whichever actually lands on
    the object. Returns (mask, overlap, which)."""
    best = (None, -1.0, None)
    for label, pose in T_pair:
        try:
            m = render_silhouette(mesh, K, W, H, pose)
        except Exception as exc:                      # a degenerate pose can throw
            print(f"    [{label}] render failed: {exc}")
            continue
        denom = ref_points_mask.sum()
        overlap = float((m * ref_points_mask).sum() / denom) if denom else 0.0
        print(f"    [{label}] silhouette covers {overlap:.3f} of the projected points")
        if overlap > best[1]:
            best = (m, overlap, label)
    return best


# ------------------------------------------------------------------ compositing
def studio_backdrop(w, h):
    """A soft vertical gradient rather than flat white. Flat colour gives DUSt3R
    literally nothing to match on; a gentle ramp reads as a backdrop to the video
    prior while still carrying no structure that could be mistaken for scene."""
    top, bot = np.array([242, 242, 244], np.float32), np.array([206, 208, 213], np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return (top * (1 - t) + bot * t).repeat(w, axis=1).astype(np.uint8)


def contact_shadow(mask, w, h):
    """A soft ellipse under the silhouette, so the object sits on something
    instead of floating. Purely a cue for the video prior."""
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size == 0:
        return np.zeros((h, w), np.float32)
    # float() throughout: these come out of numpy as 0-d arrays, and PIL's
    # GaussianBlur takes the radius as a plain number.
    cx, base = float(xs.min() + xs.max()) / 2, float(ys.max())
    rx = float(xs.max() - xs.min()) * 0.55
    ry = max(6.0, float(ys.max() - ys.min()) * 0.07)
    # 8-bit "L", not float "F": PIL's GaussianBlur rejects float images.
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).ellipse([cx - rx, base - ry, cx + rx, base + ry], fill=115)
    img = img.filter(ImageFilter.GaussianBlur(max(4.0, ry * 0.9)))
    return np.asarray(img, np.float32) / 255.0


def feather(mask, px=FEATHER_PX):
    m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(px))
    return np.asarray(m, np.float32) / 255.0


def make_cutout(rgb, alpha):
    h, w = alpha.shape
    bg = studio_backdrop(w, h).astype(np.float32)
    sh = contact_shadow(alpha, w, h)[..., None]
    bg = bg * (1.0 - sh)
    a = alpha[..., None]
    return (rgb.astype(np.float32) * a + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)


def make_context(rgb, alpha):
    """Object untouched; everything else blurred, darkened and desaturated."""
    im = Image.fromarray(rgb)
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(9)), np.float32)
    grey = blur @ np.array([0.299, 0.587, 0.114], np.float32)
    back = (0.30 * blur + 0.70 * grey[..., None]) * 0.55
    a = alpha[..., None]
    return (rgb.astype(np.float32) * a + back * (1.0 - a)).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------ framing
def crop_window(box, W, H):
    """16:9 window centred on the silhouette. Tighter than experiment 1's, because
    with the scene gone there is no reason to keep room for it -- and ViewCrafter
    pivots the orbit at the depth of the centre pixel, which now lands on the
    object itself rather than on whatever was behind it."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ch = max((y1 - y0) / FILL, (x1 - x0) / FILL * OUT_H / OUT_W)
    cw = ch * OUT_W / OUT_H
    if cw > W:
        cw, ch = W, W * OUT_H / OUT_W
    if ch > H:
        ch, cw = H, H * OUT_W / OUT_H
    cx = min(max(cx, cw / 2), W - cw / 2)
    cy = min(max(cy, ch / 2), H - ch / 2)
    return (round(cx - cw / 2), round(cy - ch / 2), round(cx + cw / 2), round(cy + ch / 2))


def camera_distance(T_cam_model, centre):
    """How far the camera sits from the object's centre, in model units.
    T is camera-from-model (established by the overlap test above), so the
    camera's position in model coordinates is -R^T t."""
    T = np.asarray(T_cam_model, dtype=np.float64)
    pos = -T[:3, :3].T @ T[:3, 3]
    return float(np.linalg.norm(pos - centre))


# ------------------------------------------------------------------ main
def process(name, out_dir):
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
    cams, poses = d["rgb_camera_params"], d["Ts_rgbCamera_model"]
    rotate_cw = not bool(d["is_ariagen2"])

    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    centre = mesh.bounds.mean(axis=0)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    W0, H0 = Image.open(io.BytesIO(frames[0])).size
    candidates = rank_raw_frames(projs, W0, H0)
    print(f"\n=== {name}: {len(candidates)} usable frames, rectifying top {N_CANDIDATES} ===")

    best = None
    for _, idx, _, npts in candidates[:N_CANDIDATES]:
        dist = camera_distance(poses[idx].numpy(), centre)
        if dist < MIN_CAM_DIST * diag:
            print(f"    frame {idx}: camera {dist / diag:.2f} diagonals away -- too close, skipped")
            continue
        rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
        pts = np.asarray(projs[idx].numpy(), np.float32).reshape(-1, 2)
        rect, rmask, K = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
        H, W = rect.shape[:2]

        T = np.asarray(poses[idx].numpy(), np.float64)
        sil, overlap, which = silhouette_for_frame(
            mesh, [("inv(T) : T is camera-from-model", rigid(np.linalg.inv(T))),
                   ("T      : T is model-from-camera", rigid(T))],
            K, W, H, (rmask > 0.5).astype(np.float32))
        if sil is None or overlap < MIN_OVERLAP or sil.sum() < 2000:
            continue

        box = bbox_of_mask(sil, W, H)
        if box is None:
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        pad = 0.012 * min(W, H)
        if box[0] < pad or box[1] < pad or box[2] > W - pad or box[3] > H - pad:
            continue                                   # silhouette clipped by the frame
        score = np.sqrt(bw * bh) / np.sqrt(W * H) + 0.5 * overlap
        if best is None or score > best[0]:
            best = (score, idx, rect, sil, box, overlap, which, npts, (W, H), K)

    if best is None:
        raise SystemExit(f"{name}: no frame produced a usable, unclipped silhouette")
    score, idx, rect, sil, box, overlap, which, npts, (W, H), K = best

    if rotate_cw:
        rect = np.rot90(rect, 3).copy()
        sil = np.rot90(sil, 3).copy()
        H, W = rect.shape[:2]
        box = bbox_of_mask(sil, W, H)

    alpha = feather(sil)
    cut = make_cutout(rect, alpha)
    ctx = make_context(rect, alpha)
    win = crop_window(box, W, H)

    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, img in (("cutout", cut), ("context", ctx)):
        Image.fromarray(img).crop(win).resize((OUT_W, OUT_H), Image.LANCZOS) \
             .save(out_dir / f"{name}__{tag}.png")

    # Contact sheet: original crop, the silhouette, and both treatments, so the
    # mask can be checked without opening anything else.
    panels = []
    for img in (rect, (np.stack([sil] * 3, -1) * 255).astype(np.uint8), cut, ctx):
        panels.append(Image.fromarray(img).crop(win).resize((460, 259), Image.LANCZOS))
    sheet = Image.new("RGB", (460 * 2, 259 * 2), (18, 18, 20))
    for k, p in enumerate(panels):
        sheet.paste(p, (460 * (k % 2), 259 * (k // 2)))
    sheet.save(out_dir / f"CONTEXT__{name}.jpg", quality=88)

    meta = {
        "sample": name, "caption": d["caption"], "rgb_frame_index": int(idx),
        "rgb_frames_available": len(frames), "rotated_cw90": bool(rotate_cw),
        "rectified_resolution": [W, H], "pose_reading_used": which,
        "silhouette_overlap_with_points": round(float(overlap), 4),
        "silhouette_px": int(sil.sum()),
        "object_bbox_px": [round(v, 1) for v in box],
        "object_px_extent": [round(box[2] - box[0]), round(box[3] - box[1])],
        "crop_window_px": list(win), "frame_score": round(float(score), 4),
        "treatments": {t: str(out_dir / f"{name}__{t}.png") for t in ("cutout", "context")},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> frame {idx}, overlap {overlap:.3f} via {which.split(':')[0].strip()}, "
          f"object {meta['object_px_extent'][0]}x{meta['object_px_extent'][1]} px, crop {win}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/viewcrafter_orbits/experiment2/inputs"))
    args = ap.parse_args()
    metas = [process(s, args.out_dir) for s in args.samples]
    (args.out_dir / "manifest.json").write_text(json.dumps(metas, indent=2))
    print(f"\nwrote {len(metas)} x 2 treatments to {args.out_dir}")
