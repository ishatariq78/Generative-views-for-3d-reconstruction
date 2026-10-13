#!/usr/bin/env python3
"""Figures that explain the pipeline rather than report it.

The results figures assume you already know what VGGT is being asked to do and
what "SLAM points reproduced" counts. These four show it instead:

  explain_vggt.png     video frames in, a 3D point cloud out
  explain_compare.png  ShapeR's SLAM points and VGGT's points over the same mesh
  explain_2pct.png     every SLAM point coloured by whether VGGT found it
  explain_pipeline.png the whole chain as one diagram

    python scripts/render_explainer_figures.py
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v3 as iio          # noqa: E402
import numpy as np                # noqa: E402
import pyrender                   # noqa: E402
import trimesh                    # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402
from scipy.spatial import cKDTree             # noqa: E402

import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "experiment4" / "scripts"))
from render_comparison import camera_pose, point_mesh   # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
OUT = PROJ / "experiment9" / "outputs"
OBJ = "ADT1292__chair"
COND = "p3_level"
RES = 520
ELEV = 20.0
MESH_RGB = (150, 176, 150)
SLAM_RGB = (20, 128, 122)
VGGT_RGB = (235, 104, 52)
HIT_RGB, MISS_RGB = (26, 175, 122), (200, 58, 44)
INK, INK2, MUTED = (20, 24, 30), (74, 82, 94), (140, 146, 154)
TAU = 0.02


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def font_bold(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return font(sz)


def load_ply_rgb(path):
    """The project's PLY reader drops colour; VGGT's own per-point colour is what
    makes the cloud legible as a chair rather than a beige blob."""
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        n = int([l for l in head.split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
        rec = np.frombuffer(f.read(n * 15), count=n,
                            dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    rgb = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.uint8)
    return xyz, rgb


def shoot_points(renderer, pts, rgb, az, radius):
    """The raw cloud in its own frame, drawn with its own colours."""
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_points(pts, colors=rgb))
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=camera_pose(az, ELEV, radius))
    return Image.fromarray(renderer.render(scene)[0][..., :3])


def load(obj):
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    slam = np.asarray(d["points_model"].numpy(), np.float64)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    c = mesh.bounds.mean(axis=0)
    Rz = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    m = mesh.copy(); m.apply_translation(-c); m.apply_transform(Rz)
    m.visual.vertex_colors = np.array(list(MESH_RGB) + [255], np.uint8)
    return m, slam, diag, (lambda p: (Rz[:3, :3] @ (p - c).T).T)


def trim(im, pad=10):
    """Drop the white margin pyrender leaves around a tall, thin object."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a < 248)
    if not len(ys):
        return im
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(a.shape[1], xs.max() + pad), min(a.shape[0], ys.max() + pad)))


def place(im, box_w, box_h):
    """Fit a trimmed render into a fixed cell without distorting it."""
    sc = min(box_w / im.width, box_h / im.height)
    r = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
    cell = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    cell.paste(r, ((box_w - r.width) // 2, (box_h - r.height) // 2))
    return cell


def shoot(renderer, mesh, clouds, az, radius, with_mesh=True):
    """One render: the reference mesh plus any number of (points, rgb, radius)."""
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.45, 0.45, 0.45])
    if with_mesh:
        scene.add(pyrender.Mesh.from_trimesh(mesh))
    for pts, rgb, rad in clouds:
        if len(pts):
            scene.add(point_mesh(pts, rad, rgb))
    pose = camera_pose(az, ELEV, radius)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.2),
              pose=camera_pose(az + 35, 40, radius))
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.4),
              pose=camera_pose(az - 70, 10, radius))
    return Image.fromarray(renderer.render(scene)[0][..., :3])


def band(w, h, text, sub="", pad=14):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((pad, pad - 2), text, fill=INK, font=font_bold(22))
    if sub:
        dr.text((pad, pad + 26), sub, fill=INK2, font=font(17))
    return im


def fig_vggt(renderer, mesh, radius, vggt_s):  # noqa: C901
    """Frames in, points out."""
    v = iio.imread(PROJ / f"experiment9/outputs/{OBJ}__level/diffusion0.mp4", plugin="pyav")
    idx = [0, 6, 12, 18, 24]
    TW = 384; TH = int(TW * v.shape[1] / v.shape[2])
    SW, SH = 500, 560
    xyz, rgb = load_ply_rgb(PROJ / f"experiment3/outputs/{OBJ}__{COND}/points_object.ply")
    r2 = np.random.default_rng(0)
    if len(xyz) > 220000:
        k = r2.choice(len(xyz), 220000, replace=False)
        xyz, rgb = xyz[k], rgb[k]
    c = xyz.mean(0)
    xyz = xyz - c
    # VGGT's frame is arbitrary; scale to the renderer's working radius and flip
    # to y-up so the same camera helper frames it.
    xyz = xyz / max(np.linalg.norm(xyz, axis=1).max(), 1e-9) * (0.55 * radius * np.tan(np.pi / 8))
    xyz = xyz[:, [0, 1, 2]] * np.array([1.0, -1.0, 1.0])
    shots = [place(trim(shoot_points(renderer, xyz, rgb, az, radius)), SW, SH)
             for az in (0, 120, 240)]
    W = TW * len(idx)
    H = 66 + TH + 74 + SH + 40
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((14, 16), "1.  the 360° video ViewCrafter produced  —  25 frames",
            fill=INK, font=font_bold(23))
    for j, k in enumerate(idx):
        im.paste(Image.fromarray(v[k]).resize((TW, TH), Image.LANCZOS), (TW * j, 66))
        dr.text((TW * j + 8, 66 + TH - 24), f"frame {k}", fill=(255, 255, 255), font=font_bold(16))
    y = 66 + TH + 12
    dr.text((14, y + 4), "2.  VGGT reads all 25 frames at once and returns 3D points  —  in 0.7 seconds",
            fill=INK, font=font_bold(23))
    dr.text((14, y + 34),
            "no camera poses given, no optimisation, no training on this object",
            fill=MUTED, font=font(17))
    x0 = (W - SW * 3) // 2
    for j, sh in enumerate(shots):
        im.paste(sh, (x0 + SW * j, y + 70))
    dr.text((x0 + 10, y + 70 + SH + 6),
            "the recovered cloud, seen from three sides   ·   %s points, each with a colour"
            % format(len(xyz), ","), fill=INK2, font=font(19))
    im.save(OUT / "explain_vggt.png")
    print("wrote explain_vggt.png")


def fig_compare(renderer, mesh, radius, slam_s, vggt_s):
    cols = [("ShapeR's SLAM points", "what the real Aria capture measured  ·  %s points" % format(len(slam_s), ","),
             [(slam_s, SLAM_RGB, 0.006 * DIAG)]),
            ("VGGT's points, from our video", "recovered from the 25 generated frames  ·  %s drawn" % format(len(vggt_s), ","),
             [(vggt_s, VGGT_RGB, 0.004 * DIAG)]),
            ("the two together", "orange should land where teal is",
             [(slam_s, SLAM_RGB, 0.006 * DIAG), (vggt_s, VGGT_RGB, 0.0035 * DIAG)])]
    LAB, CELL_H = 62, 560
    im = Image.new("RGB", (RES * 3, CELL_H + LAB), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for j, (title, sub, clouds) in enumerate(cols):
        dr.text((RES * j + 12, 10), title, fill=INK, font=font_bold(21))
        dr.text((RES * j + 12, 36), sub, fill=INK2, font=font(16))
        im.paste(place(trim(shoot(renderer, mesh, clouds, 25, radius)), RES, CELL_H), (RES * j, LAB))
    im.save(OUT / "explain_compare.png")
    print("wrote explain_compare.png")


def fig_2pct(renderer, mesh, radius, slam_s, dists):
    hit = dists < TAU * DIAG
    LAB, CELL_H = 74, 560
    az = (25, 145, 265)
    im = Image.new("RGB", (RES * len(az), CELL_H + LAB), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((12, 10), "Every SLAM point, coloured by whether VGGT put a point near it",
            fill=INK, font=font_bold(23))
    dr.text((12, 40),
            "green = a VGGT point within 2%% of the object's size  (found)      "
            "red = nothing that close  (missed)      "
            "score = %d of %d = %.3f" % (hit.sum(), len(hit), hit.mean()),
            fill=INK2, font=font(17))
    for j, a in enumerate(az):
        im.paste(place(trim(shoot(renderer, mesh,
                                  [(slam_s[hit], HIT_RGB, 0.007 * DIAG),
                                   (slam_s[~hit], MISS_RGB, 0.007 * DIAG)], a, radius)),
                       RES, CELL_H), (RES * j, LAB))
    im.save(OUT / "explain_2pct.png")
    print("wrote explain_2pct.png")


def fig_pipeline():
    steps = [("3 photos", "of one object,\nbackground removed", "#0E6FB5"),
             ("ViewCrafter", "builds a 3D point cloud,\nflies a chosen route\nround it", "#B5641E"),
             ("a 360° video", "25 frames, the object\nturning all the way\nround", "#0F7A75"),
             ("VGGT", "reads the 25 frames,\nreturns 3D points\nin 0.7 s", "#7C4DA8"),
             ("compare", "against ShapeR's\nown SLAM points —\nhow many found?", "#2E7A34")]
    fig, ax = plt.subplots(figsize=(13.0, 2.9), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
    bw, gap = 16.6, 4.2
    x = 1.0
    for title, sub, col in steps:
        ax.add_patch(FancyBboxPatch((x, 6), bw, 17, boxstyle="round,pad=0.0,rounding_size=1.2",
                                    linewidth=0, facecolor="#F5F7F9"))
        ax.add_patch(FancyBboxPatch((x, 6), 0.55, 17, boxstyle="square,pad=0", linewidth=0,
                                    facecolor=col))
        ax.text(x + 1.6, 19.6, title, fontsize=12.5, fontweight="bold", color="#14181E", va="top")
        ax.text(x + 1.6, 16.0, sub, fontsize=9.4, color="#4A525E", va="top", linespacing=1.5)
        if x + bw + gap < 100:
            ax.add_patch(FancyArrowPatch((x + bw + 0.9, 14.5), (x + bw + gap - 0.9, 14.5),
                                         arrowstyle="-|>", mutation_scale=15,
                                         linewidth=1.6, color="#9AA2AC"))
        x += bw + gap
    ax.text(50, 2.2, "the whole experiment varies exactly one thing: the route in step 2",
            fontsize=10.5, color="#7C8592", ha="center", style="italic")
    fig.savefig(OUT / "explain_pipeline.png", bbox_inches="tight", facecolor="white")
    print("wrote explain_pipeline.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.parse_args()
    mesh, slam, DIAG, to_scene = load(OBJ)
    radius = float(np.linalg.norm(mesh.vertices, axis=1).max()) / (0.60 * np.tan(np.pi / 8))
    rng = np.random.default_rng(0)
    vggt = np.load(PROJ / f"experiment4/outputs/{OBJ}__{COND}__aligned_to_slam.npy").astype(np.float64)
    # Not the saved slam_to_vggt.npy: that one is measured against experiment 3's
    # mesh-aligned cloud, a different registration, and would print a score that
    # disagrees with the deck. Recomputed here against the SLAM-aligned cloud.
    dists, _ = cKDTree(vggt).query(slam)
    draw = vggt[rng.choice(len(vggt), 9000, replace=False)]
    renderer = pyrender.OffscreenRenderer(RES, RES)
    fig_compare(renderer, mesh, radius, to_scene(slam), to_scene(draw))
    fig_2pct(renderer, mesh, radius, to_scene(slam), dists)
    renderer.delete()
    renderer = pyrender.OffscreenRenderer(620, 620)
    fig_vggt(renderer, mesh, radius, to_scene(draw))
    renderer.delete()
    fig_pipeline()
