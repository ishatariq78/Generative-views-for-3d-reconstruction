"""Build every figure in assets/ from the experiments' own outputs.

    ~/.conda/envs/shaper/bin/python scripts/make_figures.py [--only teaser gallery ...]

Runs on CPU (pyrender through EGL; the login node's software EGL is enough). Reads only:
generated frames/videos, aligned point clouds, predicted .glb meshes and their metric JSONs,
and the tables in results/ (run scripts/build_results.py first). Nothing is re-generated.

Every metric printed on a mesh panel is read from the JSON written next to that exact mesh
(one run, one seed) -- never an aggregate. Captions say which file.
"""
import argparse
import csv
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
A = REPO / "assets"
RES = REPO / "results"
VC = HOME / "viewcrafter_orbits"
SV = HOME / "stable_virtual_camera"
S2S = HOME / "seva_to_shaper"
DATA = HOME / "shapeR_testing/data"

# ---- one visual language for every figure ------------------------------------
INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
C_REF = "#1baf7a"      # reference mesh
C_REAL = "#8f8d88"     # real-only input
C_VIEWS = "#2a78d6"    # real + generated images (pathway 2)
C_PTS = "#eb6834"      # points reconstructed from generated video (pathway 1)
C_GOOD, C_BAD = "#2a78d6", "#e34948"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "legend.frameon": False,
})
TILE = 300
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 17)
    FONT_B = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    FONT_S = ImageFont.truetype("DejaVuSans.ttf", 14)
except OSError:
    FONT = FONT_B = FONT_S = ImageFont.load_default()


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


# ---- rendering ------------------------------------------------------------------
def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def frame_of(ref):
    """Camera framing taken from the REFERENCE mesh, so every mesh of one object is
    rendered from exactly the same camera and at the same scale."""
    c = ref.bounds.mean(0)
    d = float(np.linalg.norm(ref.bounds[1] - ref.bounds[0]))
    return c, d


def render(mesh, colour, frame, az=30, el=0.48, n=TILE):
    c, d = frame
    a = np.radians(az)
    eye = c + d * 1.55 * np.array([np.cos(a) * 0.84, np.sin(a) * 0.84, el])
    sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.38] * 3)
    m = mesh.copy()
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(hex_rgb(colour)) * 255), 255], np.uint8))
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    pose = look_at(eye, c)
    sc.add(pyrender.PerspectiveCamera(yfov=np.radians(40)), pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=3.6), pose=pose)
    r = pyrender.OffscreenRenderer(n, n)
    rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    al = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8)


def load_mesh(p):
    m = trimesh.load(p, force="mesh", process=False)
    return m


def tile_img(path_or_arr, n=TILE):
    im = Image.open(path_or_arr).convert("RGB") if not isinstance(path_or_arr, np.ndarray) else Image.fromarray(path_or_arr)
    im.thumbnail((n, n), Image.LANCZOS)
    cv = Image.new("RGB", (n, n), (255, 255, 255))
    cv.paste(im, ((n - im.width) // 2, (n - im.height) // 2))
    return np.asarray(cv)


def tile_rect(arr, w, h):
    im = Image.fromarray(arr) if isinstance(arr, np.ndarray) else Image.open(arr).convert("RGB")
    im.thumbnail((w, h), Image.LANCZOS)
    cv = Image.new("RGB", (w, h), (255, 255, 255))
    cv.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return np.asarray(cv)


def labelled(tile, top, bottom=None, colour=INK, n=TILE):
    """Tile with a caption above and an optional metric line below."""
    h_top, h_bot = 30, (26 if bottom else 0)
    im = Image.new("RGB", (n, n + h_top + h_bot), (255, 255, 255))
    im.paste(Image.fromarray(tile), (0, h_top))
    dr = ImageDraw.Draw(im)
    if top:
        dr.rectangle([0, 0, 6, 22], fill=colour)
        dr.text((12, 3), top, fill=INK, font=FONT)
    if bottom:
        dr.text((12, n + h_top + 4), bottom, fill=INK2, font=FONT_S)
    return np.asarray(im)


def hcat(tiles, gap=10):
    h = max(t.shape[0] for t in tiles)
    out = []
    for i, t in enumerate(tiles):
        if t.shape[0] < h:
            t = np.concatenate([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)], 0)
        out.append(t)
        if i < len(tiles) - 1:
            out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.concatenate(out, 1)


def vcat(rows, gap=14):
    w = max(r.shape[1] for r in rows)
    out = []
    for i, r in enumerate(rows):
        if r.shape[1] < w:
            r = np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)], 1)
        out.append(r)
        if i < len(rows) - 1:
            out.append(np.full((gap, w, 3), 255, np.uint8))
    return np.concatenate(out, 0)


def heading(text, width, sub=None):
    h = 34 + (22 if sub else 0)
    im = Image.new("RGB", (width, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.text((2, 4), text, fill=INK, font=FONT_B)
    if sub:
        dr.text((2, 32), sub, fill=INK2, font=FONT_S)
    return np.asarray(im)


def save(arr, name, quality=88):
    p = A / name
    p.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    if p.suffix == ".jpg":
        im.save(p, quality=quality, optimize=True)
    else:
        im.save(p, optimize=True)
    print("wrote", p.relative_to(REPO), im.size, f"{p.stat().st_size / 1e3:.0f} kB")


def metrics(p):
    m = json.load(open(p))["metrics"]
    return m["chamfer"] * 100, m["f-score"]


def mline(cd, f1):
    return f"CD {cd:.2f}   F1 {f1:.3f}"


def orbit_frames(root, obj, k):
    fs = sorted((Path(root) / obj / "samples-rgb").glob("*.png"))
    idx = np.linspace(0, len(fs) - 1, k + 1)[:-1].round().astype(int) if k < len(fs) else range(len(fs))
    return [fs[i] for i in idx]


def video_frames(mp4, k):
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, n - 1, k).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


# ---- figures ----------------------------------------------------------------------
def fig_teaser():
    """Bar stool: the same generated orbit, used two ways. Seed-0 runs from outputs/seeded/s0."""
    obj = "BNB1045__barstool"
    md = S2S / "outputs/seeded/s0/per_object"
    ref = load_mesh(md / f"GT__{obj}__slam.glb")
    fr = frame_of(ref)
    ins = sorted((S2S / "inputs" / obj / "images").glob("*.png"))
    real = [labelled(tile_img(ins[i], 200), "Real views (cut out)" if j == 0 else "", None, INK, n=200) for j, i in enumerate(np.linspace(0, len(ins) - 1, 3).round().astype(int))]
    gen = [labelled(tile_img(f, 200), "Generated orbit (90/180/270 deg)" if j == 0 else "", None, C_VIEWS, n=200) for j, f in enumerate(orbit_frames(S2S / "outputs/demo/img2trajvid", obj, 4)[1:])]
    left = vcat([hcat(real, 6), hcat(gen, 6)], 8)
    cs, fs_ = metrics(md / f"{obj}__slam.json")
    cp, fp = metrics(md / f"{obj}__pi3.json")
    cv_, fv = metrics(md / f"{obj}__views.json")
    meshes = [
        labelled(render(ref, C_REF, fr), "Reference mesh", None, C_REF),
        labelled(render(load_mesh(md / f"{obj}__slam.glb"), C_REAL, fr), "Real views only", mline(cs, fs_), C_REAL),
        labelled(render(load_mesh(md / f"{obj}__pi3.glb"), C_PTS, fr), "Orbit as 3D points", mline(cp, fp), C_PTS),
        labelled(render(load_mesh(md / f"{obj}__views.glb"), C_VIEWS, fr), "Orbit as extra views", mline(cv_, fv), C_VIEWS),
    ]
    right = hcat(meshes, 10)
    body = hcat([left, np.full((left.shape[0], 24, 3), 255, np.uint8), right], 0) if left.shape[0] >= right.shape[0] else None
    if body is None:
        pad = np.full((right.shape[0] - left.shape[0], left.shape[1], 3), 255, np.uint8)
        body = hcat([np.concatenate([left, pad], 0), np.full((right.shape[0], 24, 3), 255, np.uint8), right], 0)
    head = heading("One generated orbit, two ways to use it  -  BNB1045__barstool",
                   body.shape[1], "ShapeR meshes, seed 0, 16 views, one run each (outputs/seeded/s0). CD x10^2 lower is better; F1 higher is better.")
    save(vcat([head, body], 6), "teaser.jpg")


def fig_orbit_gif():
    """The five SEVA object orbits side by side, every 4th of 80 frames."""
    objs = ["ADT1292__chair", "BNB1045__sofa", "BNB1723__bed", "ADT1292__woodentable", "BNB2727__bench"]
    root = SV / "exp1/outputs/demo/img2trajvid"
    seqs = [[tile_img(f, 220) for f in sorted((root / o / "samples-rgb").glob("*.png"))[::4]] for o in objs]
    frames = [Image.fromarray(hcat([s[i] for s in seqs], 4)) for i in range(min(map(len, seqs)))]
    p = A / "gifs/seva_orbits.gif"
    p.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)
    print("wrote", p.relative_to(REPO), frames[0].size, f"{p.stat().st_size / 1e3:.0f} kB")


def fig_full_vs_limited():
    """Same object (sofa), same real cut-out view: ViewCrafter 360, ViewCrafter +-45, SEVA 360 (8 views)."""
    obj = "BNB1045__sofa"
    vc360 = video_frames(VC / f"experiment2/outputs/{obj}__cutout__orbit360/diffusion0.mp4", 6)
    vc45 = video_frames(VC / f"experiment2/outputs/{obj}__cutout__orbit45/diffusion0.mp4", 6)
    sv = [np.asarray(Image.open(f).convert("RGB")) for f in orbit_frames(SV / "exp1/outputs/demo/img2trajvid", obj, 6)]
    rows = []
    for name, seq, col, note in (("ViewCrafter, 1 view, full 360 deg orbit", vc360, C_PTS, "frames 0-24 of 25, evenly spaced"),
                                 ("ViewCrafter, 1 view, limited +-45 deg sweep", vc45, C_REAL, "frames 0-24 of 25, evenly spaced"),
                                 ("SEVA, 8 posed views, full 360 deg orbit", sv, C_VIEWS, "frames 0-66 of 80, every 60 deg")):
        tiles = [tile_rect(f, 210, 118 if name.startswith("ViewCrafter") else 210) for f in seq]
        strip = hcat(tiles, 6)
        bar = Image.new("RGB", (strip.shape[1], 30), (255, 255, 255))
        d = ImageDraw.Draw(bar)
        d.rectangle([0, 4, 6, 26], fill=col)
        d.text((12, 6), name, fill=INK, font=FONT)
        d.text((strip.shape[1] - 290, 8), note, fill=MUTED, font=FONT_S)
        rows.append(vcat([np.asarray(bar), strip], 2))
    head = heading("Full orbit vs limited sweep  -  BNB1045__sofa",
                   rows[0].shape[1], "ViewCrafter repaints the unobserved half as a room; +-45 deg stays on the object; SEVA keeps the object across 360 deg.")
    save(vcat([head] + rows, 10), "full_orbit_vs_sweep.jpg")


def fig_false_content():
    """Magenta ball painted into one input view vs into every input view (sofa and bed)."""
    rows = []
    root = SV / "exp2/outputs/demo/img2trajvid"
    oc = {r["scene"]: r for r in json.load(open(SV / "exp2/outputs/occlusion_scores.json"))}
    for obj, k in (("BNB1045__sofa", 56), ("BNB1723__bed", 20)):
        tiles = []
        for cond, lab, col in (("occ1", "one view painted", C_REAL), ("occall", "all views painted", C_PTS)):
            ins = sorted((SV / "exp2/inputs" / f"{obj}__{cond}" / "images").glob("*.png"))
            meta = json.load(open(root / f"{obj}__{cond}/input/meta.json")) if (root / f"{obj}__{cond}/input/meta.json").exists() else {}
            painted = meta.get("painted_views") or meta.get("occluded_views")
            src = ins[painted[0]] if isinstance(painted, list) and painted and isinstance(painted[0], int) else None
            if src is None:   # find a painted input by its magenta pixels
                def mag(p):
                    a = np.asarray(Image.open(p).convert("RGB")).astype(int)
                    return ((a[..., 0] > 200) & (a[..., 1] < 80) & (a[..., 2] > 200)).sum()
                src = max(ins, key=mag)
            out = sorted((root / f"{obj}__{cond}/samples-rgb").glob("*.png"))[k]
            r = oc[f"{obj}__{cond}"]
            tiles.append(labelled(tile_img(src, 230), f"Input ({lab})", None, col, n=230))
            tiles.append(labelled(tile_img(out, 230), f"Generated frame {k}",
                                  f"copied in {r['copied']}/{r['frames_with_ball_in_view']} frames", col, n=230))
        clean = sorted((SV / "exp1/outputs/demo/img2trajvid" / obj / "samples-rgb").glob("*.png"))[k]
        tiles.insert(0, labelled(tile_img(clean, 230), f"Clean run, frame {k}", None, C_VIEWS, n=230))
        rows.append(hcat(tiles, 8))
    head = heading("False content: one view vs every view  -  SEVA, 8 posed input views",
                   rows[0].shape[1], "A magenta ball (absent from every scene) is painted with true geometry into the inputs. Counts per object, 80 frames.")
    save(vcat([head] + rows, 12), "false_content.jpg")


def _proj(P, az, el):
    a, e = np.radians(az), np.radians(el)
    R1 = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, np.cos(e), np.sin(e)], [0, -np.sin(e), np.cos(e)]])
    Q = P @ R1.T @ R2.T
    return Q[:, 0], Q[:, 2], Q[:, 1]


def fig_clouds():
    """Same SEVA orbit (80 frames) through VGGT, Pi3 and KV-Tracker, over ShapeR's SLAM points."""
    objs = [("ADT1292__chair", 25), ("BNB1723__bed", 35)]
    conds = [("seva360f", "VGGT, 80 frames"), ("seva360fp", "Pi3, 80 frames"), ("seva360fk1", "KV-Tracker, keyframe every frame")]
    rows = {r["object"] + "|" + r["condition"]: r for r in csv.DictReader(open(RES / "C_video_to_3d_per_object.csv"))}
    fig, axes = plt.subplots(len(objs), 4, figsize=(13.2, 6.6), sharex="row", sharey="row")
    rng = np.random.default_rng(0)
    for i, (obj, az) in enumerate(objs):
        s = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
        slam = s["points_model"].numpy()
        V = np.asarray(s["mesh_vertices"], float)
        lo, hi = V.min(0), V.max(0)
        ax = axes[i, 0]
        x, y, dep = _proj(slam, az, 20)
        ax.scatter(x, y, s=1.2, c=INK, linewidths=0)
        ax.text(0.0, 0.02, f"{len(slam)} SLAM points", transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        ax.set_title("Capture's SLAM points" if i == 0 else "", fontsize=10, color=INK, loc="left")
        ax.text(0.0, 0.98, obj, transform=ax.transAxes, fontsize=10, color=INK, fontweight="bold", va="top")
        for j, (c, lab) in enumerate(conds, 1):
            P = np.load(SV / f"exp5/outputs/{obj}__{c}__aligned_to_slam.npz")["points"]
            P = P[rng.choice(len(P), 25000, replace=False)]
            keep = np.all((P > lo - 0.3 * (hi - lo)) & (P < hi + 0.3 * (hi - lo)), 1)
            x, y, dep = _proj(P[keep], az, 20)
            o = np.argsort(dep)
            axes[i, j].scatter(x[o], y[o], s=0.6, c=C_VIEWS, alpha=0.35, linewidths=0)
            xs, ys, _ = _proj(slam, az, 20)
            axes[i, j].scatter(xs, ys, s=0.8, c=INK, linewidths=0)
            r = rows[f"{obj}|{c}"]
            axes[i, j].set_title(lab if i == 0 else "", fontsize=10, color=INK, loc="left")
            axes[i, j].text(0.02, 0.02, f"SLAM reached {float(r['slam_points_reached_2pct']):.2f}\ncloud->SLAM {float(r['cloud_to_slam_mean_pct_diag']):.1f}% diag",
                            transform=axes[i, j].transAxes, fontsize=9, color=INK2, va="bottom")
        for ax in axes[i]:
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        xs, ys, _ = _proj(slam, az, 20)
        pad = 0.25 * max(np.ptp(xs), np.ptp(ys))
        for ax in axes[i]:
            ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad)
    fig.suptitle("Generated orbit -> 3D: reconstructed clouds (blue) over the capture's SLAM points (black)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Same 80 SEVA frames per object; upright camera-anchored fit; one run per reconstructor. Source: results/C_video_to_3d_per_object.csv.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    p = A / "clouds_vggt_pi3_kv.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p.relative_to(REPO))


def fig_video_to_3d_chart():
    m = {r["condition"]: r for r in csv.DictReader(open(RES / "C_video_to_3d_means.csv"))}
    order = [("gen45", "ViewCrafter 1 view, +-45 deg", C_REAL), ("gen360", "ViewCrafter 1 view, 360 deg", C_REAL),
             ("v3_360", "ViewCrafter 3 views, 360 deg", C_REAL), ("sparse", "ViewCrafter 8 views (interp.)", C_REAL),
             ("real", "Real photos, 25 frames", INK),
             ("seva360", "SEVA -> VGGT, 25 frames", C_VIEWS), ("seva360f", "SEVA -> VGGT, 80 frames", C_VIEWS),
             ("seva360p", "SEVA -> Pi3, 25 frames", C_VIEWS), ("seva360fp", "SEVA -> Pi3, 80 frames", C_VIEWS),
             ("seva360fk10", "SEVA -> KV-Tracker, kf every 10", C_VIEWS), ("seva360fk5", "SEVA -> KV-Tracker, kf every 5", C_VIEWS),
             ("seva360fk1", "SEVA -> KV-Tracker, kf every frame", C_VIEWS)]
    per = list(csv.DictReader(open(RES / "C_video_to_3d_per_object.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, key, xl in ((axes[0], "slam_points_reached_2pct", "SLAM points reached within 2% of diagonal  (higher = better)"),
                        (axes[1], "cloud_to_slam_mean_pct_diag", "Mean cloud -> SLAM distance, % of diagonal  (lower = less haze)")):
        for yi, (c, lab, col) in zip(y, order):
            v = float(m[c][key])
            ax.barh(yi, v, height=0.62, color=col, alpha=0.9 if col != C_REAL else 0.75)
            pts = [float(r[key]) for r in per if r["condition"] == c]
            ax.scatter(pts, [yi] * len(pts), s=12, color="white", edgecolors=INK2, linewidths=0.7, zorder=3)
            ax.text(0.01 * ax.get_xlim()[1] if False else v * 0.02, yi, f"{v:.2f}" if key.startswith("slam") else f"{v:.1f}", fontsize=8.5,
                    color="white", va="center", ha="left", fontweight="bold")
        ax.set_title(xl, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([o[1] for o in order], fontsize=9.5)
    fig.suptitle("Generated video -> 3D, scored against the capture's SLAM points (5 objects; bar = mean, dots = objects)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "One-sided measures: SLAM points cover 17-68% of each surface, so neither panel is overall reconstruction accuracy. Source: results/C_video_to_3d_means.csv.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(A / "chart_video_to_3d.png", dpi=110); plt.close(fig)
    print("wrote assets/chart_video_to_3d.png")


def fig_budget_chart():
    rows = list(csv.DictReader(open(RES / "D_view_budget_6obj.csv")))
    S = json.load(open(RES / "summary.json"))["D_view_budget"]
    budgets = [8, 12, 16, 24, 32]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, agg, lab in ((axes[0], "d_cd", "d_cd_mean", "Paired change in CD x10^2 vs real-only (negative = better)"),
                              (axes[1], "d_f1", "d_f1_mean", "Paired change in F1 vs real-only (positive = better)")):
        ax.axhline(0, color=MUTED, lw=1)
        for cond, col, name, dx in (("views", C_VIEWS, "real + 16 generated views", -0.5), ("vgap", C_PTS, "+ generated gap-fill points", 0.5)):
            mean = [S[f"{cond}@{b}"][agg] for b in budgets]
            ax.plot(budgets, mean, color=col, lw=2, marker="o", ms=8, mec="white", mew=2, label=name, zorder=3)
            for b in budgets:
                v = [float(r[key]) for r in rows if r["condition"] == cond and int(r["budget"]) == b]
                ax.scatter([b + dx] * len(v), v, s=10, color=col, alpha=0.35, linewidths=0)
        ax.set_xticks(budgets); ax.set_xlabel("Views ShapeR consumes (budget)")
        ax.set_title(lab, fontsize=10, color=INK, loc="left")
    axes[0].set_ylim(-1.5, 2.5); axes[0].text(8, 2.35, "points beyond 2.5 not shown", fontsize=8, color=MUTED)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("View budget: the effect of adding generated views depends on how many views ShapeR consumes (6 objects, 3 seeds)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Line = mean over 6 objects of the per-object mean paired difference; dots = objects. Small sample: 6 objects. Source: results/D_view_budget_6obj.csv.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(A / "chart_view_budget.png", dpi=110); plt.close(fig)
    print("wrote assets/chart_view_budget.png")


def fig_155_chart():
    rows = list(csv.DictReader(open(RES / "D_views_into_shaper_155obj.csv")))
    S = json.load(open(RES / "summary.json"))["D_views_155"]
    d = np.array([float(r["d_cd"]) for r in rows]); f = np.array([float(r["d_f1"]) for r in rows])
    o = np.argsort(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    cols = np.where(d[o] < 0, C_GOOD, C_BAD)
    ax.bar(np.arange(len(d)), np.clip(d[o], -2, 2), color=cols, width=0.8)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-2.1, 2.1); ax.set_xlim(-1, len(d))
    ax.set_xlabel("155 objects, sorted"); ax.set_xticks([])
    ax.set_title("Paired change in CD x10^2, real + generated vs real-only (clipped at +-2)", fontsize=10, color=INK, loc="left")
    ax.text(2, 1.75, f"better on {S['objects_cd_better']} / {S['n_objects']} objects\nmean {S['d_cd_mean']:+.3f} (SE {S['d_cd_se_objects']:.3f}), median {S['d_cd_median']:+.3f}",
            fontsize=9.5, color=INK2, va="top")
    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    agree = (d < 0) == (f > 0)
    ax.scatter(d[agree], f[agree], s=16, color=C_REAL, alpha=0.7, linewidths=0, label="CD and F1 agree")
    ax.scatter(d[~agree], f[~agree], s=26, color=C_PTS, edgecolors="white", linewidths=0.8, label="CD and F1 disagree")
    ax.set_xlim(-3.5, 2.5); ax.set_ylim(-0.15, 0.2)
    ax.set_xlabel("change in CD x10^2"); ax.set_ylabel("change in F1")
    ax.set_title(f"CD better but F1 worse: {S['objects_cd_better_f1_worse']}; CD worse but F1 better: {S['objects_cd_worse_f1_better']}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Real + generated views into ShapeR at a 24-view budget, 155 objects (per object: mean of 3 seeds)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, f"Sign test over objects (descriptive): one-sided p = {S['sign_test_p_one_sided_cd']:.2f}. Source: results/D_views_into_shaper_155obj.csv.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(A / "chart_155_paired.png", dpi=110); plt.close(fig)
    print("wrote assets/chart_155_paired.png")


def fig_gallery():
    """Success / CD-F1 disagreement / regression, from the seed-0 mesh re-run (outputs_sub/meshes24)."""
    md = S2S / "outputs_sub/meshes24/per_object"
    cases = [("HWD0045__wardrobe", "Success: CD and F1 both improve (in this run and in the 3-seed mean)"),
             ("BNB2909__armchair_2", "Disagreement in this run: CD improves, F1 worsens (the 3-seed mean improves both)"),
             ("HWD0045__ottoman", "Regression: CD and F1 both worsen (in this run and in the 3-seed mean)")]
    mean = {r["object"]: r for r in csv.DictReader(open(RES / "D_views_into_shaper_155obj.csv"))}
    rows = []
    for obj, title in cases:
        ref = load_mesh(md / f"GT__{obj}__slam.glb")
        fr = frame_of(ref)
        gen = orbit_frames(S2S / "outputs_full/demo/img2trajvid", obj, 2)
        cs, fs_ = metrics(md / f"{obj}__slam.json")
        cv_, fv = metrics(md / f"{obj}__views.json")
        sl, vw = load_mesh(md / f"{obj}__slam.glb"), load_mesh(md / f"{obj}__views.glb")
        tiles = [labelled(tile_img(gen[1], 240), "Generated frame", None, C_VIEWS, n=240)]
        for az, side in ((30, "front"), (210, "back")):
            tiles += [labelled(render(ref, C_REF, fr, az, n=240), f"Reference ({side})", None, C_REF, n=240),
                      labelled(render(sl, C_REAL, fr, az, n=240), f"Real only ({side})", mline(cs, fs_) if az == 30 else None, C_REAL, n=240),
                      labelled(render(vw, C_VIEWS, fr, az, n=240), f"+ generated ({side})", mline(cv_, fv) if az == 30 else None, C_VIEWS, n=240)]
        strip = hcat(tiles, 6)
        r = mean[obj]
        sub = (f"This run (seed 0): CD {cs:.2f} -> {cv_:.2f}, F1 {fs_:.3f} -> {fv:.3f}.   "
               f"Mean of 3 seeds (scored runs): CD {float(r['base_cd']):.2f} -> {float(r['cd']):.2f}, F1 {float(r['base_f1']):.3f} -> {float(r['f1']):.3f}.")
        rows.append(vcat([heading(f"{title}  -  {obj}", strip.shape[1], sub), strip], 0))
    head = heading("Real + generated views into ShapeR: success, disagreement, regression",
                   rows[0].shape[1], "24-view budget. Each metric belongs to the mesh shown: seed 0, one run (outputs_sub/meshes24). Back views show the side the capture rarely saw.")
    save(vcat([head] + rows, 14), "gallery_shaper_views.jpg")


def fig_observability():
    """Crop of exp2's own observability figure (scripts/observability.py): sofa and bed rows."""
    im = Image.open(SV / "exp2/figs/observability_vs_output.png").convert("RGB")
    crop = np.asarray(im.crop((0, 306, im.width, 922)))
    head = heading("Observed vs never-photographed surface  -  SEVA orbits, 8 posed input views",
                   crop.shape[1], "Green = seen by at least one input view (in frame, front-facing, depth-tested); orange = never photographed. Below: the generated frame.")
    save(vcat([head, crop], 4), "observed_vs_unseen.jpg")


FIGS = {"observe": fig_observability, "teaser": fig_teaser, "gif": fig_orbit_gif, "sweep": fig_full_vs_limited, "false": fig_false_content,
        "clouds": fig_clouds, "chart3d": fig_video_to_3d_chart, "budget": fig_budget_chart, "c155": fig_155_chart,
        "gallery": fig_gallery}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    a = ap.parse_args()
    A.mkdir(exist_ok=True)
    for k in a.only:
        FIGS[k]()
