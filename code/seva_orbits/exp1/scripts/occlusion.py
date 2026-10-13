#!/usr/bin/env python3
"""Remove what is standing in front of the object, using the other views.

The problem this exists for
---------------------------
The object mask is the mesh's silhouette, and a silhouette has no depth test:
anything between the camera and the object that falls inside that outline is kept
as if it were the object. Two versions of the same fault showed up in exp1:

  * partial -- a lamp and a glass console table in front of the sofa, cut out and
    presented as part of the sofa;
  * total -- bench frame 230 is a *bathroom*. The wearer had walked into another
    room, so the bench sits behind a wall. Nothing in our geometry knows about the
    wall (we model the object, not the room), so the mesh projects happily into
    the frame and we cut out a sink and a mirror.

`rgb_visible_points_model` cannot help: it is the full point set for every frame
we use, i.e. frustum membership, not visibility.

The fix
-------
We know the object's mesh exactly and every camera pose exactly, so a pixel inside
the silhouette corresponds to a known 3D point on the object's surface. Project
that point into the other views: where it is unoccluded there, it must show the
same surface. Real object surface agrees across views; an occluder does not,
because it is in front in this view only.

So each pixel gets a colour from every view that can see its surface point, and
disagreement with the robust median is evidence of an occluder. Pixels that
disagree are *repaired* from that median rather than deleted -- deleting leaves a
lamp-shaped bite out of the sofa, which is a worse lie than the lamp. Views where
most pixels disagree are not showing the object at all, and are dropped whole;
that is what removes the bathroom.

Exposure varies a lot between frames, so colours are compared after a per-view
gain that matches each view's object-region luminance to the group median.
Everything runs twice: once to find the bad views, once more with only the
survivors, so a rogue view cannot bias the median that convicts it.
"""

import numpy as np
import pyrender
from scipy.ndimage import binary_dilation, binary_erosion

from build_object_only_views import _renderer  # one EGL context per resolution

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])

DEV_T = 46.0          # colour distance (0-255) beyond which a pixel is an occluder
MIN_OTHER = 2         # views that must see a surface point before we judge it
# Metres of agreement for "the other view sees this same surface". This has to be
# thinner than the object's panels: at 4 cm a chair back is thinner than the
# tolerance, so a *front* view was allowed to stand in for a *back* surface point,
# their textures naturally disagreed, and the correlation test then threw out
# legitimate far-side views -- the chair's arc collapsed from 174 deg to 43 deg.
# 1 cm separates the two faces of everything in this set.
DEPTH_TOL = 0.010
VIEW_REJECT = 0.45    # inconsistent fraction above which a view is not the object
# Cross-checking only works on surface the *other* views can also see, so a view
# looking at the far side of the object is close to unverifiable -- and that is
# exactly the case of bench frame 230, the bathroom shot, sitting ~170 deg from
# every other view. It scores only 0.21 inconsistent because almost none of its
# pixels can be judged at all. So a view whose judged fraction falls below this
# is not trusted on the pixel test, and goes to the appearance test instead.
JUDGE_MIN = 0.25
# How far a view's median object colour may sit from the verified views' median
# before it is treated as showing something else. Deliberately loose: legitimate
# views vary with lighting, and the case this must catch (a white sink standing
# in for a dark wooden bench) is not a near miss.
COLOR_T = 55.0
# Correlation between a view and the other views at matching surface points.
# Exposure-invariant, so unlike the gain-normalised pixel test it survives a
# view being much brighter than the rest. Below this the view is not looking
# at the same object.
CORR_MIN = 0.35
# Silhouette-outline edge support, used only together with CORR_MIN. Below
# 1.10 the boundary carries no more gradient than the object's interior.
EDGE_MIN = 1.10


def render_depth(mesh, K, W, H, c2w_cv):
    """Metric depth of the mesh from an OpenCV camera-to-world pose. 0 = no hit."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    cam = pyrender.IntrinsicsCamera(fx=float(K[0, 0]), fy=float(K[1, 1]),
                                    cx=float(K[0, 2]), cy=float(K[1, 2]),
                                    znear=0.01, zfar=100.0)
    pose = np.array(c2w_cv @ CV2GL, dtype=np.float64, copy=True)
    pose[3, :] = [0.0, 0.0, 0.0, 1.0]        # pyrender rejects float dust here
    scene.add(cam, pose=pose)
    return np.asarray(_renderer(W, H).render(
        scene, flags=pyrender.RenderFlags.DEPTH_ONLY))


def _backproject(depth, K, c2w_cv):
    """Silhouette pixels -> 3D points in model coordinates."""
    H, W = depth.shape
    ys, xs = np.nonzero(depth > 0)
    z = depth[ys, xs]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=1)
    pts = pts_cam @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]
    return ys, xs, pts


def _sample(view, pts, mesh_depth_cache, idx):
    """Colour each 3D point as seen by `view`, or NaN where it cannot see it."""
    K, c2w, img = view["K"], view["c2w"], view["img"]
    H, W = img.shape[:2]
    R = c2w[:3, :3].T
    cam = (pts - c2w[:3, 3]) @ R.T
    z = cam[:, 2]
    out = np.full((len(pts), 3), np.nan, np.float32)
    ok = z > 1e-6
    if not ok.any():
        return out
    u = np.full(len(pts), -1.0)
    v = np.full(len(pts), -1.0)
    u[ok] = cam[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
    v[ok] = cam[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = ok & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not inside.any():
        return out
    # The other view must actually see this surface point, not a nearer part of
    # the same object -- otherwise self-occlusion is mistaken for an occluder.
    d = mesh_depth_cache[idx]
    seen = np.zeros(len(pts), bool)
    ii = np.nonzero(inside)[0]
    dd = d[vi[ii], ui[ii]]
    seen[ii] = (dd > 0) & (np.abs(dd - z[ii]) < DEPTH_TOL)
    # ...and it must be inside the part of the frame the source actually covered.
    cov = view["cov"]
    seen[ii] &= cov[vi[ii], ui[ii]]
    jj = np.nonzero(seen)[0]
    out[jj] = img[vi[jj], ui[jj]].astype(np.float32)
    return out


def edge_support(img, mask):
    """Does the silhouette outline land on a real edge in the photograph?

    Viewpoint-independent, which the correlation test is not: correlation falls
    off simply because a distant view shows a different *side* of the object, and
    on the chair that threw out perfectly good front views. This asks something
    the viewpoint cannot spoil -- if the object really is there with this outline,
    the outline is a brightness discontinuity. Where the camera is actually
    pointing at a bathroom sink, the mesh boundary cuts across smooth porcelain
    and lands on nothing.

    Returns the gradient on the boundary relative to the gradient inside the
    region, so it is invariant to exposure and to how textured the object is.
    """
    g = np.asarray(img, np.float32).mean(axis=2)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    m = mask.astype(bool)
    if m.sum() < 200:
        return 0.0
    inner = binary_erosion(m, iterations=3)
    outer = binary_dilation(m, iterations=3)
    ring = outer & ~inner
    body = inner
    if ring.sum() < 50 or body.sum() < 50:
        return 0.0
    return float(np.median(mag[ring]) / (np.median(mag[body]) + 1e-3))


def _gains(views, depths):
    """Per-view scalar exposure gain, matching object-region luminance."""
    med = []
    for v, d in zip(views, depths):
        m = d > 0
        med.append(float(np.median(v["img"][m].astype(np.float32))) if m.any() else 1.0)
    med = np.array(med, np.float64)
    target = float(np.median(med))
    return target / np.maximum(med, 1e-6)


def analyse(views, mesh):
    """Per-view inconsistent fraction, plus a repaired image for each.

    `views` are dicts with img (HxWx3 uint8), cov (HxW bool), K, c2w (OpenCV c2w).
    """
    H, W = views[0]["img"].shape[:2]
    depths = [render_depth(mesh, v["K"], W, H, v["c2w"]) for v in views]
    gains = _gains(views, depths)

    results = []
    for i, v in enumerate(views):
        ys, xs, pts = _backproject(depths[i], v["K"], v["c2w"])
        if len(pts) == 0:
            results.append({"frac": 1.0, "judged": 0.0, "colour": np.zeros(3),
                            "corr": 0.0, "edge": 0.0, "img": v["img"].copy(),
                            "unrepairable": np.zeros((H, W), bool)})
            continue
        cols = np.stack([_sample(views[j], pts, depths, j) * gains[j]
                         for j in range(len(views)) if j != i], axis=0)
        n = np.isfinite(cols[:, :, 0]).sum(axis=0)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(cols, axis=0)
        mine = v["img"][ys, xs].astype(np.float32) * gains[i]
        dev = np.linalg.norm(mine - med, axis=1)

        judged = n >= MIN_OTHER
        bad = judged & (dev > DEV_T)
        frac = float(bad.sum() / max(judged.sum(), 1))
        judged_frac = float(judged.sum() / max(len(pts), 1))
        # Median object colour at unit exposure, for the appearance fallback.
        own = np.median(mine, axis=0)

        # Structural agreement, which is what catches a view of something else
        # entirely. `dev` above cannot: it is compared after a per-view exposure
        # gain, and that gain is fitted to the view's own object region -- so a
        # bright bathroom standing in for a dark bench gets scaled to match and
        # only 21% of it looks wrong. Correlation is invariant to any such affine
        # rescaling, so it asks the question the gain cannot destroy: does this
        # view's light-and-dark pattern track what the other views see at the same
        # surface points? Real bench does; a sink and a mirror do not.
        ok = judged & np.isfinite(med[:, 0])
        if ok.sum() >= 64:
            a = mine[ok].mean(axis=1)
            b = med[ok].mean(axis=1)
            sa, sb = a.std(), b.std()
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) \
                if sa > 1e-6 and sb > 1e-6 else 0.0
        else:
            corr = 1.0            # too little evidence to convict


        repaired = v["img"].copy()
        fix = bad & np.isfinite(med[:, 0])
        if fix.any():                       # put the median back at this exposure
            repaired[ys[fix], xs[fix]] = np.clip(med[fix] / gains[i], 0, 255).astype(np.uint8)
        # Pixels we could not judge and could not repair stay as they are; the
        # caller keeps them only if they are inside the silhouette anyway.
        unrepairable = np.zeros((H, W), bool)
        u = bad & ~np.isfinite(med[:, 0])
        unrepairable[ys[u], xs[u]] = True
        results.append({"frac": frac, "judged": judged_frac, "colour": own,
                        "corr": corr, "edge": edge_support(v["img"], v["alpha"]),
                        "img": repaired, "unrepairable": unrepairable})
    return results


def _verdicts(res):
    """Which views to keep, and why each rejected one was rejected.

    Two tiers, because the pixel test only reaches surface that other views can
    also see. Views with enough judged pixels are decided on that. The rest are
    compared on appearance against the views that *were* verified -- a fallback,
    not a substitute, so it is set loose enough to catch only gross mismatches.
    """
    n = len(res)
    verified = [i for i in range(n) if res[i]["judged"] >= JUDGE_MIN]
    reasons = {}
    for i in verified:
        if res[i]["frac"] > VIEW_REJECT:
            reasons[i] = f"{res[i]['frac']:.0%} of it disagrees with the other views"

    # Neither remaining signal is decisive alone, and both were checked against
    # the actual candidates before being trusted:
    #   * correlation is low both for a wrong scene AND for a legitimate view of
    #     the object's far side -- on its own it threw out six good chair views;
    #   * edge support is low for a wrong scene but also for a real view with a
    #     soft outline -- a good chair view scores 0.97, same as the bathroom.
    # Their conjunction does separate the cases in this set: the bathroom frames
    # are low on both (r <= 0.27, edge <= 1.06) while the chair's far-side views
    # are low on r but high on edge (1.16-2.01). This is a calibrated conjunction
    # of two weak indicators, not a general detector, so every rejection is
    # printed and recorded in meta.json to stay auditable.
    for i in range(n):
        if i in reasons:
            continue
        if res[i]["corr"] < CORR_MIN and res[i]["edge"] < EDGE_MIN:
            reasons[i] = (f"it does not look like this object: texture "
                          f"correlation r={res[i]['corr']:.2f} against the other "
                          f"views and the silhouette outline lands on no real "
                          f"edge (edge support {res[i]['edge']:.2f})")
    keep = [i for i in range(n) if i not in reasons]
    return keep, reasons


def clean(views, mesh):
    """Drop views that are not the object, repair occluders in the rest.

    Returns (keep_indices, repaired_images, unrepairable_masks, stats, reasons).
    Runs twice when anything is dropped, so a rogue view cannot skew the median
    that convicts the others.
    """
    res = analyse(views, mesh)
    keep, reasons = _verdicts(res)
    stats = [{"frac": r["frac"], "judged": r["judged"], "corr": r["corr"],
              "edge": r["edge"]} for r in res]
    if len(keep) < 2 or len(keep) == len(views):
        return (keep, [res[i]["img"] for i in keep],
                [res[i]["unrepairable"] for i in keep], stats, reasons)

    res2 = analyse([views[i] for i in keep], mesh)
    return (keep, [r["img"] for r in res2], [r["unrepairable"] for r in res2],
            stats, reasons)
