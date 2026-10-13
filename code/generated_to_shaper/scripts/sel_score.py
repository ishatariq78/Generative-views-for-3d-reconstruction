#!/usr/bin/env python3
"""Score every generated orbit frame for reliability, using only inference-time information.

    ~/.conda/envs/shaper/bin/python scripts/sel_score.py --objects <obj> ...

No reference mesh, no reference error, no known outcome. Three signals, all computable from
the capture's own data and the generated video:

  agree   of the object's real SLAM points that project into this frame, the fraction that
          land on the rendered object. This is agreement with real observation. It is only
          meaningful where observations support it, so points behind the camera or outside
          the frame are excluded, and a frame with too few projected points is marked
          unverified rather than wrong.
  intact  the object is one thing and nothing else has appeared: the share of the subject
          mask held by its largest connected component, times a penalty if the object's
          area departs from the orbit's median area. Catches a frame that has sprouted
          extra geometry or lost the object.
  spread  viewpoint diversity, as angular distance from the frames already chosen. Used
          greedily so the set does not collapse onto near-duplicates.

`reliability_order` ranks frames by agree x intact, then fills greedily with a spread term.
`diversity_order` uses spread alone, as the control that says whether the reliability part
earns its place. One shared rule for every object; nothing is tuned per object.
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

E = Path("/home/isha.tariq/seva_to_shaper")
DATA = Path("/home/isha.tariq/shapeR_testing/data")
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
MIN_PTS = 20            # fewer projected points than this: unverified, not wrong
UNVERIFIED = 0.5        # neutral agreement for an unverifiable frame


def subject_mask(rgb):
    a = np.asarray(rgb, np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > 0.15) | (mx < 0.70)


def score_object(obj):
    d = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
    slam = d["points_model"].numpy().astype(np.float64)
    tf = json.loads((E / "inputs_full" / obj / "transforms.json").read_text())["frames"]
    split = next((E / "inputs_full" / obj).glob("train_test_split_*.json"))
    ids = json.loads(split.read_text())["test_ids"]
    frames = sorted((E / "outputs_full/demo/img2trajvid" / obj / "samples-rgb").glob("*.png"))
    if len(frames) != len(ids):
        return None, f"{len(frames)} frames vs {len(ids)} target cameras"

    rec = []
    for k, f in enumerate(frames):
        fr = tf[ids[k]]; w, h = int(fr["w"]), int(fr["h"])
        fl, cx, cy = fr["fl_x"] * w, fr["cx"] * w, fr["cy"] * h
        c2w = np.asarray(fr["transform_matrix"], np.float64) @ GL2CV
        m = subject_mask(Image.open(f).convert("RGB"))
        P = (slam - c2w[:3, 3]) @ c2w[:3, :3]
        u = cx + fl * P[:, 0] / np.maximum(P[:, 2], 1e-9)
        v = cy + fl * P[:, 1] / np.maximum(P[:, 2], 1e-9)
        ok = (P[:, 2] > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() >= MIN_PTS:
            agree = float(m[v[ok].astype(int), u[ok].astype(int)].mean()); verified = True
        else:
            agree = UNVERIFIED; verified = False
        lab, n = ndimage.label(m)
        big = float(np.bincount(lab.ravel())[1:].max() / max(m.sum(), 1)) if n else 0.0
        az = float(np.degrees(np.arctan2(c2w[1, 3], c2w[0, 3])) % 360.0)
        rec.append({"frame": k, "agree": agree, "verified": verified, "largest_cc": big,
                    "area": float(m.mean()), "azimuth": az, "n_points": int(ok.sum())})

    areas = np.array([r["area"] for r in rec]); med = float(np.median(areas))
    for r in rec:
        dev = abs(r["area"] - med) / max(med, 1e-6)
        r["area_ok"] = float(np.clip(1.0 - dev, 0.0, 1.0))
        r["intact"] = r["largest_cc"] * r["area_ok"]
        r["reliability"] = r["agree"] * r["intact"]

    def greedy(key_fn):
        chosen, rest = [], list(range(len(rec)))
        while rest:
            best, bi = -1e9, rest[0]
            for i in rest:
                if chosen:
                    da = min(min(abs(rec[i]["azimuth"] - rec[j]["azimuth"]),
                                 360 - abs(rec[i]["azimuth"] - rec[j]["azimuth"])) for j in chosen)
                else:
                    da = 180.0
                s = key_fn(rec[i], da / 180.0)
                if s > best:
                    best, bi = s, i
            chosen.append(bi); rest.remove(bi)
        return chosen

    rel = greedy(lambda r, sp: r["reliability"] + 0.3 * sp)
    div = greedy(lambda r, sp: sp)
    return {"object": obj, "frames": rec, "reliability_order": rel, "diversity_order": div,
            "median_area": med,
            "mean_agree": float(np.mean([r["agree"] for r in rec])),
            "verified_frames": int(sum(r["verified"] for r in rec))}, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--objects", nargs="*", required=True)
    a = ap.parse_args()
    out = E / "sel/scores"; out.mkdir(parents=True, exist_ok=True)
    for o in a.objects:
        try:
            s, err = score_object(o)
        except Exception as e:
            print(f"!! {o}: {type(e).__name__}: {e}"); continue
        if err:
            print(f"!! {o}: {err}"); continue
        (out / f"{o}.json").write_text(json.dumps(s))
        r = np.array([x["reliability"] for x in s["frames"]])
        print(f"{o[:32]:32s} agree {s['mean_agree']:.3f}  verified {s['verified_frames']}/80  "
              f"reliability {r.min():.2f}-{r.max():.2f}  top5 frames {s['reliability_order'][:5]}", flush=True)
