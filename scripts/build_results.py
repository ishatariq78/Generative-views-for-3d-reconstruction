"""Rebuild every table in results/ from the original experiment outputs.

    python3 scripts/build_results.py            # stdlib only, CPU, ~5 s

Nothing here re-runs a model. It reads the per-object / per-run result files the
experiments wrote (paths below), applies one aggregation rule everywhere, and writes
compact CSV/JSON tables plus results/summary.json.

Aggregation rule (used for every ShapeR comparison):
  1. per object, average the seed-level values (or seed-level paired differences);
  2. aggregate across objects (mean, median, standard error across objects);
  3. count objects, not seeds, when reporting "better on k of n".
Seeds are repeats of the same object, not independent samples.
"""
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

HOME = os.path.expanduser("~")
VC = os.path.join(HOME, "viewcrafter_orbits")       # symlink to the ViewCrafter-orbit workspace
SV = os.path.join(HOME, "stable_virtual_camera")
S2S = os.path.join(HOME, "seva_to_shaper")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEEDS = (0, 1, 2)


def load(p):
    with open(p) as fh:
        return json.load(fh)


def write_csv(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def se(x):
    return st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def sign_p(k, n):
    """One-sided sign test P(X >= k | n, 0.5) -- descriptive only."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def shaper_runs(pattern):
    """{(object, condition): {seed: (CD x1e2, F1, NC, num_images)}} from per-object JSONs."""
    R = defaultdict(dict)
    for s in SEEDS:
        for f in glob.glob(pattern.format(seed=s)):
            d = load(f)
            obj, cond = os.path.basename(f)[:-5].rsplit("__", 1)
            m = d["metrics"]
            R[(obj, cond)][s] = (m["chamfer"] * 100, m["f-score"], m["normals"], d.get("num_images", 16))
    return R


def paired(R, cond, base="slam"):
    """Per-object rows of `cond` vs `base`, seeds averaged within object."""
    rows = []
    for obj in sorted({o for o, c in R if c == cond}):
        a, b = R.get((obj, base), {}), R[(obj, cond)]
        ss = sorted(set(a) & set(b))
        if not ss:
            continue
        rows.append({
            "object": obj, "condition": cond, "n_seeds": len(ss),
            "base_cd": st.mean(a[s][0] for s in ss), "cd": st.mean(b[s][0] for s in ss),
            "base_f1": st.mean(a[s][1] for s in ss), "f1": st.mean(b[s][1] for s in ss),
            "d_cd": st.mean(b[s][0] - a[s][0] for s in ss),
            "d_f1": st.mean(b[s][1] - a[s][1] for s in ss),
            "num_images": b[ss[0]][3],
        })
    return rows


def aggregate(rows):
    d = [r["d_cd"] for r in rows]
    f = [r["d_f1"] for r in rows]
    n = len(d)
    k = sum(x < 0 for x in d)
    return {
        "n_objects": n,
        "base_cd": st.mean(r["base_cd"] for r in rows), "cd": st.mean(r["cd"] for r in rows),
        "base_f1": st.mean(r["base_f1"] for r in rows), "f1": st.mean(r["f1"] for r in rows),
        "d_cd_mean": st.mean(d), "d_cd_median": st.median(d), "d_cd_se_objects": se(d),
        "d_f1_mean": st.mean(f), "d_f1_median": st.median(f), "d_f1_se_objects": se(f),
        "objects_cd_better": k, "objects_f1_better": sum(x > 0 for x in f),
        "objects_cd_better_f1_worse": sum(1 for x, y in zip(d, f) if x < 0 and y < 0),
        "objects_cd_worse_f1_better": sum(1 for x, y in zip(d, f) if x > 0 and y > 0),
        "sign_test_p_one_sided_cd": sign_p(k, n),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {}

    # ---- A. SEVA orbit fidelity (5 objects) -------------------------------------
    sc = load(f"{SV}/exp1/outputs/scores.json")
    arc = {r["object"]: r for r in load(f"{SV}/exp1/outputs/input_arc_corrected.json")}
    rows = [{"object": r["object"], "input_arc_true_deg": arc[r["object"]]["input_arc_true_deg"],
             "input_arc_as_first_recorded_deg": r["input_arc_deg"], "orbit_iou_mean": r["orbit_iou_mean"],
             "orbit_iou_min": r["orbit_iou_min"], "masker_ceiling_iou": r["masker_iou_on_inputs"],
             "share_of_ceiling": arc[r["object"]]["share_of_ceiling"], "area_ratio_mean": r["area_ratio_mean"],
             "frames_over_1_5x_area": r["frames_over_1_5x"]} for r in sc]
    write_csv("A_seva_orbit_fidelity.csv", rows, list(rows[0]))

    # ---- A. observed vs never-photographed surface -------------------------------
    ob = load(f"{SV}/exp2/outputs/observability.json")
    rows = [{"object": r["object"], "input_views": r["views"], "surface_observed_fraction": r["surface_observed_fraction"],
             "orbit_frames_mostly_unobserved_of_80": r["frames_mostly_unobserved"],
             "orbit_unobserved_fraction_mean": r["orbit_unobserved_fraction_mean"]} for r in ob]
    write_csv("A_observed_vs_unseen.csv", rows, list(rows[0]))

    # ---- B. false content in one view vs all views --------------------------------
    oc = load(f"{SV}/exp2/outputs/occlusion_scores.json")
    rows = [{"object": r["object"], "condition": {"occ1": "one_view", "occall": "all_views"}[r["condition"]],
             "frames_scored": r["frames_with_ball_in_view"], "copied": r["copied"],
             "reconstructed": r["reconstructed"], "invented": r["invented"]} for r in oc]
    write_csv("B_false_content.csv", rows, list(rows[0]))
    summary["B_false_content"] = {c: {k: sum(r[k] for r in rows if r["condition"] == c)
                                      for k in ("frames_scored", "copied", "reconstructed", "invented")}
                                  for c in ("one_view", "all_views")}

    # ---- C. generated video -> 3D, against the SLAM points ------------------------
    rec = {}
    for src in (f"{VC}/experiment4/outputs/realigned_to_slam.json", f"{VC}/experiment10/outputs/realigned_to_slam.json"):
        for r in load(src):                      # experiment10 overrides the wooden table (view-order fix)
            rec[(r["object"], r["condition"])] = dict(r, source=os.path.relpath(src, HOME))
    for r in load(f"{SV}/exp5/outputs/slam_scores.json"):
        rec.setdefault((r["object"], r["condition"]), dict(r, source=os.path.relpath(f"{SV}/exp5/outputs/slam_scores.json", HOME)))
    rows = [{"object": o, "condition": c,
             "slam_points_reached_2pct": r["slam_to_vggt"]["within_2pct"],
             "slam_to_cloud_mean_pct_diag": r["slam_to_vggt"]["mean_x100"],
             "cloud_to_slam_mean_pct_diag": r["vggt_to_slam"]["mean_x100"],
             "cloud_within_2pct_of_slam": r["vggt_to_slam"]["within_2pct"],
             "cloud_to_mesh_mean_pct_diag": r["vggt_to_mesh"]["mean_x100"],
             "fitted_scale": r.get("fitted_scale"), "fit": r.get("fit"), "up_z": r.get("up_z"),
             "source": r["source"]} for (o, c), r in sorted(rec.items())]
    write_csv("C_video_to_3d_per_object.csv", rows, list(rows[0]))
    byc = defaultdict(list)
    for r in rows:
        byc[r["condition"]].append(r)
    means = [{"condition": c, "n_objects": len(L),
              "slam_points_reached_2pct": st.mean(x["slam_points_reached_2pct"] for x in L),
              "cloud_to_slam_mean_pct_diag": st.mean(x["cloud_to_slam_mean_pct_diag"] for x in L),
              "cloud_to_mesh_mean_pct_diag": st.mean(x["cloud_to_mesh_mean_pct_diag"] for x in L)}
             for c, L in sorted(byc.items())]
    write_csv("C_video_to_3d_means.csv", means, list(means[0]))
    summary["C_video_to_3d_means"] = {m["condition"]: m for m in means}

    # ---- C->ShapeR. reconstructed orbit points as ShapeR's point input (6 objects) --
    R6 = shaper_runs(f"{S2S}/outputs/seeded/s{{seed}}/per_object/*.json")
    pts_rows, views_rows, agg = [], [], {}
    for cond in ("pi3", "kv", "union", "gap1", "gap4", "views", "vonly", "vg80", "v80", "vgap"):
        p = paired(R6, cond)
        (views_rows if cond.startswith("v") else pts_rows).extend(p)
        agg[cond] = aggregate(p)
    cols = ["object", "condition", "n_seeds", "num_images", "base_cd", "cd", "d_cd", "base_f1", "f1", "d_f1"]
    write_csv("C_points_into_shaper_6obj.csv", pts_rows, cols)
    write_csv("D_views_into_shaper_6obj.csv", views_rows, cols)
    summary["six_object_conditions_16_views"] = agg

    kv = shaper_runs(f"{S2S}/outputs_sub/kvseeded/s{{seed}}/per_object/*.json")
    p = paired(kv, "kvpts")
    write_csv("C_kv_points_into_shaper_subset.csv", p, cols)
    summary["C_kv_points_subset"] = aggregate(p)

    # ---- D. view budget (6 objects) -------------------------------------------------
    rows, bud = [], {}
    budgets = {16: R6}
    for nv in (8, 12, 24, 32):
        budgets[nv] = shaper_runs(f"{S2S}/outputs/nviews{nv}/s{{seed}}/per_object/*.json")
    for nv in sorted(budgets):
        for cond in ("views", "vgap"):
            p = paired(budgets[nv], cond)
            a = aggregate(p)
            bud[f"{cond}@{nv}"] = a
            for r in p:
                rows.append(dict(r, budget=nv))
    write_csv("D_view_budget_6obj.csv", rows, ["budget"] + cols)
    summary["D_view_budget"] = bud

    # ---- D. 155 objects, real-only vs real+generated at 24 views ----------------------
    R155 = shaper_runs(f"{S2S}/outputs_sub/seeded/s{{seed}}/per_object/*.json")
    p = paired(R155, "views")
    write_csv("D_views_into_shaper_155obj.csv", p, cols)
    a = aggregate(p)
    seed_pairs = [R155[(r["object"], "views")][s][0] - R155[(r["object"], "slam")][s][0] for r in p for s in SEEDS]
    a["seed_level_pairs"] = len(seed_pairs)
    a["seed_level_pairs_cd_better"] = sum(x < 0 for x in seed_pairs)
    a["se_if_seed_pairs_treated_as_independent"] = se(seed_pairs)
    s0 = [R155[(r["object"], "views")][0][0] - R155[(r["object"], "slam")][0][0] for r in p]
    a["seed0_only_d_cd_mean"] = st.mean(s0)
    a["seed0_only_objects_cd_better"] = sum(x < 0 for x in s0)
    summary["D_views_155"] = a
    sub = {r["obj"] for r in load(f"{S2S}/outputs_sub/traits.json")}
    a40 = aggregate([r for r in p if r["object"] in sub])
    summary["D_views_40_subset"] = a40

    # ---- D. frame-selection pilot (16 objects, budget-matched 15 real + 9 generated) --
    P = shaper_runs(f"{S2S}/sel/pilot/s{{seed}}/per_object/*.json")
    M = {k: (st.mean(v[0] for v in d.values()), st.mean(v[1] for v in d.values())) for k, d in P.items()}
    objs = sorted({o for o, _ in M})
    rows = [{"object": o, **{f"{c}_cd": M[(o, c)][0] for c in "ABCDE"}, **{f"{c}_f1": M[(o, c)][1] for c in "ABCDE"}} for o in objs]
    write_csv("D_frame_selection_pilot.csv", rows, list(rows[0]))
    sel = {}
    for x, y in (("B", "A"), ("C", "B"), ("D", "B"), ("D", "C"), ("D", "E"), ("E", "B")):
        dd = [M[(o, x)][0] - M[(o, y)][0] for o in objs]
        ff = [M[(o, x)][1] - M[(o, y)][1] for o in objs]
        sel[f"{x}_vs_{y}"] = {"d_cd_mean": st.mean(dd), "d_cd_se_objects": se(dd), "objects_cd_better": sum(v < 0 for v in dd),
                              "d_f1_mean": st.mean(ff), "objects_f1_better": sum(v > 0 for v in ff), "n_objects": len(objs)}
    summary["D_frame_selection_pilot"] = sel

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
