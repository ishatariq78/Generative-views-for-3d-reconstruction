#!/usr/bin/env python3
"""What actually reaches ShapeR in the 24-view configuration, and what the inputs depend on.

    ~/.conda/envs/shaper/bin/python scripts/inspect_pipeline.py --objects <obj> ...

Reports, per object: how many generated views are offered, how many ShapeR's own view
selection keeps, and therefore the real/generated split the model is conditioned on. Also
records the frame identities so a selection experiment can hold them fixed.
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np

E = Path("/home/isha.tariq/seva_to_shaper")
REPO = "/home/isha.tariq/shapeR_testing/ShapeR"
os.chdir(REPO); sys.path.insert(0, REPO)
from dataset.image_processor import cluster_and_select_images  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--objects", nargs="*", required=True)
ap.add_argument("--num_views", type=int, default=24)
ap.add_argument("--pkl_dir", default=str(E / "outputs_sub/pkls"))
a = ap.parse_args()

rows = []
for obj in a.objects:
    base = pickle.load(open(f"/home/isha.tariq/shapeR_testing/data/{obj}.pkl", "rb"))
    n_real = len(base["image_data"])
    p = Path(a.pkl_dir) / f"{obj}__views.pkl"
    if not p.exists():
        print(f"{obj}: no views pkl at {p}"); continue
    d = pickle.load(open(p, "rb"))
    n_total = len(d["image_data"]); n_gen = n_total - n_real
    scale = 0.9 / float(np.max(d["bounds"].numpy()))
    sel = cluster_and_select_images(d, a.num_views, scale, False)
    idx = [int(x[0]) for x in sel]
    gen = [i for i in idx if i >= n_real]
    rows.append({"object": obj, "real_available": n_real, "generated_offered": n_gen,
                 "selected": len(idx), "generated_selected": len(gen),
                 "real_selected": len(idx) - len(gen),
                 "selected_indices": idx, "generated_indices": gen})
    print(f"{obj:30s} real {n_real:5d} + gen {n_gen:3d} offered -> selected {len(idx):2d} "
          f"= {len(idx)-len(gen):2d} real + {len(gen):2d} generated", flush=True)
(E / "outputs_sub/view_composition.json").write_text(json.dumps(rows, indent=2))
if rows:
    print(f"\nmean generated share of the {a.num_views}-view budget: "
          f"{np.mean([r['generated_selected'] / max(r['selected'],1) for r in rows]):.2f}")
