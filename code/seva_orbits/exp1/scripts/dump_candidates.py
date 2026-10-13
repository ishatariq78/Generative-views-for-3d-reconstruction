#!/usr/bin/env python3
"""Contact sheet of every candidate view with its consistency scores.

    python scripts/dump_candidates.py ADT1292__chair

Written because the consistency filter started rejecting views and the only
honest way to set its thresholds is to look at what it is rejecting. Each tile is
labelled with the azimuth, the disagreeing-pixel fraction, and the correlation
against the other views.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp1/scripts")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# build_scenes first: importing it is what puts ShapeR and the exp2 scripts on
# sys.path, and occlusion needs them at import time.
from build_scenes import gather_views, make_cutout, feather  # noqa: E402
import occlusion  # noqa: E402

OUTDIR = Path("/home/isha.tariq/stable_virtual_camera/exp1/logs")
S = 190


def main(name):
    d, mesh, centre, diag, kept = gather_views(name)
    kept.sort(key=lambda k: k["az"])
    print(f"{name}: {len(kept)} candidates")
    res = occlusion.analyse(kept, mesh)

    cols = 8
    rows = (len(kept) + cols - 1) // cols
    sheet = Image.new("RGB", (S * cols, (S + 18) * rows), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (v, r) in enumerate(zip(kept, res)):
        cut = make_cutout(v["img"], feather(v["alpha"].astype(np.float32)))
        x, y = (i % cols) * S, (i // cols) * (S + 18)
        sheet.paste(Image.fromarray(cut).resize((S, S)), (x, y))
        es = occlusion.edge_support(v["img"], v["alpha"])
        verdict = "DROP" if r["frac"] > occlusion.VIEW_REJECT else "keep"
        dr.text((x + 3, y + S + 3),
                f"az{v['az']:.0f} bad{r['frac']:.2f} r{r['corr']:+.2f} e{es:.2f}",
                fill=(0, 0, 0))
        print(f"  az {v['az']:6.1f}  frame {v['idx']:4d}  bad {r['frac']:.3f}  "
              f"judged {r['judged']:.2f}  corr {r['corr']:+.3f}  "
              f"edge {es:.3f}  {verdict}")
    p = OUTDIR / f"candidates_{name}.png"
    sheet.save(p)
    print("wrote", p)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["ADT1292__chair"]:
        main(n)
