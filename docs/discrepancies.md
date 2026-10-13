# Discrepancies between records

Where two records of the same result disagree, this page says which one this repository uses and why. Nothing was silently overwritten: the older files are still in the original workspaces.

| # | Topic | Older record | Value used here | Why |
|---|---|---|---|---|
| 1 | **Summary document** | A summary PDF of these experiments was referenced but is **not available on the machine** where this repository was built | — | Its numbers could not be cross-checked. Every number here was recomputed from the run outputs. **Cross-check against the PDF: not yet done.** |
| 2 | Pathway-1 fitter (all ViewCrafter / real / SEVA clouds) | Two-axis ICP fit: 60/60 clouds upside down. SO(3) grid fit: 21/60 still flipped. Workspace READMEs, decks, and `outputs/oldfit/`, `outputs/so3fit/` carry these | Upright camera-anchored fit (`experiment4/outputs/realigned_to_slam.json`; `experiment10/…` for the table; `exp5/outputs/slam_scores.json`) | The older fits scored mirrored clouds. Several conclusions flip between layers: e.g. real photos 0.426 → **0.669**, ±45° 0.420 → **0.660**, ViewCrafter 1-view 360° 0.526 → **0.421** |
| 3 | Wooden-table view order (ViewCrafter 3/8-view) | Wrap-around bug rotated the 8 views to start mid-arc | `experiment10` re-run with the circular rule | Five-object means shift ≤ 0.05; no ranking reverses |
| 4 | `real` control mean | `experiment4/README.md`: the failed wooden-table control is *excluded* from averages | **Included** (0.669); excluded it is 0.834 | The current means file includes it. Both values are stated in [C1](../experiments/C1_video_to_3d.md) |
| 5 | SEVA input arc (chair, bench) | 216.4° and 200.3° in `exp1/outputs/scores.json` (an arc measurement that ignored the 0° wrap) | 132.8° and 102.4° (`exp1/outputs/input_arc_corrected.json`) | Orbits and IoU are unaffected; only the arc label was wrong |
| 6 | Woven stool, π³ points (C2) | Workspace README, unseeded first pass: CD 3.60 → 2.62, F1 0.487 → 0.640, "~5σ, all three repeats" | Seed-matched paired runs: 3.24 → 3.06, 0.517 → 0.603 | Paired runs remove sampler noise from the comparison; the "~5σ" figure compared unpaired repeats |
| 7 | 155-object standard error (D2) | `outputs_sub/full155_summary.json`: SE 0.031 | SE **0.046** over objects | 0.031 treats 465 seed-level pairs as independent samples |
| 8 | Selection pilot (D3) | `sel/pilot/summary.json`: 8 objects; B vs A −0.446 | All 16 objects, recomputed from per-object JSONs: B vs A −0.087 | The summary file was written after one shard |
| 9 | Placement guards (C2) | `align_clouds.py` docstring: scale < 15%, shift < 10% | Constants that ran: scale 0.7–1.6×, shift < 15%, extent 0.6–1.7, rotation < 25° | The docstring was not updated when the band was widened |
| 10 | ViewCrafter backdrop invention | "20–23 of 25 frames above 1.5× area" stated for all objects | 4 of 5 objects; the wooden table is 0/25 | Recomputed from `experiment2/outputs/subject_fraction.json` |
| 11 | False content, one view (B1) | Earlier summary mentioned only the 2/400 copied frames | 2 copied **and 44 invented** of 400 | The `invented` count is in the same score file |
| 12 | Gallery mesh metrics (D2) | — | Mesh panels come from a seed-0 re-run (`meshes24`) that differs from the scored seed-0 run by up to 0.06 CD | Captions cite the re-run and print the 3-seed mean beside it |

## Older presentation material

Earlier slide decks and HTML reports built inside the original workspaces predate items 2–3 and 6–8, and are not part of this repository.
