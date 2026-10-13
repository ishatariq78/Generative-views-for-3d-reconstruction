# Results

Compact tables rebuilt from the raw run outputs by [`scripts/build_results.py`](../scripts/build_results.py). Each row is one object × condition. ShapeR rows give seed-averaged values plus the per-object paired change against real-only. Where each number comes from: [PROVENANCE.md](PROVENANCE.md).

| file | experiment | rows |
|---|---|---|
| `A_seva_orbit_fidelity.csv` | A2 | 5 objects |
| `A_observed_vs_unseen.csv` | A2 | 5 objects |
| `B_false_content.csv` | B1 | 5 objects × {one view, all views} |
| `C_video_to_3d_per_object.csv` | C1 | 5 objects × 20 conditions |
| `C_video_to_3d_means.csv` | C1 | 20 conditions |
| `C_points_into_shaper_6obj.csv` | C2 | 6 objects × 5 point conditions |
| `C_kv_points_into_shaper_subset.csv` | C2 | 9 objects |
| `D_views_into_shaper_6obj.csv` | D1 | 6 objects × 5 view conditions (16 views) |
| `D_view_budget_6obj.csv` | D1 | 6 objects × 2 conditions × 5 budgets |
| `D_views_into_shaper_155obj.csv` | D2 | 155 objects (24 views) |
| `D_frame_selection_pilot.csv` | D3 | 16 objects × conditions A–E |
| `summary.json` | all | aggregates: object-level mean/median/SE, win counts, CD/F1 disagreement counts, sign tests |

Columns: `base_cd`/`base_f1` = real-only (`slam`) baseline; `cd`/`f1` = the condition; `d_cd`/`d_f1` = mean over seeds of the paired difference (negative `d_cd` = better). CD is ×10². Pathway-1 distances are % of the reference-mesh diagonal.

## What is not in Git

Raw outputs stay in the original workspaces on the cluster. They are too large for Git, and much of them is derived from the benchmark data, which carries its own licence:

| what | where | approx. size |
|---|---|---|
| SEVA orbits for 155 objects (80 frames each) | `~/seva_to_shaper/outputs_full/` | 2.5 GB |
| 6-object study: orbits, clouds, pkls, meshes, per-run JSONs | `~/seva_to_shaper/outputs/` | 21 GB |
| 40/155-object runs: per-run JSONs, example meshes | `~/seva_to_shaper/outputs_sub/` | 2.7 GB |
| Selection pilot per-run JSONs (pkls deleted after scoring) | `~/seva_to_shaper/sel/` | 2 MB |
| VGGT clouds and SLAM-aligned clouds | `~/viewcrafter_orbits/experiment3,4,10/outputs/` | 8.0 + 0.4 + 0.4 GB |
| π³ / KV-Tracker clouds | `~/stable_virtual_camera/exp4,5/outputs/` | 0.1 + 1.8 GB |
| ViewCrafter videos and point clouds | `~/viewcrafter_orbits/experiment{1,2,5–9}/outputs/` | 3.3 GB |
| ShapeR benchmark pkls | `~/shapeR_testing/data/` | ~25 GB |

If they are published, they belong in separate storage (e.g. a release asset or a dataset host) under a licence compatible with the benchmark's. That is **not yet decided**.
