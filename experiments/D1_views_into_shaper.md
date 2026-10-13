# D1 · Pathway 2: real + generated images into ShapeR

[← experiment index](README.md) · code: [`code/generated_to_shaper/scripts/make_pkls_views.py`](../code/generated_to_shaper/scripts/make_pkls_views.py), [`run_shaper_nviews.py`](../code/generated_to_shaper/scripts/run_shaper_nviews.py) · env: `shaper`

Here the generated orbit enters ShapeR as **images with cameras**. The point cloud stays the capture's own SLAM points, except in `vgap`. Nothing is reconstructed from the generated frames, which is the difference from [C2](C2_points_into_shaper.md).

## Question and hypothesis

ShapeR's image conditioning is whatever views the wearer happened to walk past: a partial arc, clutter, occluders. A SEVA orbit is the target object alone, cleaned, all the way round. Does adding those frames as extra views improve the mesh? Hypothesis: it helps one-sided or occluded captures. It can hurt when generated views *displace* real ones within ShapeR's fixed view budget.

## Input preparation and camera handling

Each generated frame was rendered from a camera we specified. Its pose is therefore known exactly in the model frame, and no pose estimation is involved (`orbit_views()`).

1. Requested camera from `transforms.json` (OpenGL c2w) → OpenCV → `T_camera_model`.
2. Frame converted to greyscale. The benchmark's `image_data` is 8-bit greyscale (checked on one pkl: mode `L`, 640×480).
3. Pinhole → **equidistant** remap. ShapeR reads Fisheye624 cameras, and a Fisheye624 with zero distortion terms is exactly equidistant (r = f·θ), so the frame is declared as `[f, cx, cy, 0×12]`. ShapeR's own `rectify_images` maps it back to pinhole.
4. Placed unstretched in the middle of the capture's own frame size.
5. The capture's SLAM points are projected into the generated camera to give that view's `object_point_projections` and visible points. This is a frustum test only, with no occlusion test. Views with fewer than 60 projected points are not used (`DOT_MIN = 60`).
6. Generated views are **appended** to the real ones (`views`, `vg80`, `vgap`) or **replace** them (`vonly`, `v80`). ShapeR's own view selector (k-means over camera centres) then picks the views it consumes.

`scripts/check_views.py` renders exactly what the loader feeds the model (images with the ingested point mask), to confirm the object lands where the mask says.

## Conditions (6 objects of [C2](C2_points_into_shaper.md), 3 seeds, RNG reset per object)

| condition | images offered | points |
|---|---|---|
| `slam` (baseline) | real only | SLAM |
| `views` | real + 16 generated | SLAM |
| `vonly` | 16 generated only | SLAM |
| `vg80` | real + all 80 generated | SLAM |
| `v80` | all 80 generated only | SLAM |
| `vgap` | real + 16 generated | SLAM + 1× gap-fill generated points (`gap1`) |

**Real/generated composition of the views ShapeR actually used** was recorded for only two objects, at a 24-view budget (`outputs_sub/view_composition.json`): `HWD0045__desktable` used 15 real + 9 generated, and `BNB2909__armchair_2` 23 real + 1 generated. For the other runs the split is **not recorded**. The frame-selection study ([D3](D3_frame_selection.md)) fixes it exactly.

```bash
cd ~/seva_to_shaper
~/.conda/envs/shaper/bin/python scripts/make_pkls_views.py --only <obj> --check
sbatch slurm/08_views.sbatch       # views conditions at the preset budget (16)
sbatch slurm/13_nviews.sbatch      # budgets 8 / 12 / 24 / 32 -> outputs/nviews<N>/s{0,1,2}
```

## Results at the preset budget (16 views)

CD ×10² / F1, mean of 3 seeds (`results/D_views_into_shaper_6obj.csv`):

| object | real-only | `views` | `vonly` | `vg80` | `v80` | `vgap` |
|---|---|---|---|---|---|---|
| `BNB1045__barstool` | 2.36 / 0.679 | 1.65 / 0.771 | 1.67 / 0.777 | 1.65 / 0.787 | 1.65 / 0.795 | 1.98 / 0.717 |
| `BNB1045__wovenstool` | 3.24 / 0.517 | 3.18 / 0.475 | 3.26 / 0.446 | 3.64 / 0.371 | 3.72 / 0.353 | 3.17 / 0.559 |
| `BNB1182__sidetable` | 2.92 / 0.697 | 3.80 / 0.574 | 4.23 / 0.515 | 4.01 / 0.544 | 4.14 / 0.529 | 3.77 / 0.573 |
| `BNB1182__stool` | 2.47 / 0.701 | 1.82 / 0.754 | 3.26 / 0.448 | 2.28 / 0.739 | 3.02 / 0.519 | 1.73 / 0.823 |
| `HWD0045__desktable` | 4.76 / 0.727 | 5.64 / 0.716 | 8.98 / 0.568 | 11.24 / 0.557 | 11.30 / 0.535 | 5.92 / 0.736 |
| `HWD0045__woodentable` | 2.22 / 0.683 | 2.53 / 0.651 | 2.67 / 0.590 | 2.57 / 0.625 | 2.69 / 0.607 | 5.36 / 0.490 |
| **mean paired change** | — | +0.11 / −0.010 | +1.02 / −0.110 | +1.24 / −0.064 | +1.43 / −0.111 | +0.66 / −0.018 |
| **objects CD better** | — | 3 / 6 | 1 / 6 | 2 / 6 | 1 / 6 | 3 / 6 |

The heatmap and mesh comparison for these conditions are in [C2](C2_points_into_shaper.md#results) (right half of the heatmap = this page).

Generated-only input is worse than real-only on 5 of 6 objects. The real views carry information the generated ones do not, even when the real views are partial.

The bar stool is the object whose orbit **failed as points** in C2 (CD 2.36 → 7.43). **As images**, the same orbit improves it in every condition (→ 1.65). See the [teaser](../README.md).

## View budget

ShapeR's `balance` preset consumes 16 views, so at 16 the generated views compete with real ones for slots. `run_shaper_nviews.py --num_images N` changes the budget, with the baseline run at the same budget.

<img src="../assets/chart_view_budget.png" width="100%" alt="Change in CD and F1 vs view budget">

| budget | `views`: ΔCD / ΔF1 (CD better) | `vgap`: ΔCD / ΔF1 (CD better) |
|---|---|---|
| 8 | +1.00 / −0.050 (1/6) | +1.66 / −0.067 (1/6) |
| 12 | +0.16 / −0.035 (2/6) | +1.17 / −0.074 (3/6) |
| 16 | +0.11 / −0.010 (3/6) | +0.66 / −0.018 (3/6) |
| 24 | **−0.19 / −0.001 (3/6)** | +0.79 / −0.037 (2/6) |
| 32 | +0.06 / −0.031 (3/6) | +0.90 / −0.086 (2/6) |

`results/D_view_budget_6obj.csv`, from `outputs/nviews{8,12,24,32}` and `outputs/seeded` (16).

## Interpretation

- At small budgets, generated views displace real ones and the mesh gets worse. At 24 the average change in CD turns slightly negative. That is a 6-object mean with 3 of 6 objects improving, not a reliable gain.
- **The 24-view budget used at scale ([D2](D2_views_at_scale.md)) was chosen from this sweep.** These 6 objects are also part of D2's 155, so the budget choice is not independent of that test.
- CD and F1 move together on average but not per object (e.g. `views` on the woven stool: CD −0.06, F1 −0.042).

## Limitations

- 6 objects. Per-object effects flip sign between budgets.
- Generated views are cleaned, object-only cut-outs. Their effect is confounded with "a clean view of the object", and no cleaned-real-views control exists yet.
- The generated cameras sit on a circle defined with the reference mesh (reference-assisted).
- Visible points in generated views come from a frustum test with no occlusion test.

## Next experiment motivated

Run the `views` condition at the chosen budget on many more objects → [D2](D2_views_at_scale.md).
