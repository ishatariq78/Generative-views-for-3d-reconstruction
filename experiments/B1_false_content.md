# B1 · False content: one view vs every view

[← experiment index](README.md) · code: [`code/seva_orbits/exp2`](../code/seva_orbits/exp2) · env: `svcam`, `shaper`

## Question and hypothesis

Real captures contain things that are not the object, such as a lamp in front of a sofa or a frame of the next room. If false content appears in **one** conditioning view, does the generator copy it into the orbit? If it appears **consistently** in every view, does it? Hypothesis: a multi-view generator weighs evidence by agreement across views. It should overrule a lone outlier and reproduce consistent content, whether or not that content is true.

## Setup

- Same 5 objects, 8 input views, 80 target cameras, seed and settings as [A2](A2_seva_orbits.md). The clean exp1 run is the control.
- **False content:** a magenta sphere (radius ≈ 0.11 × object diagonal) floated in front of the best-observed spot of the object. It is projected with true geometry into **one** view (`occ1`) or **all** views (`occall`) (`scripts/build_occluded.py`). Magenta appears nowhere in these scenes, so "did it draw it" becomes a pixel count.
- **Scoring** (`scripts/score_occlusion.py`), per frame where the ball would be visible: `copied` (magenta present), `reconstructed` (region matches the clean run), `invented` (region differs from the clean run, but not magenta).

```bash
cd ~/stable_virtual_camera
sbatch exp2/slurm/build.sbatch && sbatch exp2/slurm/run_seva.sbatch && sbatch exp2/slurm/score.sbatch
```

## Results

`exp2/outputs/occlusion_scores.json`, 80 frames per object:

| object | one view: copied | reconstructed | invented | all views: copied | reconstructed | invented |
|---|---:|---:|---:|---:|---:|---:|
| `ADT1292__chair` | 0 | 76 | 4 | 79 | 0 | 1 |
| `ADT1292__woodentable` | 0 | 77 | 3 | 80 | 0 | 0 |
| `BNB1045__sofa` | **2** | 54 | **24** | 80 | 0 | 0 |
| `BNB1723__bed` | 0 | 80 | 0 | 80 | 0 | 0 |
| `BNB2727__bench` | 0 | 67 | 13 | 55 | 25 | 0 |
| **total** | **2 / 400** | 354 | 44 | **374 / 400** | 25 | 1 |

<img src="../assets/false_content.jpg" width="100%" alt="Magenta ball in one input view vs all input views, sofa and bed">

## Interpretation

- **One view:** the false content is overruled. The 2 copied frames are sofa frames 23–24 (168–173° on an orbit starting at 65°), just past painted view 7 at the end of the input arc (158°), where that view is the sole witness. No frame of the other four objects copies it. The pattern is consistent with weighing evidence by agreement across views.
- **But "not copied" is not "unchanged."** 44 of 400 frames are scored `invented`: the region differs from the clean run without being magenta. They are concentrated on the sofa (24) and bench (13). The generator does not simply ignore the corrupted view. It produces *something else* there.
- **All views:** the ball is drawn as a solid object in 374 of 400 frames. The exception is a run of bench frames (24–48), where the ball is absent and the region matches the clean run. Whether the ball is geometrically hidden from those cameras or simply dropped is **not yet verified**.
- For real captures, a stray occluder mid-arc is mostly harmless, while one at the ends of the arc, or present in every view, will be generated as real. That is why the cleaning step in [A2](A2_seva_orbits.md) matters, and why consistency is not evidence of truth.

## Limitations

- A synthetic, maximally distinctive occluder. Real false content (a sibling object, a reflection) is not magenta and would not be counted this way.
- One seed per condition, 5 objects.
- The `invented` verdict depends on a pixel-difference threshold against the clean run. It was not calibrated against human judgement.
- Only the generated *images* are scored here. What the false content does to 3D reconstruction or to ShapeR was not measured.

## Next experiment motivated

Carry the (clean) orbits into 3D and see how much of the generated geometry survives → [C1](C1_video_to_3d.md).
