# C2 · Bridge: reconstructed orbit points as ShapeR's point input

[← experiment index](README.md) · code: [`code/generated_to_shaper/`](../code/generated_to_shaper/) · envs: `shaper`, `kvtracker`, `svcam`

This is Pathway 1's output fed into ShapeR. **Only the point cloud changes.** No generated image enters ShapeR here. That is what separates it from [D1](D1_views_into_shaper.md).

## Question and hypothesis

ShapeR conditions on a sparse SLAM point cloud. Its weakest benchmark results include objects whose cloud is very thin. Can points reconstructed from a generated 360° orbit replace, or supplement, the SLAM points? Hypothesis: they help where the real cloud is too thin to constrain the shape *and* the generated cloud is accurate.

## Setup

**Objects (6)** from the ShapeR evaluation benchmark, stools and tables (`objects.json`):

| object | SLAM points | Aria |
|---|---:|---|
| `BNB1045__wovenstool` | 300 | Gen2 |
| `BNB1045__barstool` | 407 | Gen2 |
| `BNB1182__sidetable` | 445 | Gen2 |
| `BNB1182__stool` | 706 | Gen2 |
| `HWD0045__desktable` | 1 710 | Gen1 |
| `HWD0045__woodentable` | 7 628 | Gen1 |

`BNB2727__stool` was built but dropped (3 usable views over a 21° arc).

**Pipeline:** scenes as in [A2](A2_seva_orbits.md) → SEVA 80-frame orbit → π³ batch and KV-Tracker (keyframe every frame) → placement → ShapeR.

**Placement** (`scripts/align_clouds.py`):
1. Umeyama from the reconstructor's 80 estimated camera centres to the 80 requested cameras, which are known in the model frame.
2. Sim(3) ICP refinement of the SLAM points onto the cloud, accepted only if rotation moves < 25°, scale stays within 0.7–1.6×, the centre shifts < 15% of the diagonal, the extent ratio stays within 0.6–1.7, and the symmetric Chamfer improves (`MAX_ROT_DEG, SCALE_BAND, MAX_SHIFT, EXTENT_BAND` in the script).

Only the 7-DoF pose is touched. An earlier ±15% scale cap rejected 8 of 12 refinements and left clouds 3–10% off, which is why the band is wide. The script's docstring still states the older limits; the constants above are what ran.

**ShapeR input** (`scripts/make_pkls.py`, `make_pkls_gapfill.py`): only `points_model` (and its zeroed per-point std arrays) changes. Images, masks, cameras, caption, bounds and the reference mesh are the benchmark's bytes. Generated clouds are clipped to 1.1× the object box, voxel-averaged on ShapeR's 256³ grid and capped at 4 096 points.

| condition | `points_model` |
|---|---|
| `slam` | the capture's SLAM points (baseline, re-run under the same seeds) |
| `pi3`, `kv` | π³ / KV-Tracker orbit cloud |
| `union` | SLAM points + the better cloud |
| `gap1`, `gap4` | SLAM points + generated points only where no SLAM point is within τ, capped at 1× / 4× the SLAM count |

**Evaluation:** ShapeR `balance` preset (16 views), seeds 0, 1, 2, with the RNG reset per object so every condition sees the same noise. Metrics from ShapeR's own `evaluation/eval.py` ([evaluation](../docs/evaluation.md)).

```bash
cd ~/seva_to_shaper
sbatch slurm/01_build_scenes.sbatch && sbatch slurm/02_run_seva.sbatch
sbatch slurm/03_reconstruct.sbatch     # kvtracker env: Pi3 + KV-Tracker
sbatch slurm/07_seeded.sbatch          # paired, seeded ShapeR runs -> outputs/seeded/s{0,1,2}
python scripts/summarise_seeded.py
```

## Results

CD ×10² / F1, mean of 3 seeds (`results/C_points_into_shaper_6obj.csv`, from `outputs/seeded/s*/per_object/*.json`):

| object | real-only (`slam`) | `pi3` | `kv` | `union` | `gap1` | `gap4` |
|---|---|---|---|---|---|---|
| `BNB1045__barstool` | 2.36 / 0.679 | 7.43 / 0.252 | 6.00 / 0.313 | 4.25 / 0.456 | 2.95 / 0.587 | 3.97 / 0.479 |
| `BNB1045__wovenstool` | 3.24 / 0.517 | 3.06 / 0.603 | 3.75 / 0.527 | 3.63 / 0.535 | 3.23 / 0.592 | 2.84 / 0.650 |
| `BNB1182__sidetable` | 2.92 / 0.697 | 3.07 / 0.592 | 2.70 / 0.622 | 2.57 / 0.630 | 2.72 / 0.670 | 2.56 / 0.645 |
| `BNB1182__stool` | 2.47 / 0.701 | 2.23 / 0.731 | 2.15 / 0.774 | 2.09 / 0.767 | 1.82 / 0.796 | 2.03 / 0.777 |
| `HWD0045__desktable` | 4.76 / 0.727 | 5.82 / 0.616 | 10.97 / 0.545 | 4.70 / 0.638 | 5.19 / 0.639 | 6.87 / 0.585 |
| `HWD0045__woodentable` | 2.22 / 0.683 | 7.14 / 0.476 | 8.64 / 0.403 | 6.65 / 0.497 | 5.62 / 0.495 | 5.52 / 0.490 |
| **mean paired change** | — | +1.80 / −0.122 | +2.71 / −0.137 | +0.99 / −0.080 | +0.59 / −0.038 | +0.97 / −0.063 |
| **objects CD better** | — | 2 / 6 | 2 / 6 | 3 / 6 | 3 / 6 | 3 / 6 |

**Replication on 9 more objects** (`results/C_kv_points_into_shaper_subset.csv`, from `outputs_sub/kvseeded/`; KV-Tracker points replace SLAM points at a 24-view budget, 3 seeds): mean ΔCD **+1.63** (SE 0.65), ΔF1 −0.17, CD better on **2 of 9** (`BNB1045__nightstand_2` −0.51, `BNB1723__chair` −0.56). A tenth object (`BNB2727__lighting_2`) could not be placed: camera RMS was 183% of the diagonal, so no points landed in the box (`outputs_sub/kvtest_placement.json`).

## Interpretation

- **Generated points are a poor substitute for SLAM points.** Every point condition loses on average. The losses are largest for the bar stool, whose orbit drifted (both reconstructors' cameras collapse, rms 75–100% of the diagonal, and the cloud sits ~20% off the surface), and for the wooden table, which already had 7 628 accurate SLAM points to lose.
- Keeping all SLAM points and adding few generated ones (`gap1`) is the least harmful point condition.
- The 6 objects support no claim about *which* point mix is best: 2–3 of 6 objects improve in every column.

## Discrepancy with the first record

The workspace README for this study reports a woven-stool gain from an **unseeded** first pass (3 unpaired repeats): CD 3.60 → 2.62, F1 0.487 → 0.640, "~5 standard deviations". In the later **seed-matched paired** runs above, the same comparison is 3.24 → 3.06 / 0.517 → 0.603. That is still a gain, but much smaller. The "~5σ" figure should not be cited. See [discrepancies](../docs/discrepancies.md).

## Limitations

- 6 + 9 objects; seeds are averaged within an object and are not independent samples.
- Placement uses the requested orbit cameras, which are defined in the reference-mesh frame (reference-assisted).
- The density-matched controls (`pi3_n`, `kv_n`) exist only in the unseeded first pass (`outputs/summary.json`), not in the paired runs.

## Next experiment motivated

ShapeR's strongest input is its images, not its points. Feed the orbit as **images** with exact cameras → [D1](D1_views_into_shaper.md).
