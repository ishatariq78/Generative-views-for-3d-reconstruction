# Evaluation

Two evaluation regimes, one per pathway. They measure different things and are never combined.

## Pathway 2 (and the C2 bridge): ShapeR meshes vs the reference mesh

Metrics come from ShapeR's released `evaluation/eval.py`, imported unchanged (`_eval_single_mesh`):

| metric | definition (as implemented) | reported as |
|---|---|---|
| Chamfer distance | **L1** Chamfer, 0.5 × (accuracy + completeness), on 100 000 surface samples per mesh | CD ×10², lower is better |
| F-score | harmonic mean of precision and recall at threshold `linspace(1/144, 2, 1000)[10]` ≈ **0.0270** | F1, higher is better |
| Normal consistency | mean absolute cosine between matched normals | NC (in the CSVs, not used for claims) |

- **Units.** Both meshes are scaled by `2 × max(bounds)` before scoring, so the metrics are in ShapeR's normalised evaluation units, not metres. CD and the F1 threshold therefore scale with object size. The conversion to metres is **not verified in this repository**.
- **Reference.** This is the benchmark's reference mesh, as distributed with the benchmark. How it was produced is documented by the ShapeR authors and **not re-verified here**. Treat it as pseudo-ground truth, not a measurement made for this study.
- **Seeds.** The runner resets the RNG to the seed before every object. Each condition therefore sees the same rectified-flow noise and the same evaluation samples, and the difference between two conditions on one object and one seed is a paired observation. Kernels are not bit-deterministic: a re-run at the same seed differs by ≈ 0.03 CD ×10² on average (measured earlier on 8 objects × 3 repeats; an example is the meshes24 re-run vs the scored seed-0 run in [D2](../experiments/D2_views_at_scale.md#qualitative-examples)).
- **Aggregation.** Per object, the seed-level values (or paired differences) are averaged first. Means, medians, standard errors and "better on k of n" counts are then taken **over objects**. Seeds are repeats of one object, not independent samples.
- **Sign test.** A one-sided binomial test on the number of objects with lower CD. It is given as a descriptive figure only, and no significance claim is made.
- **CD and F1 can disagree.** CD averages over all distances, while F1 counts points inside a fixed threshold. A mesh can reduce a few large errors (CD better) while moving many points just outside the threshold (F1 worse). Both are reported everywhere.

## Pathway 1: reconstructed clouds vs the capture's SLAM points

| metric | definition | note |
|---|---|---|
| SLAM points reached | share of SLAM points with a cloud point within 2% of the reference diagonal | measured side only |
| cloud → SLAM | mean distance from cloud points to the nearest SLAM point, % of diagonal | penalises off-surface haze, but cannot tell haze on the *unmeasured* side from error |
| SLAM → cloud | mean distance from SLAM points to the nearest cloud point | |
| cloud → mesh | to the reference mesh | **diagnostic only** |
| ATE / RPE | estimated cameras vs the cameras SEVA was asked for, Sim(3) via evo | mixes reconstructor error with generator drift |

- **Why SLAM points.** They are a real measurement of this capture: 0.35–0.73% of the diagonal from the reference surface, and 97–100% within 2% of it on the 5 objects (`experiment4/outputs/baselines.json`). But they are **sparse and one-sided**: they lie within 2% of only 17–68% of each reference surface.
- **Alignment and scoring share the SLAM points.** The Sim(3) fit is optimised against the same points it is then scored on.
- **Determinism.** Cloud subsamples are seeded from `zlib.crc32` of the cloud bytes, and the mesh sample from `np.random.seed(0)`, so identical clouds score identically regardless of loop order.

## Image-level metrics (A, B)

- **Orbit IoU and share of ceiling** ([A2](../experiments/A2_seva_orbits.md)): silhouette IoU against the reference mesh rasterised from the requested camera, divided by the masker's IoU on inputs with known masks.
- **Subject-area ratio and render coverage** ([A1](../experiments/A1_viewcrafter_orbits.md)).
- **Copied / reconstructed / invented** ([B1](../experiments/B1_false_content.md)): magenta fraction, and the inside/outside difference ratio against the clean run (threshold `RECON_T = 1.6`).

## Splits and sample counts

| study | objects | seeds | split |
|---|---:|---:|---|
| A1 ViewCrafter | 3 (paths), 5 (others) | 1 (seed 123) | none |
| A2, B1, C1 | 5 | 1 | none |
| C2, D1 | 6 | 3 | none |
| C2 KV replication | 9 (+1 failed placement) | 3 | none |
| D2 | 40 ⊂ 155 | 3 | 40 chosen from capture traits only |
| D3 | 16 | 3 | development objects only, with no test split yet |
