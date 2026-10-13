# Experiment code (snapshot)

These are the scripts that produced the results, copied from the working directories they ran in. They are grouped the same way, so relative paths inside each folder still hold. Only the author's own code is here: no upstream repository, checkpoint or dataset is included.

| folder | from | what |
|---|---|---|
| `viewcrafter_orbits/` | `~/viewcrafter_orbits` | ViewCrafter orbit runs (`experiment1`–`10`), VGGT runner, Sim(3) fitters and SLAM-point scoring (`experiment3/scripts/align_and_score.py`, `experiment4/scripts/realign_to_slam.py`), trajectory files, `shared/objects.json` |
| `seva_orbits/` | `~/stable_virtual_camera` | SEVA env setup and weight prefetch; `exp1` scene builder, occlusion cleaning, orbit scoring; `exp2` observability and false content; `exp3` VGGT on SEVA; `exp4` π³ runner and camera-anchored fit; `exp5` KV-Tracker configs, collection and scoring |
| `kv_tracker/` | `~/kv_tracker_testing` | KV-Tracker env setup, env check, TUM sanity run and scorer |
| `pi3/` | `~/pi3` | standalone π³ env setup and smoke test |
| `generated_to_shaper/` | `~/seva_to_shaper` | C2 and D1–D3: scene building, SEVA driver, π³/KV placement, ShapeR pkl builders (`make_pkls*.py`), view-budget runner, selection study, summaries, renderers |

## Differences from the originals

Only two kinds of edit were made in the copy. A check confirmed that no other line differs.

1. The ViewCrafter workspace path was rewritten to `~/viewcrafter_orbits`. On the cluster that is a symlink to the original directory, so the scripts resolve to the same files.
2. A few comments and two SLURM job names were reworded.

Deliberately not copied: slide/deck builders, one superseded comparison script and its job file, generated outputs, logs, and `__pycache__`.

## Running

The scripts use absolute paths and the cluster's conda envs. See [docs/reproducibility.md](../docs/reproducibility.md) and [docs/environments.md](../docs/environments.md). Local patches to upstream KV-Tracker and SEVA are described, not included, in [docs/environments.md](../docs/environments.md#local-patches-to-upstream-code).
