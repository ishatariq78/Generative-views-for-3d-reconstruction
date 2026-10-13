# Reproducibility

## What was actually run while preparing this repository

| command | where | result |
|---|---|---|
| `python3 scripts/build_results.py` | login node, CPU | rebuilt every file in `results/` from the raw run outputs; condition means equal the experiments' own summary files (C1 to 3 d.p.; D2 means to 1e-12) |
| `~/.conda/envs/shaper/bin/python scripts/make_figures.py` | login node, CPU, software EGL | rendered every file in `assets/` |
| `python3 -m py_compile` on every file in `code/` | login node | all compile |

**No GPU experiment was re-run.** The pipeline commands below are the ones recorded in the workspaces' SLURM scripts. They are documented as run, and were **not re-executed** for this repository.

## Workspace layout on the cluster

`code/` mirrors four working directories. The scripts use absolute paths into them, so a fresh machine needs the same layout, or the path constants at the top of each script edited.

| `code/` folder | working directory | contents |
|---|---|---|
| `viewcrafter_orbits/` | `~/viewcrafter_orbits` (a symlink to the original directory) | ViewCrafter runs, VGGT, SLAM-point scoring, fitters (`experiment1`–`10`, `shared/`) |
| `seva_orbits/` | `~/stable_virtual_camera` | SEVA env setup, `exp1`–`exp5` |
| `kv_tracker/` | `~/kv_tracker_testing` | KV-Tracker env setup, TUM check |
| `generated_to_shaper/` | `~/seva_to_shaper` | C2, D1–D3 |
| `pi3/` | `~/pi3` | standalone π³ setup and smoke test |

Upstream repositories are cloned under each workspace's `third_party/` at the commits in [environments](environments.md#upstream-versions). ShapeR, its weights and the benchmark data (`~/shapeR_testing/{ShapeR,ckpts,data}`) come from ShapeR's own instructions. `slurm/env.sh` in that workspace (sourced by the ShapeR jobs) is **not** part of this snapshot.

## Pipeline per experiment (as recorded)

| exp | order of jobs | rough cost |
|---|---|---|
| A1 | `experiment2/slurm/build_views.sbatch` → `run_orbits.sbatch`; `experiment5/slurm/run_paths.sbatch`; `experiment6/…/run_sparse.sbatch`; `experiment8/…/run_orbits.sbatch`; `experiment9/…/run_paths.sbatch` | ~130 s per 25-frame orbit; sparse 8-view ~16 min per object |
| A2 | `exp1/slurm/build_scenes.sbatch` → `run_seva.sbatch` → `score.sbatch` | ~4 min build, ~50 min for 5 × 80 frames |
| B1 | `exp2/slurm/build.sbatch` → `run_seva.sbatch` → `score.sbatch` | as A2, ×2 |
| C1 | `experiment3/slurm/build_frames.sbatch` → `run_vggt.sbatch` → `score.sbatch`; `experiment4/slurm/compare.sbatch`; `exp4/slurm/pi3_gpu.sbatch`; `exp5/slurm/track_gpu.sbatch`, `track_kf1_gpu.sbatch` → `score_cpu.sbatch` | VGGT < 10 s per set; KV kf1 ~4 min per object |
| C2 | `seva_to_shaper/slurm/01_build_scenes` → `02_run_seva` → `03_reconstruct` → `07_seeded` (+ `09_combined` for gap/union, `30_kv_subset`, `31_kv_score`) | ShapeR ~40 s per object per seed |
| D1 | `08_views.sbatch`, `13_nviews.sbatch` | |
| D2 | `20_build_full` → `21_seva_subset` → `23_subset_shaper` (2 shards) | 155 × 2 conditions × 3 seeds = 930 ShapeR runs |
| D3 | `40_sel_smoke` → `41_sel_pilot` | 16 × 5 × 3 = 240 runs |

## Cluster traps that affect reproduction

- `/home` quota, not free space, is the limit. Jobs fail silently when it fills. ShapeR pkls are ~120 MB each, so the large runs build, score and delete per object.
- Hugging Face cache must be on shared storage (`HF_HOME`), or gated SEVA weights return a 401 that looks like revoked access.
- `demo.py` in SEVA builds the VAE on `cuda:0` at import, so even `--help` needs a GPU and the weights.
- Some GPU nodes hand out a device CUDA cannot open; the job scripts allocate a tensor first and requeue on failure.

## Not yet verified

- That the pipeline runs end to end on a fresh machine from this snapshot alone.
- The ShapeR-side helper scripts sourced from outside `code/` (`env.sh`, the seeded runner's original in another workspace). `run_shaper_nviews.py` is the copy actually used for D1–D3.
- The exact environment files: only Python/PyTorch/CUDA versions are recorded, not full lockfiles.
