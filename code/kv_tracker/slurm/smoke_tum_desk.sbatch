#!/bin/bash
#SBATCH --job-name=kvt-tum
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
# [STEP 2] track TUM fr1_desk (573 frames) camera-only, headless, then score vs
#          ground truth with evo.      sbatch slurm/smoke_tum_desk.sbatch
# No --rerun: the viewer needs a display. And therefore no --export_pcd either:
# that block reads `pixels_to_viz`, which upstream only defines inside
# `if args.rerun:` (main.py:675), so the two flags are coupled. Poses are what the
# TUM benchmark scores in any case.
# Paths in the config are relative to the repo root (SAM2 checkpoint, datasets/),
# so cd there.
set -eo pipefail
P=/home/isha.tariq/kv_tracker_testing
PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
export TMPDIR=/tmp
cd "$P/third_party/kv_tracker"
echo "host: $(hostname)"
# `timeout` because the consumer runs `while data_proc.is_alive()`: if the main
# loop raises, the still-running data-stream child keeps the job alive until the
# wall clock runs out. An upstream NameError once burnt a whole hour that way.
timeout 1800 $PY main.py "$P/configs/tum_desk.yaml" --cam_only --resize_dim 308
echo; echo "### score against ground truth"
$PY "$P/scripts/score_tum.py" "$P/outputs/tum_desk" datasets/tum_rgbd/rgbd_dataset_freiburg1_desk
