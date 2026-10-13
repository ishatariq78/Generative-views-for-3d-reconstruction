#!/bin/bash
#SBATCH --job-name=kvt-io
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
# [STEP 3] input/output test: TUM fr1_desk camera-only, headless, WITH --export_pcd
#          (main.py locally patched so pixels_to_viz exists without --rerun),
#          then score vs GT.               sbatch slurm/io_tum_desk.sbatch
set -eo pipefail
P=/home/isha.tariq/kv_tracker_testing
PY=/home/isha.tariq/.conda/envs/kvtracker/bin/python
export TMPDIR=/tmp
cd "$P/third_party/kv_tracker"
echo "host: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
# timeout: a raise in the main loop leaves the data-stream child alive → job hangs.
timeout 1200 $PY main.py "$P/configs/tum_desk_pcd.yaml" --cam_only --resize_dim 308 --export_pcd
echo; echo "### score against ground truth"
$PY "$P/scripts/score_tum.py" "$P/outputs/tum_desk_pcd" datasets/tum_rgbd/rgbd_dataset_freiburg1_desk
echo; echo "### outputs"; ls -la "$P/outputs/tum_desk_pcd"
