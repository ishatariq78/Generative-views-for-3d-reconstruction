#!/bin/bash
#SBATCH --job-name=kvt-check
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/kv_tracker_testing/logs/%x-%j.out
# [STEP 1] env works on a real GPU:   sbatch slurm/check_env.sbatch
set -eo pipefail
export TMPDIR=/tmp
/home/isha.tariq/.conda/envs/kvtracker/bin/python /home/isha.tariq/kv_tracker_testing/scripts/check_env.py
