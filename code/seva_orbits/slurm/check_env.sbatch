#!/bin/bash
#SBATCH --job-name=svc-check
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/logs/%x-%j.out

# [STEP 1] Does the env work on a real GPU?
#   sbatch slurm/check_env.sbatch
set -eo pipefail
P=/home/isha.tariq/stable_virtual_camera
export TMPDIR=/tmp
/home/isha.tariq/.conda/envs/svcam/bin/python $P/scripts/check_env.py
