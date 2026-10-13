#!/bin/bash
#SBATCH --job-name=exp2-score
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp2/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp2/logs/%x-%j.out
# [STEP 3] score every occluded run against exp1's clean run.
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"
cd /home/isha.tariq/stable_virtual_camera/exp2
/home/isha.tariq/.conda/envs/shaper/bin/python scripts/score_occlusion.py
