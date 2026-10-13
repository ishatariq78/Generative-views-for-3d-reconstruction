#!/bin/bash
#SBATCH --job-name=exp3-cmp
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
# [STEP 3] SLAM vs SEVA-cloud comparison sheets, one per object -> exp3/figs/
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR" /home/isha.tariq/stable_virtual_camera/exp3/figs
/home/isha.tariq/.conda/envs/shaper/bin/python /home/isha.tariq/stable_virtual_camera/exp3/scripts/render_comparison_seva.py
