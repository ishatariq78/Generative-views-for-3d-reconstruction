#!/bin/bash
#SBATCH --job-name=exp4-slamfig
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
# SLAM-referenced figures, no mesh: both families.
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; export PYOPENGL_PLATFORM=egl; mkdir -p "$XDG_RUNTIME_DIR"
cd /home/isha.tariq/stable_virtual_camera/exp4
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
$PY scripts/render_slam_reference.py --family pi3
$PY scripts/render_slam_reference.py --family vggt
