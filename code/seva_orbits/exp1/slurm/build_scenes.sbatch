#!/bin/bash
#SBATCH --job-name=exp1-build
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp1/logs/%x-%j.out

# [STEP 1] Cut 8 posed, background-removed views of each object and attach a 360 deg
#          target orbit.
#   sbatch exp1/slurm/build_scenes.sbatch
#
# GPU needed: the silhouettes come from rasterising each object's mesh through EGL.
# Runs in the `shaper` env (ShapeR's rectifier + pyrender), not `svcam`.
set -eo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
export PYOPENGL_PLATFORM=egl
mkdir -p "$XDG_RUNTIME_DIR"

echo "host: $(hostname)"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/stable_virtual_camera/exp1/scripts/build_scenes.py
